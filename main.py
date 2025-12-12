#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import certifi
import httpx
from playwright.async_api import async_playwright, Browser, Page, Locator

# ──────────────────────────── CONFIG ────────────────────────────

SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

# Интервал опроса (сек)
POLL_SEC = float(os.getenv("POLL_SEC", "10"))

# Сколько последних сделок читать на каждом опросе
LIMIT = int(os.getenv("LIMIT", "20"))

# Логи: как часто писать "нет новых сделок", чтобы не спамить
SHOW_EMPTY_EVERY_SEC = int(os.getenv("SHOW_EMPTY_EVERY_SEC", "60"))

# Playwright: ставить браузеры при старте (на Render лучше ставить в build, см. ниже)
PW_INSTALL_ON_START = os.getenv("PW_INSTALL_ON_START", "0") in ("1", "true", "True")

# Headless
HEADLESS = os.getenv("HEADLESS", "1") not in ("0", "false", "False")

# URLs
RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB")
ABCEX_URL = os.getenv("ABCEX_URL", "https://abcex.io/client/spot/USDTRUB")
ABCEX_STATE_PATH = os.getenv("ABCEX_STATE_PATH", "abcex_state.json")

# Grinex
GRINEX_BASE = os.getenv("GRINEX_BASE", "https://grinex.io")
GRINEX_MARKET = os.getenv("GRINEX_MARKET", "usdta7a5")
GRINEX_URL = f"{GRINEX_BASE}/api/v2/trades?market={GRINEX_MARKET}&limit={LIMIT}&order_by=desc"

GRINEX_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": f"{GRINEX_BASE}/trading/{GRINEX_MARKET}?lang=ru",
}

# ABCEX creds (только через env на Render)
ABCEX_EMAIL = os.getenv("ABCEX_EMAIL")
ABCEX_PASSWORD = os.getenv("ABCEX_PASSWORD")

# ──────────────────────────── LOGGING ────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("trades-worker")

# глушим шум httpx
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ──────────────────────────── MODELS ────────────────────────────

@dataclass(frozen=True)
class Trade:
    exchange: str
    price: float
    qty: float               # базовая валюта (USDT)
    time_raw: str            # как на сайте/API (может быть без даты)
    side: Optional[str] = None
    tid: Optional[str] = None
    ts_utc: Optional[datetime] = None  # если есть (например, Grinex)


@dataclass
class DeltaMetrics:
    new_count: int
    sum_qty: float
    turnover: float
    vwap: Optional[float]


# ──────────────────────────── COMMON UTILS ────────────────────────────

def ensure_playwright_browsers() -> None:
    """
    Ставит браузеры Playwright. На Render лучше делать в build command,
    но оставляем опцию PW_INSTALL_ON_START=1.
    """
    try:
        log.info("Ensuring Playwright Chromium is installed ...")
        result = subprocess.run(
            ["playwright", "install", "chromium", "chromium-headless-shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            log.info("Playwright Chromium is installed (or already present).")
        else:
            log.error(
                "playwright install returned code %s\nSTDOUT:\n%s\nSTDERR:\n%s",
                result.returncode,
                result.stdout,
                result.stderr,
            )
    except FileNotFoundError:
        log.error("playwright CLI not found in PATH.")
    except Exception as e:
        log.error("Unexpected error while installing Playwright browsers: %s", e)


def normalize_num(text: str) -> float:
    t = text.strip().replace("\xa0", " ").replace(" ", "")
    if "," in t and "." in t:
        # "1,234.56" -> 1234.56
        t = t.replace(",", "")
    else:
        # "1234,56" -> 1234.56
        t = t.replace(",", ".")
    return float(t)


def looks_like_time_hms(s: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}:\d{2}", s.strip()))


def trade_key(t: Trade) -> Tuple:
    # Сначала стабильный id (если есть)
    if t.tid:
        return ("id", t.exchange, t.tid)
    # Иначе — композитный ключ
    # (time_raw оставляем как есть: для dedupe на коротком горизонте достаточно)
    return ("pqt", t.exchange, round(t.price, 8), round(t.qty, 8), t.time_raw.strip())


def compute_delta_metrics(trades: List[Trade]) -> DeltaMetrics:
    if not trades:
        return DeltaMetrics(new_count=0, sum_qty=0.0, turnover=0.0, vwap=None)
    sum_qty = sum(t.qty for t in trades)
    turnover = sum(t.price * t.qty for t in trades)
    vwap = (turnover / sum_qty) if sum_qty > 0 else None
    return DeltaMetrics(new_count=len(trades), sum_qty=sum_qty, turnover=turnover, vwap=vwap)


async def accept_cookies_best_effort(page: Page) -> None:
    candidates = ["Принять", "Согласен", "Я согласен", "Accept", "I agree"]
    for txt in candidates:
        try:
            btn = page.locator(f"text={txt}")
            if await btn.count() > 0 and await btn.first.is_visible():
                await btn.first.click(timeout=5_000)
                await page.wait_for_timeout(600)
                return
        except Exception:
            continue


# ──────────────────────────── RAPIRA (Playwright) ────────────────────────────

async def rapira_fetch(browser: Browser) -> List[Trade]:
    context = await browser.new_context(
        viewport={"width": 1440, "height": 810},
        locale="ru-RU",
        timezone_id="Europe/Moscow",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    page = await context.new_page()

    try:
        await page.goto(RAPIRA_URL, wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(3_000)
        await accept_cookies_best_effort(page)

        # включаем вкладку "Последние сделки" / "История сделок" / "История"
        for text in ("Последние сделки", "История сделок", "История"):
            try:
                tab = page.locator(f"text={text}")
                if await tab.count() > 0 and await tab.first.is_visible():
                    await tab.first.click(timeout=5_000)
                    await page.wait_for_timeout(1_000)
                    break
            except Exception:
                continue

        # ищем строки таблицы
        selectors = [
            "div.table-responsive.table-orders table.table-row-dashed tbody tr.table-orders-row",
            "div.table-responsive.table-orders table.table-row-dashed tbody tr",
            "div.table-responsive.table-orders table tbody tr.table-orders-row",
            "div.table-responsive.table-orders table tbody tr",
            "table.table-row-dashed tbody tr.table-orders-row",
            "table.table-row-dashed tbody tr",
            "tr.table-orders-row",
        ]

        rows = []
        for _ in range(40):
            for sel in selectors:
                rows = await page.query_selector_all(sel)
                if rows:
                    break
            if rows:
                break
            await page.wait_for_timeout(1_000)

        trades: List[Trade] = []
        for row in rows[:LIMIT]:
            cells = await row.query_selector_all("th, td")
            if len(cells) < 3:
                continue
            price_raw = (await cells[0].inner_text()).strip()
            qty_raw = (await cells[1].inner_text()).strip()
            time_raw = (await cells[2].inner_text()).strip()
            if not price_raw or not qty_raw or not time_raw:
                continue
            try:
                price = normalize_num(price_raw)
                qty = normalize_num(qty_raw)
            except Exception:
                continue

            trades.append(
                Trade(
                    exchange="rapira",
                    price=price,
                    qty=qty,
                    time_raw=time_raw,
                )
            )
        return trades
    finally:
        try:
            await page.close()
        except Exception:
            pass
        try:
            await context.close()
        except Exception:
            pass


# ──────────────────────────── GRINEX (HTTPX) ────────────────────────────

def _as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).replace(",", ".").replace(" ", ""))
    except Exception:
        return None


def _parse_ts(obj: Dict[str, Any]) -> Optional[datetime]:
    for k in ("created_at", "timestamp", "ts", "time", "at", "date"):
        v = obj.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)):
            sec = float(v) / 1000.0 if v > 10_000_000_000 else float(v)
            return datetime.fromtimestamp(sec, tz=timezone.utc)
        if isinstance(v, str):
            s = v.strip()
            try:
                if s.endswith("Z"):
                    s = s[:-1] + "+00:00"
                return datetime.fromisoformat(s).astimezone(timezone.utc)
            except Exception:
                pass
    return None


def _extract_trades(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ("trades", "data", "result"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


async def grinex_fetch(client: httpx.AsyncClient) -> List[Trade]:
    r = await client.get(GRINEX_URL)
    r.raise_for_status()
    payload = r.json()
    raw = _extract_trades(payload)

    out: List[Trade] = []
    for t in raw[:LIMIT]:
        price = _as_float(t.get("price"))
        amount = _as_float(t.get("amount")) or _as_float(t.get("volume"))
        if price is None or amount is None:
            continue

        tid = None
        for k in ("id", "tid", "trade_id"):
            if t.get(k) is not None:
                tid = str(t.get(k))
                break

        side = None
        v = t.get("side") or t.get("taker_type") or t.get("type")
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("buy", "sell"):
                side = s

        ts = _parse_ts(t)

        # time_raw: делаем красивый и стабильный
        time_raw = (
            ts.strftime("%Y-%m-%d %H:%M:%S") if ts else (str(t.get("created_at") or t.get("time") or ""))
        )

        out.append(
            Trade(
                exchange="grinex",
                price=price,
                qty=amount,
                time_raw=time_raw,
                side=side,
                tid=tid,
                ts_utc=ts,
            )
        )

    # сортируем "свежие сверху", если есть ts
    out.sort(key=lambda x: x.ts_utc.timestamp() if x.ts_utc else 0, reverse=True)
    return out


# ──────────────────────────── ABCEX (Playwright) ────────────────────────────

async def abcex_is_login_visible(page: Page) -> bool:
    try:
        pw = page.locator("input[type='password']")
        return (await pw.count() > 0) and (await pw.first.is_visible())
    except Exception:
        return False


async def abcex_login_if_needed(page: Page, email: str, password: str) -> None:
    if not await abcex_is_login_visible(page):
        return

    # поля
    email_candidates = [
        "input[type='email']",
        "input[name='email']",
        "input[placeholder*='mail' i]",
        "input[placeholder*='Email' i]",
        "input[placeholder*='Почта' i]",
        "input[placeholder*='E-mail' i]",
    ]
    pw_candidates = [
        "input[type='password']",
        "input[name='password']",
        "input[placeholder*='Пароль' i]",
        "input[placeholder*='Password' i]",
    ]

    email_filled = False
    for sel in email_candidates:
        loc = page.locator(sel)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                await loc.first.fill(email, timeout=10_000)
                email_filled = True
                break
        except Exception:
            continue

    pw_filled = False
    for sel in pw_candidates:
        loc = page.locator(sel)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                await loc.first.fill(password, timeout=10_000)
                pw_filled = True
                break
        except Exception:
            continue

    if not email_filled or not pw_filled:
        raise RuntimeError("ABCEX: не нашёл поля email/password.")

    # кнопка
    btn_texts = ["Войти", "Вход", "Sign in", "Login", "Войти в аккаунт"]
    clicked = False
    for t in btn_texts:
        try:
            btn = page.locator(f"button:has-text('{t}')")
            if await btn.count() > 0 and await btn.first.is_visible():
                await btn.first.click(timeout=10_000)
                clicked = True
                break
        except Exception:
            continue
    if not clicked:
        try:
            await page.keyboard.press("Enter")
        except Exception:
            pass

    try:
        await page.wait_for_load_state("networkidle", timeout=30_000)
    except Exception:
        pass
    await page.wait_for_timeout(2_500)

    if await abcex_is_login_visible(page):
        raise RuntimeError("ABCEX: логин не прошёл (форма всё ещё видна).")


async def abcex_click_trades_tab_best_effort(page: Page) -> None:
    candidates = ["Сделки", "История", "Order history", "Trades"]
    for t in candidates:
        try:
            tab = page.locator(f"[role='tab']:has-text('{t}')")
            if await tab.count() > 0 and await tab.first.is_visible():
                await tab.first.click(timeout=8_000)
                await page.wait_for_timeout(600)
                return
        except Exception:
            continue


async def abcex_get_order_history_panel(page: Page) -> Locator:
    panel = page.locator("div[role='tabpanel'][id*='panel-orderHistory']")
    cnt = await panel.count()
    if cnt == 0:
        raise RuntimeError("ABCEX: не нашёл panel-orderHistory.")
    for i in range(cnt):
        p = panel.nth(i)
        try:
            if await p.is_visible():
                return p
        except Exception:
            continue
    raise RuntimeError("ABCEX: panel-orderHistory есть, но не видим.")


async def abcex_wait_trades_visible(page: Page, timeout_ms: int = 25_000) -> None:
    start = datetime.utcnow().timestamp()
    while (datetime.utcnow().timestamp() - start) * 1000 < timeout_ms:
        try:
            ok = await page.evaluate(
                """() => {
                    const re = /^\\d{2}:\\d{2}:\\d{2}$/;
                    const ps = Array.from(document.querySelectorAll('p'));
                    return ps.some(p => re.test((p.textContent||'').trim()));
                }"""
            )
            if ok:
                return
        except Exception:
            pass
        await page.wait_for_timeout(800)
    raise RuntimeError("ABCEX: не дождался сделок (HH:MM:SS).")


async def abcex_extract_trades_from_panel(panel: Locator, limit: int = LIMIT) -> List[Trade]:
    handle = await panel.element_handle()
    if handle is None:
        raise RuntimeError("ABCEX: panel element_handle is None.")

    raw_rows: List[Dict[str, Any]] = await handle.evaluate(
        """(root, limit) => {
          const isTime = (s) => /^\\d{2}:\\d{2}:\\d{2}$/.test((s||'').trim());
          const isNum = (s) => /^[0-9][0-9\\s\\u00A0.,]*$/.test((s||'').trim());

          const out = [];
          const divs = Array.from(root.querySelectorAll('div'));

          for (const g of divs) {
            const ps = Array.from(g.querySelectorAll(':scope > p'));
            if (ps.length < 3) continue;

            const t0 = (ps[0].textContent || '').trim();
            const t1 = (ps[1].textContent || '').trim();
            const t2 = (ps[2].textContent || '').trim();

            if (!isTime(t2)) continue;
            if (!isNum(t0) || !isNum(t1)) continue;

            const style0 = (ps[0].getAttribute('style') || '').toLowerCase();
            let side = null;
            if (style0.includes('green')) side = 'buy';
            if (style0.includes('red')) side = 'sell';

            out.push({ price_raw: t0, qty_raw: t1, time: t2, side });
            if (out.length >= limit) break;
          }
          return out;
        }""",
        limit,
    )

    out: List[Trade] = []
    for r in raw_rows:
        price_raw = str(r.get("price_raw", "")).strip()
        qty_raw = str(r.get("qty_raw", "")).strip()
        time_raw = str(r.get("time", "")).strip()
        side = r.get("side", None)

        if not price_raw or not qty_raw or not time_raw:
            continue
        if not looks_like_time_hms(time_raw):
            continue
        try:
            price = normalize_num(price_raw)
            qty = normalize_num(qty_raw)
        except Exception:
            continue

        out.append(
            Trade(
                exchange="abcex",
                price=price,
                qty=qty,
                time_raw=time_raw,
                side=side if side in ("buy", "sell") else None,
            )
        )
    return out


async def abcex_fetch(browser: Browser) -> List[Trade]:
    storage_state = ABCEX_STATE_PATH if os.path.exists(ABCEX_STATE_PATH) else None

    context = await browser.new_context(
        viewport={"width": 1440, "height": 810},
        locale="ru-RU",
        timezone_id="Europe/Moscow",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        storage_state=storage_state,
    )

    page = await context.new_page()
    try:
        await page.goto(ABCEX_URL, wait_until="networkidle", timeout=60_000)
        await page.wait_for_timeout(1_500)
        await accept_cookies_best_effort(page)

        # логин только если надо
        if await abcex_is_login_visible(page):
            if not ABCEX_EMAIL or not ABCEX_PASSWORD:
                raise RuntimeError("ABCEX требует логин: задай ABCEX_EMAIL/ABCEX_PASSWORD в env.")
            await abcex_login_if_needed(page, ABCEX_EMAIL, ABCEX_PASSWORD)

        # сохраним state
        try:
            await context.storage_state(path=ABCEX_STATE_PATH)
        except Exception:
            pass

        await abcex_click_trades_tab_best_effort(page)
        await abcex_wait_trades_visible(page)

        panel = await abcex_get_order_history_panel(page)
        trades = await abcex_extract_trades_from_panel(panel, limit=LIMIT)
        return trades
    finally:
        try:
            await page.close()
        except Exception:
            pass
        try:
            await context.close()
        except Exception:
            pass


# ──────────────────────────── ORCHESTRATOR ────────────────────────────

class ExchangeState:
    def __init__(self, name: str) -> None:
        self.name = name
        self.seen: set[Tuple] = set()
        self.initialized = False
        self.last_empty_log = 0.0

    def mark_seen(self, trades: List[Trade]) -> None:
        for t in trades:
            self.seen.add(trade_key(t))
        # не даём бесконечно расти
        if len(self.seen) > 10000:
            # оставляем только текущий срез
            self.seen = {trade_key(t) for t in trades}

    def delta(self, trades: List[Trade]) -> List[Trade]:
        return [t for t in trades if trade_key(t) not in self.seen]


def format_cmp_line(name: str, m: DeltaMetrics) -> str:
    vwap = "-" if m.vwap is None else f"{m.vwap:.6f}"
    return f"{name:7s} | new={m.new_count:3d} | sum_usdt={m.sum_qty:12.4f} | turnover_rub={m.turnover:14.4f} | vwap={vwap}"


async def run_loop() -> None:
    log.info("Starting unified trades worker. poll=%ss limit=%s symbol=%s", POLL_SEC, LIMIT, SYMBOL)
    log.info("Targets: rapira=%s | grinex=%s | abcex=%s", RAPIRA_URL, GRINEX_URL, ABCEX_URL)

    if PW_INSTALL_ON_START:
        ensure_playwright_browsers()

    # httpx client (grinex)
    async with httpx.AsyncClient(
        headers=GRINEX_HEADERS,
        timeout=15,
        follow_redirects=True,
        verify=certifi.where(),
    ) as http_client:
        # Playwright (rapira + abcex)
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=HEADLESS,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            states = {
                "rapira": ExchangeState("rapira"),
                "grinex": ExchangeState("grinex"),
                "abcex": ExchangeState("abcex"),
            }

            while True:
                started = asyncio.get_event_loop().time()

                try:
                    # параллельно
                    rap_task = rapira_fetch(browser)
                    gr_task = grinex_fetch(http_client)
                    ab_task = abcex_fetch(browser)

                    rap_trades, gr_trades, ab_trades = await asyncio.gather(
                        rap_task, gr_task, ab_task, return_exceptions=True
                    )

                    results: Dict[str, List[Trade]] = {}

                    for name, data in (("rapira", rap_trades), ("grinex", gr_trades), ("abcex", ab_trades)):
                        if isinstance(data, Exception):
                            log.warning("%s fetch failed: %s", name, data)
                            results[name] = []
                        else:
                            results[name] = data

                    # считаем дельту и печатаем сравнение
                    lines: List[str] = []
                    total_new_turnover = 0.0
                    total_new_qty = 0.0

                    for name in ("rapira", "grinex", "abcex"):
                        st = states[name]
                        trades = results[name]

                        if not st.initialized:
                            st.mark_seen(trades)
                            st.initialized = True
                            # INIT: не считаем как дельту (иначе "скачок" в начале)
                            m = DeltaMetrics(0, 0.0, 0.0, None)
                            lines.append(format_cmp_line(name, m) + " | INIT")
                            continue

                        new_trades = st.delta(trades)
                        st.mark_seen(trades)

                        m = compute_delta_metrics(new_trades)
                        total_new_turnover += m.turnover
                        total_new_qty += m.sum_qty

                        now = asyncio.get_event_loop().time()
                        if m.new_count == 0:
                            if now - st.last_empty_log >= SHOW_EMPTY_EVERY_SEC:
                                lines.append(format_cmp_line(name, m))
                                st.last_empty_log = now
                        else:
                            lines.append(format_cmp_line(name, m))

                    if lines:
                        log.info("────────────────────────────────────────────────────────────────────────────")
                        log.info("Δ compare (only NEW trades since last poll)")
                        for ln in lines:
                            log.info(ln)
                        if total_new_qty > 0:
                            total_vwap = total_new_turnover / total_new_qty
                            log.info(
                                "TOTAL   | new_turnover_rub=%.4f | new_sum_usdt=%.4f | total_vwap=%.6f",
                                total_new_turnover, total_new_qty, total_vwap
                            )

                    # JSON line (удобно потом парсить в лог-агрегаторе)
                    snapshot = {
                        "ts_utc": datetime.now(timezone.utc).isoformat(),
                        "symbol": SYMBOL,
                        "poll_sec": POLL_SEC,
                        "limit": LIMIT,
                        "grinex_market": GRINEX_MARKET,
                    }
                    log.info("SNAPSHOT_JSON %s", json.dumps(snapshot, ensure_ascii=False))

                except Exception as e:
                    log.warning("Loop failed: %s", e)

                # выдерживаем интервал
                elapsed = asyncio.get_event_loop().time() - started
                sleep_for = max(0.0, POLL_SEC - elapsed)
                await asyncio.sleep(sleep_for)


def main() -> None:
    # Важно: на Render TTY нет, поэтому creds только env
    if not HEADLESS:
        log.info("HEADLESS disabled: browser will be visible (useful locally).")
    if ABCEX_EMAIL and not ABCEX_PASSWORD:
        log.warning("ABCEX_EMAIL set but ABCEX_PASSWORD missing (ABCeX may fail if login is required).")

    asyncio.run(run_loop())


if __name__ == "__main__":
    main()
