#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

import certifi
import httpx
from playwright.async_api import async_playwright, Page, Locator

# ============================================================
#                    GLOBAL CONFIG / LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("three-exchanges")

POLL_SEC = float(os.getenv("POLL_SEC", "15"))  # общий цикл
SHOW_EMPTY_EVERY_SEC = int(os.getenv("SHOW_EMPTY_EVERY_SEC", "60"))

MAX_TRADES = int(os.getenv("LIMIT", "20"))  # общий лимит (и для Rapira/ABCEX); Grinex берёт GRINEX_LIMIT

# ============================================================
#                    PLAYWRIGHT INSTALL (shared)
# ============================================================

def ensure_playwright_browsers() -> None:
    """
    Гарантируем, что нужные браузеры Playwright скачаны.
    Только install, без системных deps.
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
        log.error("playwright CLI not found in PATH. Убедись, что Playwright установлен и доступен как 'playwright'.")
    except Exception as e:
        log.error("Unexpected error while installing Playwright browsers: %s", e)


# ============================================================
#                         RAPIRA SCRAPER
# ============================================================

RAPIRA_URL = "https://rapira.net/exchange/USDT_RUB"
rapira_log = logging.getLogger("rapira")


async def rapira_close_silently(page: Page) -> None:
    try:
        await page.close()
    except Exception:
        pass


async def rapira_accept_cookies_if_any(page: Page) -> None:
    try:
        btn = page.locator("text=Я согласен")
        if await btn.count() > 0:
            rapira_log.info("Found cookies banner, clicking 'Я согласен'...")
            await btn.first.click(timeout=5_000)
        else:
            rapira_log.info("No 'Я согласен' (cookies) button found.")
    except Exception as e:
        rapira_log.info("Ignoring cookies click error: %s", e)


async def rapira_ensure_last_trades_tab(page: Page) -> None:
    candidates = ["Последние сделки", "История сделок", "История"]
    for text in candidates:
        try:
            tab = page.locator(f"text={text}")
            if await tab.count() > 0:
                rapira_log.info("Clicking tab '%s' ...", text)
                await tab.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception as e:
            rapira_log.info("Failed to click tab '%s': %s", text, e)

    rapira_log.info("Last-trades tab not found explicitly, возможно, уже активна.")


async def rapira_poll_for_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    selectors = [
        "div.table-responsive.table-orders table.table-row-dashed tbody tr.table-orders-row",
        "div.table-responsive.table-orders table.table-row-dashed tbody tr",
        "div.table-responsive.table-orders table tbody tr.table-orders-row",
        "div.table-responsive.table-orders table tbody tr",
        "table.table-row-dashed tbody tr.table-orders-row",
        "table.table-row-dashed tbody tr",
        "tr.table-orders-row",
    ]

    for i in range(max_wait_seconds):
        for selector in selectors:
            rows = await page.query_selector_all(selector)
            if rows:
                rapira_log.info(
                    "Found %d rows using selector '%s' on attempt %d/%d.",
                    len(rows),
                    selector,
                    i + 1,
                    max_wait_seconds,
                )
                return rows

        rapira_log.info("No trade rows found yet (attempt %d/%d), waiting 1 second...", i + 1, max_wait_seconds)
        await page.wait_for_timeout(1_000)

    rapira_log.warning("No history table rows found by any of the candidate selectors.")
    return []


def rapira_normalize_num(text: str) -> float:
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


async def rapira_parse_trades_from_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            cells = await row.query_selector_all("th, td")
            if len(cells) < 3:
                rapira_log.info("Row %d skipped: has only %d cells (need >= 3).", idx, len(cells))
                continue

            price_text = (await cells[0].inner_text()).strip()
            volume_text = (await cells[1].inner_text()).strip()
            time_text = (await cells[2].inner_text()).strip()

            if not price_text or not volume_text or not time_text:
                continue

            try:
                price = rapira_normalize_num(price_text)
                volume = rapira_normalize_num(volume_text)
            except Exception:
                continue

            trades.append(
                {
                    "price": price,
                    "volume": volume,   # предполагаем, что это USDT-объём (как у тебя в остальных)
                    "time": time_text,
                    "price_raw": price_text,
                    "volume_raw": volume_text,
                }
            )
        except Exception:
            continue

    return trades


async def scrape_rapira_trades_snapshot(limit: int = MAX_TRADES) -> Dict[str, Any]:
    """
    Снимок Rapira (без бесконечного цикла). Логика парсинга сохранена 1:1.
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
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
            rapira_log.info("Opening Rapira page %s ...", RAPIRA_URL)
            await page.goto(RAPIRA_URL, wait_until="networkidle", timeout=60_000)
            await page.wait_for_timeout(5_000)

            await rapira_accept_cookies_if_any(page)
            await rapira_ensure_last_trades_tab(page)
            await page.wait_for_timeout(3_000)

            rows = await rapira_poll_for_trade_rows(page, max_wait_seconds=40)
            if not rows:
                return {"exchange": "rapira", "symbol": "USDT/RUB", "count": 0, "trades": []}

            trades = await rapira_parse_trades_from_rows(rows)
            trades = trades[:limit]
            return {"exchange": "rapira", "symbol": "USDT/RUB", "count": len(trades), "trades": trades}
        finally:
            await rapira_close_silently(page)
            await context.close()
            await browser.close()


def rapira_trade_key(t: Dict[str, Any]) -> Tuple:
    return (round(float(t["price"]), 8), round(float(t["volume"]), 8), str(t["time"]).strip())


def rapira_metrics(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"count": 0, "sum_qty_usdt": 0.0, "turnover_rub": 0.0, "vwap": None}

    sum_qty = 0.0
    turnover = 0.0
    for t in trades:
        qty = float(t["volume"])
        price = float(t["price"])
        sum_qty += qty
        turnover += qty * price

    vwap = (turnover / sum_qty) if sum_qty > 0 else None
    return {"count": len(trades), "sum_qty_usdt": sum_qty, "turnover_rub": turnover, "vwap": vwap}


# ============================================================
#                         GRINEX SCRAPER (API)
# ============================================================

grinex_log = logging.getLogger("grinex")

BASE_URL = "https://grinex.io"
MARKET = os.getenv("GRINEX_MARKET", "usdta7a5")
GRINEX_LIMIT = int(os.getenv("GRINEX_LIMIT", str(MAX_TRADES)))

TRADES_URL = f"{BASE_URL}/api/v2/trades?market={MARKET}&limit={GRINEX_LIMIT}&order_by=desc"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": f"{BASE_URL}/trading/{MARKET}?lang=ru",
}

KALININGRAD_TZ = timezone(timedelta(hours=2))


@dataclass(frozen=True)
class GrinexTrade:
    price: float
    amount: float  # base (USDT)
    total: float   # quote (A7A5)
    ts_utc: datetime
    tid: Optional[str] = None
    side: Optional[str] = None


def grinex_as_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(str(x).replace(",", ".").replace(" ", ""))
    except Exception:
        return None


def grinex_parse_ts(obj: Dict[str, Any]) -> datetime:
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

    return datetime.now(timezone.utc)


def grinex_parse_side(obj: Dict[str, Any]) -> Optional[str]:
    for k in ("side", "taker_type", "type", "order_type"):
        v = obj.get(k)
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("buy", "sell"):
                return s
    return None


def grinex_normalize_trade(t: Dict[str, Any]) -> Optional[GrinexTrade]:
    price = grinex_as_float(t.get("price"))
    amount = grinex_as_float(t.get("amount")) or grinex_as_float(t.get("volume"))
    total = grinex_as_float(t.get("total")) or grinex_as_float(t.get("funds")) or grinex_as_float(t.get("quote_volume"))

    if price is None or amount is None:
        return None
    if total is None:
        total = price * amount

    tid = None
    for k in ("id", "tid", "trade_id"):
        if t.get(k) is not None:
            tid = str(t.get(k))
            break

    return GrinexTrade(
        price=price,
        amount=amount,
        total=total,
        ts_utc=grinex_parse_ts(t),
        tid=tid,
        side=grinex_parse_side(t),
    )


def grinex_extract_trades(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ("trades", "data", "result"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def grinex_trade_key(tr: GrinexTrade) -> Tuple:
    if tr.tid:
        return ("id", tr.tid)
    return ("pats", round(tr.price, 8), round(tr.amount, 8), int(tr.ts_utc.timestamp()))


async def grinex_fetch_trades_snapshot(client: httpx.AsyncClient) -> List[GrinexTrade]:
    r = await client.get(TRADES_URL)
    grinex_log.info("HTTP %s %s", r.status_code, TRADES_URL)
    r.raise_for_status()

    payload = r.json()
    raw = grinex_extract_trades(payload)

    trades: List[GrinexTrade] = []
    for item in raw:
        tr = grinex_normalize_trade(item)
        if tr:
            trades.append(tr)

    trades.sort(key=lambda x: x.ts_utc, reverse=True)
    return trades[:GRINEX_LIMIT]


def grinex_metrics(trades: List[GrinexTrade]) -> Dict[str, Any]:
    if not trades:
        return {"count": 0, "sum_qty_usdt": 0.0, "turnover_rub": 0.0, "vwap": None}

    sum_usdt = sum(t.amount for t in trades)
    turnover = sum(t.total for t in trades)  # A7A5 ~ RUB
    vwap = (sum(t.price * t.amount for t in trades) / sum_usdt) if sum_usdt > 0 else None
    return {"count": len(trades), "sum_qty_usdt": sum_usdt, "turnover_rub": turnover, "vwap": vwap}


# ============================================================
#                         ABCEX SCRAPER
# ============================================================

ABCEX_URL = "https://abcex.io/client/spot/USDTRUB"
STATE_PATH = "abcex_state.json"
abcex_log = logging.getLogger("abcex")


def abcex_normalize_num(text: str) -> float:
    t = text.strip().replace("\xa0", " ")
    t = t.replace(" ", "")
    if "," in t and "." in t:
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    return float(t)


async def abcex_accept_cookies_if_any(page: Page) -> None:
    candidates = ["Принять", "Согласен", "Я согласен", "Accept", "I agree"]
    for txt in candidates:
        try:
            btn = page.locator(f"text={txt}")
            if await btn.count() > 0 and await btn.first.is_visible():
                await btn.first.click(timeout=5_000)
                abcex_log.info("Cookies banner handled via: text=%s", txt)
                await page.wait_for_timeout(800)
                return
        except Exception:
            pass
    abcex_log.info("No cookies banner button found (ok).")


def abcex_looks_like_time(s: str) -> bool:
    return bool(re.fullmatch(r"\d{2}:\d{2}:\d{2}", s.strip()))


@dataclass
class AbcexTrade:
    price: float
    qty: float
    time: str
    side: Optional[str]
    price_raw: str
    qty_raw: str


async def abcex_is_login_visible(page: Page) -> bool:
    try:
        pw = page.locator("input[type='password']")
        if await pw.count() > 0 and await pw.first.is_visible():
            return True
    except Exception:
        pass
    return False


async def abcex_login_if_needed(page: Page, email: str, password: str) -> None:
    if not await abcex_is_login_visible(page):
        abcex_log.info("Login not required (already in session).")
        return

    abcex_log.info("Login detected. Performing sign-in ...")

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
        raise RuntimeError("ABCEX: не смог найти поля email/password.")

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
        await page.wait_for_timeout(1_000)
        await page.wait_for_load_state("networkidle", timeout=30_000)
    except Exception:
        pass

    await page.wait_for_timeout(2_500)

    if await abcex_is_login_visible(page):
        raise RuntimeError("ABCEX: логин не прошёл (форма всё ещё видна).")

    abcex_log.info("Login successful (form disappeared).")


async def abcex_click_trades_tab_best_effort(page: Page) -> None:
    candidates = ["Сделки", "История", "Order history", "Trades"]
    for t in candidates:
        try:
            tab = page.locator(f"[role='tab']:has-text('{t}')")
            if await tab.count() > 0 and await tab.first.is_visible():
                await tab.first.click(timeout=8_000)
                await page.wait_for_timeout(800)
                abcex_log.info("Clicked tab: %s", t)
                return
        except Exception:
            continue

    for t in candidates:
        try:
            tab = page.locator(f"text={t}")
            if await tab.count() > 0 and await tab.first.is_visible():
                await tab.first.click(timeout=8_000)
                await page.wait_for_timeout(800)
                abcex_log.info("Clicked text tab: %s", t)
                return
        except Exception:
            continue

    abcex_log.info("Trades tab click skipped (not found explicitly).")


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

    raise RuntimeError("ABCEX: panel-orderHistory есть, но не видимая.")


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
                abcex_log.info("Trades look visible (time cells detected).")
                return
        except Exception:
            pass
        await page.wait_for_timeout(800)

    raise RuntimeError("ABCEX: не дождался появления сделок (HH:MM:SS).")


async def abcex_extract_trades_from_panel(panel: Locator, limit: int = MAX_TRADES) -> List[AbcexTrade]:
    handle = await panel.element_handle()
    if handle is None:
        raise RuntimeError("ABCEX: не смог получить element_handle панели.")

    raw_rows: List[Dict[str, Any]] = await handle.evaluate(
        """(root, limit) => {
          const isTime = (s) => /^\\d{2}:\\d{2}:\\d{2}$/.test((s||'').trim());
          const isNum = (s) => /^[0-9][0-9\\s\\u00A0.,]*$/.test((s||'').trim());
          const out = [];
          const grids = Array.from(root.querySelectorAll('div'));

          for (const g of grids) {
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

    trades: List[AbcexTrade] = []
    for r in raw_rows:
        try:
            price_raw = str(r.get("price_raw", "")).strip()
            qty_raw = str(r.get("qty_raw", "")).strip()
            time_txt = str(r.get("time", "")).strip()
            side = r.get("side", None)

            if not price_raw or not qty_raw or not time_txt:
                continue
            if not abcex_looks_like_time(time_txt):
                continue

            price = abcex_normalize_num(price_raw)
            qty = abcex_normalize_num(qty_raw)

            trades.append(
                AbcexTrade(
                    price=price,
                    qty=qty,
                    time=time_txt,
                    side=side if side in ("buy", "sell") else None,
                    price_raw=price_raw,
                    qty_raw=qty_raw,
                )
            )
        except Exception:
            continue

    return trades


def abcex_metrics(trades: List[AbcexTrade]) -> Dict[str, Any]:
    if not trades:
        return {"count": 0, "sum_qty_usdt": 0.0, "turnover_rub": 0.0, "vwap": None}

    sum_qty = sum(t.qty for t in trades)
    turnover = sum(t.qty * t.price for t in trades)
    vwap = (turnover / sum_qty) if sum_qty > 0 else None
    return {"count": len(trades), "sum_qty_usdt": sum_qty, "turnover_rub": turnover, "vwap": vwap}


def abcex_trade_key(t: AbcexTrade) -> Tuple:
    return (round(t.price, 8), round(t.qty, 8), t.time.strip())


async def scrape_abcex_trades_snapshot(
    email: str,
    password: str,
    headless: bool = True,
    limit: int = MAX_TRADES,
) -> Dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        storage_state = STATE_PATH if os.path.exists(STATE_PATH) else None
        if storage_state:
            abcex_log.info("Using saved session state: %s", storage_state)

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
            abcex_log.info("Opening ABCEX: %s", ABCEX_URL)
            await page.goto(ABCEX_URL, wait_until="networkidle", timeout=60_000)
            await page.wait_for_timeout(2_000)

            await abcex_accept_cookies_if_any(page)

            if await abcex_is_login_visible(page):
                await abcex_login_if_needed(page, email=email, password=password)

            try:
                await context.storage_state(path=STATE_PATH)
                abcex_log.info("Saved session state to %s", STATE_PATH)
            except Exception as e:
                abcex_log.info("Could not save storage state: %s", e)

            await abcex_click_trades_tab_best_effort(page)
            await abcex_wait_trades_visible(page)

            panel = await abcex_get_order_history_panel(page)
            trades = await abcex_extract_trades_from_panel(panel, limit=limit)

            return {
                "exchange": "abcex",
                "symbol": "USDT/RUB",
                "url": ABCEX_URL,
                "count": len(trades),
                "trades": [
                    {
                        "price": t.price,
                        "qty": t.qty,
                        "time": t.time,
                        "side": t.side,
                        "price_raw": t.price_raw,
                        "qty_raw": t.qty_raw,
                    }
                    for t in trades
                ],
            }
        finally:
            try:
                await page.close()
            except Exception:
                pass
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass


# ============================================================
#                    ORCHESTRATOR (3 exchanges)
# ============================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pick_best(ex_stats: Dict[str, Dict[str, Any]], key: str) -> Optional[str]:
    best_name = None
    best_val = None
    for name, st in ex_stats.items():
        v = st.get(key)
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_name = name
    return best_name


async def main() -> None:
    log.info("Starting unified 3-exchange scraper ...")
    ensure_playwright_browsers()

    abcex_email = os.getenv("ABCEX_EMAIL")
    abcex_password = os.getenv("ABCEX_PASSWORD")
    abcex_headless = os.getenv("ABCEX_HEADLESS", "1").strip() not in ("0", "false", "False")

    # На Render нельзя спрашивать пароль в консоли — сразу валим с понятной ошибкой
    if not abcex_email or not abcex_password:
        raise RuntimeError(
            "ENV не настроены: требуется ABCEX_EMAIL и ABCEX_PASSWORD (для Render обязательно)."
        )

    seen_rapira: set[Tuple] = set()
    seen_grinex: set[Tuple] = set()
    seen_abcex: set[Tuple] = set()

    initialized = False

    last_empty_log = 0.0

    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=15,
        follow_redirects=True,
        verify=certifi.where(),
    ) as client:
        while True:
            cycle_started = asyncio.get_event_loop().time()

            try:
                # Параллельный сбор (ускоряет цикл)
                rapira_task = asyncio.create_task(scrape_rapira_trades_snapshot(limit=MAX_TRADES))
                grinex_task = asyncio.create_task(grinex_fetch_trades_snapshot(client))
                abcex_task = asyncio.create_task(
                    scrape_abcex_trades_snapshot(
                        email=abcex_email,
                        password=abcex_password,
                        headless=abcex_headless,
                        limit=MAX_TRADES,
                    )
                )

                rapira_res, grinex_trades, abcex_res = await asyncio.gather(
                    rapira_task, grinex_task, abcex_task
                )

                # ---------------- Rapira: INIT/NEW ----------------
                rapira_trades = rapira_res.get("trades", [])
                rapira_new = []
                if not initialized:
                    for t in rapira_trades:
                        seen_rapira.add(rapira_trade_key(t))
                else:
                    for t in rapira_trades:
                        k = rapira_trade_key(t)
                        if k not in seen_rapira:
                            rapira_new.append(t)
                        seen_rapira.add(k)
                    if len(seen_rapira) > 5000:
                        seen_rapira = {rapira_trade_key(t) for t in rapira_trades}

                # ---------------- Grinex: INIT/NEW ----------------
                grinex_new: List[GrinexTrade] = []
                if not initialized:
                    for t in grinex_trades:
                        seen_grinex.add(grinex_trade_key(t))
                else:
                    for t in grinex_trades:
                        k = grinex_trade_key(t)
                        if k not in seen_grinex:
                            grinex_new.append(t)
                        seen_grinex.add(k)
                    if len(seen_grinex) > 5000:
                        seen_grinex = {grinex_trade_key(t) for t in grinex_trades}

                # ---------------- ABCEX: INIT/NEW ----------------
                abcex_trades_raw = abcex_res.get("trades", [])
                abcex_trades: List[AbcexTrade] = []
                for t in abcex_trades_raw:
                    abcex_trades.append(
                        AbcexTrade(
                            price=float(t["price"]),
                            qty=float(t["qty"]),
                            time=str(t["time"]),
                            side=t.get("side"),
                            price_raw=str(t.get("price_raw", t["price"])),
                            qty_raw=str(t.get("qty_raw", t["qty"])),
                        )
                    )

                abcex_new: List[AbcexTrade] = []
                if not initialized:
                    for t in abcex_trades:
                        seen_abcex.add(abcex_trade_key(t))
                else:
                    for t in abcex_trades:
                        k = abcex_trade_key(t)
                        if k not in seen_abcex:
                            abcex_new.append(t)
                        seen_abcex.add(k)
                    if len(seen_abcex) > 5000:
                        seen_abcex = {abcex_trade_key(t) for t in abcex_trades}

                # ---------------- Метрики ----------------
                rapira_stats_snapshot = rapira_metrics(rapira_trades)
                rapira_stats_new = rapira_metrics(rapira_new)

                grinex_stats_snapshot = grinex_metrics(grinex_trades)
                grinex_stats_new = grinex_metrics(grinex_new)

                abcex_stats_snapshot = abcex_metrics(abcex_trades)
                abcex_stats_new = abcex_metrics(abcex_new)

                exchanges_snapshot = {
                    "rapira": rapira_stats_snapshot,
                    "grinex": grinex_stats_snapshot,
                    "abcex": abcex_stats_snapshot,
                }
                exchanges_new = {
                    "rapira": rapira_stats_new,
                    "grinex": grinex_stats_new,
                    "abcex": abcex_stats_new,
                }

                best_turnover = _pick_best(exchanges_new, "turnover_rub") or _pick_best(exchanges_snapshot, "turnover_rub")
                best_volume = _pick_best(exchanges_new, "sum_qty_usdt") or _pick_best(exchanges_snapshot, "sum_qty_usdt")

                # Спред VWAP между биржами (по snapshot)
                vwaps = [v.get("vwap") for v in exchanges_snapshot.values() if v.get("vwap") is not None]
                vwap_spread = (max(vwaps) - min(vwaps)) if vwaps else None

                payload = {
                    "ts_utc": _utc_now_iso(),
                    "market": "USDT/RUB",
                    "urls": {
                        "rapira": RAPIRA_URL,
                        "grinex": TRADES_URL,
                        "abcex": ABCEX_URL,
                    },
                    "snapshot": exchanges_snapshot,
                    "new_since_last_poll": exchanges_new,
                    "comparison": {
                        "best_turnover_rub": best_turnover,
                        "best_volume_usdt": best_volume,
                        "vwap_spread_snapshot": vwap_spread,
                    },
                    # по желанию можно включить новые сделки (удобно, но увеличит логи)
                    "new_trades": {
                        "rapira": rapira_new[:MAX_TRADES],
                        "grinex": [
                            {
                                "price": t.price,
                                "amount": t.amount,
                                "total": t.total,
                                "ts_utc": t.ts_utc.isoformat(),
                                "side": t.side,
                                "id": t.tid,
                            }
                            for t in grinex_new[:MAX_TRADES]
                        ],
                        "abcex": [
                            {
                                "price": t.price,
                                "qty": t.qty,
                                "time": t.time,
                                "side": t.side,
                            }
                            for t in abcex_new[:MAX_TRADES]
                        ],
                    },
                }

                if not initialized:
                    log.info("INIT completed (3 exchanges). Printing first snapshot JSON ...")
                    initialized = True
                else:
                    if rapira_new or grinex_new or abcex_new:
                        log.info(
                            "NEW: rapira=%d grinex=%d abcex=%d | best_turnover=%s best_volume=%s",
                            len(rapira_new), len(grinex_new), len(abcex_new),
                            best_turnover, best_volume
                        )
                    else:
                        now = asyncio.get_event_loop().time()
                        if now - last_empty_log >= SHOW_EMPTY_EVERY_SEC:
                            log.info("No new trades on all exchanges.")
                            last_empty_log = now

                print(json.dumps(payload, ensure_ascii=False, indent=2))

            except Exception as e:
                log.warning("Cycle failed: %s", e)

            # выдерживаем POLL_SEC с учётом времени цикла
            elapsed = asyncio.get_event_loop().time() - cycle_started
            sleep_for = max(0.0, POLL_SEC - elapsed)
            await asyncio.sleep(sleep_for)


if __name__ == "__main__":
    asyncio.run(main())
