#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import certifi
import httpx
from playwright.async_api import async_playwright, Page, Locator, Frame

# =========================
# ENV (Render)
# =========================
# обязательно:
#   PLAYWRIGHT_BROWSERS_PATH=/opt/render/project/src/.cache/ms-playwright
#
# ABCEX:
#   ABCEX_EMAIL=...
#   ABCEX_PASSWORD=...
#
# опционально:
#   POLL_SEC=10
#   LIMIT=20
#   GRINEX_MARKET=usdta7a5
#   ABCEX_HEADLESS=1
#   PROXY_SERVER=http://host:port
#   PROXY_USERNAME=...
#   PROXY_PASSWORD=...
#
#   RAPIRA_GOTO_WAIT=networkidle|domcontentloaded   (по умолчанию networkidle)
# =========================

POLL_SEC = float(os.getenv("POLL_SEC", "10"))
LIMIT = int(os.getenv("LIMIT", "20"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("three-exchanges")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# =========================
# Playwright install (Render-safe)
# =========================

def ensure_playwright_browsers() -> None:
    if not os.getenv("PLAYWRIGHT_BROWSERS_PATH"):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "/opt/render/project/src/.cache/ms-playwright"

    try:
        log.info(
            "Ensuring Playwright Chromium is installed ... (PLAYWRIGHT_BROWSERS_PATH=%s)",
            os.environ.get("PLAYWRIGHT_BROWSERS_PATH"),
        )

        if shutil.which("playwright"):
            cmd = ["playwright", "install", "chromium", "chromium-headless-shell"]
        else:
            cmd = [sys.executable, "-m", "playwright", "install", "chromium", "chromium-headless-shell"]

        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        if result.returncode == 0:
            log.info("Playwright Chromium is installed (or already present).")
        else:
            log.error(
                "playwright install failed (code=%s)\nCMD: %s\nSTDOUT:\n%s\nSTDERR:\n%s",
                result.returncode,
                " ".join(cmd),
                result.stdout,
                result.stderr,
            )
    except Exception as e:
        log.error("Unexpected error while installing Playwright browsers: %s", e)

def _proxy_config() -> Optional[Dict[str, str]]:
    server = os.getenv("PROXY_SERVER")
    if not server:
        return None
    cfg: Dict[str, str] = {"server": server}
    u = os.getenv("PROXY_USERNAME")
    p = os.getenv("PROXY_PASSWORD")
    if u:
        cfg["username"] = u
    if p:
        cfg["password"] = p
    return cfg

# ============================================================
# 1) RAPIRA (твоя логика, без изменений парсинга; добавлен retry на goto)
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
                    len(rows), selector, i + 1, max_wait_seconds,
                )
                return rows

        rapira_log.info("No trade rows found yet (attempt %d/%d)...", i + 1, max_wait_seconds)
        await page.wait_for_timeout(1_000)

    rapira_log.warning("No history table rows found by any selector.")
    return []

def rapira_normalize_num(text: str) -> float:
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)

async def rapira_parse_trades_from_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    for row in rows:
        try:
            cells = await row.query_selector_all("th, td")
            if len(cells) < 3:
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
                    "qty": volume,
                    "time": time_text,
                    "price_raw": price_text,
                    "qty_raw": volume_text,
                }
            )
        except Exception:
            continue
    return trades

def compute_metrics_qty(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not trades:
        return {"count": 0, "sum_qty_usdt": 0.0, "turnover_quote": 0.0, "vwap": None}

    sum_qty = 0.0
    turnover = 0.0
    for t in trades:
        q = float(t["qty"])
        p = float(t["price"])
        sum_qty += q
        turnover += q * p

    vwap = turnover / sum_qty if sum_qty > 0 else None
    return {"count": len(trades), "sum_qty_usdt": sum_qty, "turnover_quote": turnover, "vwap": vwap}

async def scrape_rapira_trades(limit: int = LIMIT) -> Dict[str, Any]:
    wait_until = os.getenv("RAPIRA_GOTO_WAIT", "networkidle").strip()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy=_proxy_config(),
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

            # иногда networkidle на Render подвисает — делаем 2 попытки
            try:
                await page.goto(RAPIRA_URL, wait_until=wait_until, timeout=60_000)
            except Exception:
                await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)

            await page.wait_for_timeout(5_000)

            await rapira_accept_cookies_if_any(page)
            await rapira_ensure_last_trades_tab(page)
            await page.wait_for_timeout(3_000)

            rows = await rapira_poll_for_trade_rows(page, max_wait_seconds=40)
            trades = await rapira_parse_trades_from_rows(rows) if rows else []
            trades = trades[:limit]

            metrics = compute_metrics_qty(trades)
            return {
                "exchange": "rapira",
                "symbol": "USDT/RUB",
                "url": RAPIRA_URL,
                "quote_ccy": "RUB",
                **metrics,
                "trades": trades,
            }
        finally:
            await rapira_close_silently(page)
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass

# ============================================================
# 2) GRINEX (httpx как было + fallback на Playwright при HTML)
# ============================================================

grinex_log = logging.getLogger("grinex")

BASE_URL = "https://grinex.io"
MARKET = os.getenv("GRINEX_MARKET", "usdta7a5")
TRADES_URL = f"{BASE_URL}/api/v2/trades?market={MARKET}&limit={LIMIT}&order_by=desc"
TRADING_URL = f"{BASE_URL}/trading/{MARKET}?lang=ru"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": TRADING_URL,
}

@dataclass(frozen=True)
class GrinexTrade:
    price: float
    amount: float
    total: float
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

def grinex_extract_trades(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for k in ("trades", "data", "result"):
            v = payload.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []

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

def _grinex_pack(trades: List[GrinexTrade]) -> Dict[str, Any]:
    trades = trades[:LIMIT]
    sum_usdt = sum(t.amount for t in trades)
    sum_quote = sum(t.total for t in trades)
    vwap = (sum(t.price * t.amount for t in trades) / sum_usdt) if sum_usdt > 0 else None

    return {
        "exchange": "grinex",
        "symbol": "USDT/A7A5",
        "url": TRADES_URL,
        "quote_ccy": "A7A5",
        "count": len(trades),
        "sum_qty_usdt": float(sum_usdt),
        "turnover_quote": float(sum_quote),
        "vwap": float(vwap) if vwap is not None else None,
        "trades": [
            {
                "price": t.price,
                "qty": t.amount,
                "total": t.total,
                "time_utc": t.ts_utc.strftime("%Y-%m-%d %H:%M:%S"),
                "side": t.side,
                "id": t.tid,
            }
            for t in trades
        ],
    }

async def fetch_grinex_httpx() -> Dict[str, Any]:
    async with httpx.AsyncClient(
        headers=HEADERS,
        timeout=15,
        follow_redirects=True,
        verify=certifi.where(),
    ) as client:
        r = await client.get(TRADES_URL)
        grinex_log.info("HTTP %s %s", r.status_code, TRADES_URL)
        r.raise_for_status()

        text = (r.text or "").strip()
        if not text:
            raise ValueError("Empty response body from Grinex")
        if text.startswith("<"):
            raise ValueError(f"Non-JSON response (HTML?) from Grinex: {text[:200]}")

        payload = r.json()
        raw = grinex_extract_trades(payload)
        trades: List[GrinexTrade] = []
        for item in raw:
            tr = grinex_normalize_trade(item)
            if tr:
                trades.append(tr)

        trades.sort(key=lambda x: x.ts_utc, reverse=True)
        return _grinex_pack(trades)

async def fetch_grinex_playwright() -> Dict[str, Any]:
    # Фоллбек: заходим браузером, получаем cookies, затем дергаем API через page.request
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            proxy=_proxy_config(),
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
            user_agent=HEADERS["User-Agent"],
        )
        page = await context.new_page()
        try:
            grinex_log.warning("Grinex returned HTML; switching to Playwright fallback via %s", TRADING_URL)
            await page.goto(TRADING_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(3_000)

            resp = await page.request.get(TRADES_URL, headers=HEADERS, timeout=30_000)
            if not resp.ok:
                raise ValueError(f"Playwright request failed: HTTP {resp.status} {TRADES_URL}")

            text = (await resp.text()).strip()
            if not text or text.startswith("<"):
                raise ValueError(f"Playwright got non-JSON too. Body head: {text[:200]}")

            payload = await resp.json()
            raw = grinex_extract_trades(payload)

            trades: List[GrinexTrade] = []
            for item in raw:
                tr = grinex_normalize_trade(item)
                if tr:
                    trades.append(tr)

            trades.sort(key=lambda x: x.ts_utc, reverse=True)
            return _grinex_pack(trades)

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

async def fetch_grinex_once() -> Dict[str, Any]:
    try:
        return await fetch_grinex_httpx()
    except Exception as e:
        # если это HTML-антибот — идем через браузер
        msg = str(e)
        if "Non-JSON response" in msg or "HTML" in msg:
            return await fetch_grinex_playwright()
        raise

# ============================================================
# 3) ABCEX (твоя логика сделок + поиск логин-формы по frames + лог входов)
# ============================================================

abcex_log = logging.getLogger("abcex")

ABCEX_URL = "https://abcex.io/client/spot/USDTRUB"
STATE_PATH = "abcex_state.json"
MAX_TRADES = 20

def abcex_normalize_num(text: str) -> float:
    t = text.strip().replace("\xa0", " ")
    t = t.replace(" ", "")
    if "," in t and "." in t:
        t = t.replace(",", "")
    else:
        t = t.replace(",", ".")
    return float(t)

async def abcex_save_debug(page: Page, html_path: str, png_path: str) -> None:
    try:
        await page.screenshot(path=png_path, full_page=True)
        abcex_log.info("Saved debug screenshot: %s", png_path)
    except Exception as e:
        abcex_log.info("Could not save screenshot: %s", e)

    try:
        content = await page.content()
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(content)
        abcex_log.info("Saved debug html: %s", html_path)
    except Exception as e:
        abcex_log.info("Could not save html: %s", e)

async def abcex_log_visible_inputs(page: Page) -> None:
    try:
        data = await page.evaluate(
            """() => {
              const els = Array.from(document.querySelectorAll('input,textarea,select'))
                .filter(el => {
                  const r = el.getBoundingClientRect();
                  const style = window.getComputedStyle(el);
                  return r.width > 0 && r.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
                })
                .slice(0, 30)
                .map(el => {
                  const attrs = {};
                  for (const a of el.attributes) attrs[a.name] = a.value;
                  return { tag: el.tagName.toLowerCase(), attrs, outer: el.outerHTML.slice(0, 300) };
                });
              return els;
            }"""
        )
        abcex_log.warning("ABCEX visible inputs sample (top 30): %s", json.dumps(data, ensure_ascii=False))
    except Exception as e:
        abcex_log.warning("Failed to dump visible inputs: %s", e)

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

def _all_scopes(page: Page) -> List[Any]:
    # Page + все frames
    scopes: List[Any] = [page]
    try:
        scopes.extend(page.frames)
    except Exception:
        pass
    return scopes

async def _is_login_visible_anywhere(page: Page) -> Tuple[bool, Any]:
    # если password есть в каком-то frame — логин нужен, и вернем scope (page/frame)
    for scope in _all_scopes(page):
        try:
            pw = scope.locator("input[type='password']")
            if await pw.count() > 0 and await pw.first.is_visible():
                return True, scope
        except Exception:
            continue
    return False, page

async def _pick_email_locator(scope: Any) -> Optional[Locator]:
    # можно переопределить через ENV, если понадобится
    env_sel = os.getenv("ABCEX_EMAIL_SELECTOR")
    if env_sel:
        loc = scope.locator(env_sel)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                return loc.first
        except Exception:
            pass

    candidates = [
        "input[type='email']",
        "input[autocomplete='email']",
        "input[name*='mail' i]",
        "input[id*='mail' i]",
        "input[placeholder*='mail' i]",
        "input[placeholder*='email' i]",
        "input[placeholder*='почт' i]",
        "input[aria-label*='mail' i]",
        "input[aria-label*='email' i]",
        "input[aria-label*='почт' i]",
    ]
    for sel in candidates:
        loc = scope.locator(sel)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                return loc.first
        except Exception:
            continue

    # fallback: первый видимый input, который не password
    fallback = scope.locator("input:not([type='password'])")
    try:
        if await fallback.count() > 0:
            for i in range(min(await fallback.count(), 10)):
                el = fallback.nth(i)
                try:
                    if await el.is_visible():
                        return el
                except Exception:
                    continue
    except Exception:
        pass

    return None

async def _pick_password_locator(scope: Any) -> Optional[Locator]:
    env_sel = os.getenv("ABCEX_PASSWORD_SELECTOR")
    if env_sel:
        loc = scope.locator(env_sel)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                return loc.first
        except Exception:
            pass

    candidates = [
        "input[type='password']",
        "input[autocomplete='current-password']",
        "input[name*='pass' i]",
        "input[id*='pass' i]",
        "input[placeholder*='парол' i]",
        "input[placeholder*='password' i]",
        "input[aria-label*='парол' i]",
        "input[aria-label*='password' i]",
    ]
    for sel in candidates:
        loc = scope.locator(sel)
        try:
            if await loc.count() > 0 and await loc.first.is_visible():
                return loc.first
        except Exception:
            continue
    return None

async def abcex_login_if_needed(page: Page, email: str, password: str) -> None:
    needed, scope = await _is_login_visible_anywhere(page)
    if not needed:
        abcex_log.info("Login not required (already in session).")
        return

    abcex_log.info("Login detected. Performing sign-in ... (scope=%s)", type(scope).__name__)

    email_loc = await _pick_email_locator(scope)
    pw_loc = await _pick_password_locator(scope)

    if email_loc is None or pw_loc is None:
        await abcex_log_visible_inputs(page)
        await abcex_save_debug(page, "abcex_login_fields_not_found.html", "abcex_login_fields_not_found.png")
        raise RuntimeError("Не смог найти поля email/password. См. abcex_login_fields_not_found.*")

    # fill email
    try:
        await email_loc.click(timeout=10_000)
        await email_loc.fill(email, timeout=10_000)
    except Exception:
        try:
            await email_loc.click(timeout=10_000)
            await page.keyboard.type(email)
        except Exception:
            await abcex_log_visible_inputs(page)
            await abcex_save_debug(page, "abcex_login_email_fill_failed.html", "abcex_login_email_fill_failed.png")
            raise

    # fill password
    try:
        await pw_loc.click(timeout=10_000)
        await pw_loc.fill(password, timeout=10_000)
    except Exception:
        try:
            await pw_loc.click(timeout=10_000)
            await page.keyboard.type(password)
        except Exception:
            await abcex_log_visible_inputs(page)
            await abcex_save_debug(page, "abcex_login_pass_fill_failed.html", "abcex_login_pass_fill_failed.png")
            raise

    # click login button (в том же scope)
    btn_texts = ["Войти", "Вход", "Sign in", "Login", "Войти в аккаунт"]
    clicked = False
    for t in btn_texts:
        try:
            btn = scope.locator(f"button:has-text('{t}')")
            if await btn.count() > 0 and await btn.first.is_visible():
                await btn.first.click(timeout=10_000)
                clicked = True
                break
        except Exception:
            continue

    if not clicked:
        try:
            await page.keyboard.press("Enter")
            clicked = True
        except Exception:
            pass

    try:
        await page.wait_for_timeout(1_000)
        await page.wait_for_load_state("networkidle", timeout=30_000)
    except Exception:
        pass

    await page.wait_for_timeout(2_500)

    # проверка: password пропал во всех scopes
    still, _ = await _is_login_visible_anywhere(page)
    if still:
        await abcex_log_visible_inputs(page)
        await abcex_save_debug(page, "abcex_login_failed.html", "abcex_login_failed.png")
        raise RuntimeError("Логин не прошёл (password всё ещё виден). Возможны 2FA/капча/иной флоу.")

    abcex_log.info("Login successful (password disappeared).")

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
        await abcex_save_debug(page, "abcex_no_panel.html", "abcex_no_panel.png")
        raise RuntimeError("Не нашёл панель panel-orderHistory.")
    for i in range(cnt):
        p = panel.nth(i)
        try:
            if await p.is_visible():
                return p
        except Exception:
            continue
    await abcex_save_debug(page, "abcex_no_visible_panel.html", "abcex_no_visible_panel.png")
    raise RuntimeError("panel-orderHistory есть, но не видима.")

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

    await abcex_save_debug(page, "abcex_trades_not_visible.html", "abcex_trades_not_visible.png")
    raise RuntimeError("Не дождался появления сделок (HH:MM:SS).")

async def abcex_extract_trades_from_panel(panel: Locator, limit: int = MAX_TRADES) -> List[AbcexTrade]:
    handle = await panel.element_handle()
    if handle is None:
        raise RuntimeError("Не смог получить element_handle панели.")

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

def abcex_compute_metrics(trades: List[AbcexTrade]) -> Dict[str, Any]:
    if not trades:
        return {"count": 0, "sum_qty_usdt": 0.0, "turnover_quote": 0.0, "vwap": None}
    sum_qty = 0.0
    turnover = 0.0
    for t in trades:
        sum_qty += t.qty
        turnover += t.qty * t.price
    vwap = turnover / sum_qty if sum_qty > 0 else None
    return {"count": len(trades), "sum_qty_usdt": sum_qty, "turnover_quote": turnover, "vwap": vwap}

async def scrape_abcex_trades(headless: bool = True) -> Dict[str, Any]:
    email = os.getenv("ABCEX_EMAIL")
    password = os.getenv("ABCEX_PASSWORD")
    if not email or not password:
        raise RuntimeError("ABCEX_EMAIL/ABCEX_PASSWORD не заданы в env.")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=headless,
            proxy=_proxy_config(),
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

            await abcex_login_if_needed(page, email=email, password=password)

            try:
                await context.storage_state(path=STATE_PATH)
                abcex_log.info("Saved session state to %s", STATE_PATH)
            except Exception as e:
                abcex_log.info("Could not save storage state: %s", e)

            await abcex_click_trades_tab_best_effort(page)
            await abcex_wait_trades_visible(page)

            panel = await abcex_get_order_history_panel(page)
            trades = await abcex_extract_trades_from_panel(panel, limit=MAX_TRADES)

            if not trades:
                await abcex_save_debug(page, "abcex_debug_no_trades_parsed.html", "abcex_debug_no_trades_parsed.png")
                raise RuntimeError("panel-orderHistory найдена, но сделки не распарсились.")

            metrics = abcex_compute_metrics(trades)

            return {
                "exchange": "abcex",
                "symbol": "USDT/RUB",
                "url": ABCEX_URL,
                "quote_ccy": "RUB",
                **metrics,
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
# Unified loop
# ============================================================

def pack_ok(result: Dict[str, Any]) -> Dict[str, Any]:
    return {"ok": True, "error": None, "data": result}

def pack_err(exchange: str, err: Exception) -> Dict[str, Any]:
    return {"ok": False, "error": f"{type(err).__name__}: {err}", "data": {"exchange": exchange}}

def compare_block(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    for r in results:
        if not r.get("ok"):
            continue
        d = r["data"]
        rows.append(
            {
                "exchange": d.get("exchange"),
                "symbol": d.get("symbol"),
                "count": d.get("count"),
                "sum_qty_usdt": d.get("sum_qty_usdt"),
                "turnover_quote": d.get("turnover_quote"),
                "quote_ccy": d.get("quote_ccy"),
                "vwap": d.get("vwap"),
            }
        )

    rows_by_turnover = sorted(rows, key=lambda x: (x["turnover_quote"] or 0.0), reverse=True)
    rows_by_qty = sorted(rows, key=lambda x: (x["sum_qty_usdt"] or 0.0), reverse=True)

    return {
        "by_turnover_quote_desc": rows_by_turnover,
        "by_sum_qty_usdt_desc": rows_by_qty,
    }

async def run_one_cycle() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    # Grinex
    try:
        g = await fetch_grinex_once()
        results.append(pack_ok(g))
    except Exception as e:
        results.append(pack_err("grinex", e))

    # Rapira
    try:
        r = await scrape_rapira_trades(limit=LIMIT)
        results.append(pack_ok(r))
    except Exception as e:
        results.append(pack_err("rapira", e))

    # ABCEX
    try:
        headless = os.getenv("ABCEX_HEADLESS", "1").strip() not in ("0", "false", "False")
        a = await scrape_abcex_trades(headless=headless)
        results.append(pack_ok(a))
    except Exception as e:
        results.append(pack_err("abcex", e))

    return {
        "ts_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "poll_sec": POLL_SEC,
        "limit": LIMIT,
        "results": results,
        "compare": compare_block(results),
    }

async def main() -> None:
    log.info("Starting unified 3-exchange scraper ...")
    ensure_playwright_browsers()

    while True:
        summary = await run_one_cycle()
        print(json.dumps(summary, ensure_ascii=False))
        await asyncio.sleep(POLL_SEC)

if __name__ == "__main__":
    asyncio.run(main())
