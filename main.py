# main.py
import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Deque, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# ───────────────────────── CONFIG ─────────────────────────

RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/ru/exchange/USDT_RUB").strip()
SOURCE = os.getenv("SOURCE", "rapira").strip()
SYMBOL = os.getenv("SYMBOL", "USDT/RUB").strip()

LIMIT = int(os.getenv("LIMIT", "150"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))

SCRAPE_TIMEOUT_SECONDS = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "25"))
UPSERT_TIMEOUT_SECONDS = float(os.getenv("UPSERT_TIMEOUT_SECONDS", "25"))
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "30"))
RELOAD_EVERY_SECONDS = float(os.getenv("RELOAD_EVERY_SECONDS", "600"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades").strip()

ON_CONFLICT = os.getenv("ON_CONFLICT", "source,symbol,trade_time,price,volume_usdt").strip()

SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"
SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    force=True,
)
logger = logging.getLogger("rapira-worker")

Q8 = Decimal("0.00000001")
TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")

# ───────────────────────── TARGET PAIR LOCK ─────────────────────────

def _extract_pair_slug(url: str) -> str:
    try:
        path = urlparse(url).path or ""
        m = re.search(r"/exchange/([^/?#]+)", path)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "USDT_RUB"

PAIR_SLUG = os.getenv("PAIR_SLUG", _extract_pair_slug(RAPIRA_URL)).strip() or "USDT_RUB"
PAIR_TEXT = os.getenv("PAIR_TEXT", "USDT/RUB").strip() or "USDT/RUB"

CANON_URL_NO_RU = f"https://rapira.net/exchange/{PAIR_SLUG}"
CANON_URL_RU = f"https://rapira.net/ru/exchange/{PAIR_SLUG}"
TARGET_URLS = [CANON_URL_RU, CANON_URL_NO_RU]

PAIR_TEXT_RE = re.compile(rf"{re.escape(PAIR_TEXT).replace('/', r'\s*/\s*')}", re.IGNORECASE)

# ───────────────────────── HELPERS ─────────────────────────

def normalize_decimal(text: str) -> Optional[Decimal]:
    t = (text or "").strip()
    if not t:
        return None
    t = t.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    try:
        return Decimal(t)
    except (InvalidOperation, ValueError):
        return None

def extract_time(text: str) -> Optional[str]:
    m = TIME_RE.search((text or "").replace("\xa0", " "))
    if not m:
        return None
    hh, mm, ss = m.group(0).split(":")
    if len(hh) == 1:
        hh = "0" + hh
    return f"{hh}:{mm}:{ss}"

def q8_str(x: Decimal) -> str:
    return str(x.quantize(Q8, rounding=ROUND_HALF_UP))

@dataclass(frozen=True)
class TradeKey:
    source: str
    symbol: str
    trade_time: str
    price: str
    volume_usdt: str

def trade_key(t: Dict[str, Any]) -> TradeKey:
    return TradeKey(
        source=t["source"],
        symbol=t["symbol"],
        trade_time=t["trade_time"],
        price=t["price"],
        volume_usdt=t["volume_usdt"],
    )

# ───────────────────────── PLAYWRIGHT INSTALL ─────────────────────────

_last_install_ts = 0.0

def _playwright_install() -> None:
    global _last_install_ts
    now = time.time()
    if now - _last_install_ts < 600:
        logger.warning("Playwright install was attempted recently; skipping (cooldown).")
        return
    _last_install_ts = now

    logger.warning("Installing Playwright browsers (runtime)...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "chromium-headless-shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.error(
                "playwright install failed (%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
                r.returncode, r.stdout, r.stderr
            )
        else:
            logger.info("Playwright browsers installed.")
    except Exception as e:
        logger.error("Cannot run playwright install: %s", e)

def _should_force_install(err: Exception) -> bool:
    s = str(err)
    return (
        "Executable doesn't exist" in s
        or "playwright install" in s
        or "chromium_headless_shell" in s
        or ("ms-playwright" in s and "doesn't exist" in s)
    )

# ───────────────────────── SELECTORS ─────────────────────────
# Таблица “Последние сделки”
TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table tbody tr.table-orders-row"
)

EVAL_JS = """
(rows, limit) => rows.slice(0, limit).map(row => {
  const tds = Array.from(row.querySelectorAll('td'));
  const texts = tds.map(td => (td.innerText || '').trim());
  return { texts };
})
"""

# ───────────────────────── PAGE ACTIONS ─────────────────────────

async def accept_cookies_if_any(page: Page) -> None:
    # Нажимаем ОК/Accept, но максимально безопасно
    for label in ["OK", "ОК", "Принять", "Accept", "Я согласен"]:
        try:
            btn = page.locator(f"button:has-text('{label}')").first
            if await btn.count() and await btn.is_visible(timeout=800):
                await btn.click(timeout=2000, no_wait_after=True)
                await page.wait_for_timeout(250)
                return
        except Exception:
            pass

async def ensure_last_trades_tab(page: Page) -> None:
    # Пытаемся включить вкладку “Последние сделки”
    try:
        tab = page.get_by_role("tab", name=re.compile(r"Последние\s+сделки", re.I))
        if await tab.count():
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
            return
    except Exception:
        pass

    try:
        tab = page.locator("text=/Последние\\s+сделки/i").first
        if await tab.count():
            await tab.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
    except Exception:
        pass

async def is_correct_pair(page: Page) -> bool:
    url = page.url or ""
    if f"/exchange/{PAIR_SLUG}" in url:
        return True

    # Фолбэк: проверим, что на странице реально видна строка USDT/RUB
    try:
        loc = page.locator(f"text=/{PAIR_TEXT_RE.pattern}/i").first
        if await loc.count() and await loc.is_visible(timeout=1200):
            return True
    except Exception:
        pass

    return False

async def try_select_pair_via_ui(page: Page) -> bool:
    """
    Универсальная попытка: клик по текущей паре -> поиск -> выбрать USDT/RUB.
    Селекторы сделаны максимально “общими”, без жёсткой привязки к классам.
    """
    # 1) Открыть список инструментов кликом по текущей паре (BTC/USDT или USDT/RUB и т.п.)
    # Ищем любой кликабельный элемент с паттерном XXX/YYY
    try:
        current_pair = page.locator(
            "text=/\\b[A-Z]{2,10}\\s*\\/\\s*[A-Z]{2,10}\\b/"
        ).first
        if await current_pair.count() and await current_pair.is_visible(timeout=1500):
            await current_pair.click(timeout=5000, no_wait_after=True)
            await page.wait_for_timeout(300)
    except Exception:
        pass

    # 2) Попробовать найти поле поиска
    search_input = None
    for sel in [
        "input[placeholder*='Поиск']",
        "input[placeholder*='Search']",
        "input[type='search']",
        "input[name*='search']",
    ]:
        try:
            loc = page.locator(sel).first
            if await loc.count() and await loc.is_visible(timeout=800):
                search_input = loc
                break
        except Exception:
            continue

    # Если поиска нет — всё равно попробуем просто кликнуть по USDT/RUB в списке
    try:
        if search_input is not None:
            await search_input.fill("")
            await search_input.type(PAIR_TEXT, delay=25)
            await page.wait_for_timeout(250)
    except Exception:
        pass

    # 3) Клик по варианту пары
    try:
        option = page.locator(f"text=/{PAIR_TEXT_RE.pattern}/i").first
        if await option.count() and await option.is_visible(timeout=1500):
            await option.click(timeout=5000, no_wait_after=True)
            await page.wait_for_timeout(500)
    except Exception:
        pass

    # 4) Дождаться URL нужной пары (если SPA обновляет адрес)
    try:
        await page.wait_for_url(re.compile(rf"/exchange/{re.escape(PAIR_SLUG)}"), timeout=10_000)
    except Exception:
        pass

    return await is_correct_pair(page)

async def force_pair_lock(page: Page) -> None:
    """
    Главная функция: гарантирует, что мы на USDT_RUB.
    Если сайт уводит на BTC_USDT — вернём и переключим через UI.
    """
    for attempt in range(1, 10):
        if await is_correct_pair(page):
            return

        url = page.url or ""
        logger.warning(
            "Wrong pair detected (attempt %d). url=%s; need /exchange/%s and text %s",
            attempt, url, PAIR_SLUG, PAIR_TEXT
        )

        # 1) Жёстко перейдём по канонической ссылке
        target = TARGET_URLS[(attempt - 1) % len(TARGET_URLS)]
        await page.goto(target, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(700)
        await accept_cookies_if_any(page)
        await ensure_last_trades_tab(page)

        if await is_correct_pair(page):
            return

        # 2) Если снова унесло — попробуем выбрать пару через интерфейс
        ok = await try_select_pair_via_ui(page)
        if ok:
            return

        # 3) Если внезапно оказались на главной — ещё раз попробуем канон + UI
        await page.wait_for_timeout(500)

    raise RuntimeError(f"Cannot lock pair {PAIR_SLUG}. final_url={page.url}")

# ───────────────────────── PARSING ─────────────────────────

def parse_row_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    По твоим данным:
      td[0] = курс
      td[1] = объем USDT
      td[2] = время
    """
    texts: List[str] = payload.get("texts") or []
    if len(texts) < 3:
        return None

    price = normalize_decimal(texts[0])
    volume_usdt = normalize_decimal(texts[1])
    trade_time = extract_time(texts[2])

    if price is None or volume_usdt is None or not trade_time:
        return None
    if price <= 0 or volume_usdt <= 0:
        return None

    volume_rub = price * volume_usdt

    return {
        "source": SOURCE,
        "symbol": SYMBOL,
        "price": q8_str(price),
        "volume_usdt": q8_str(volume_usdt),
        "volume_rub": q8_str(volume_rub),
        "trade_time": trade_time,
    }

async def scrape_window_fast(page: Page) -> List[Dict[str, Any]]:
    t0 = time.monotonic()
    await page.wait_for_selector(TRADE_ROWS_SELECTOR, timeout=int(SCRAPE_TIMEOUT_SECONDS * 1000))

    payloads = await page.eval_on_selector_all(
        TRADE_ROWS_SELECTOR,
        EVAL_JS,
        LIMIT,
    )

    out: List[Dict[str, Any]] = []
    for p in payloads:
        t = parse_row_payload(p)
        if t:
            out.append(t)

    dt = time.monotonic() - t0
    if dt > SCRAPE_TIMEOUT_SECONDS:
        raise asyncio.TimeoutError(f"scrape_window_fast took {dt:.2f}s")
    return out

# ───────────────────────── SUPABASE ─────────────────────────

def _sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }

async def supabase_upsert(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set; skipping insert.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    params = {"on_conflict": ON_CONFLICT}

    async with httpx.AsyncClient(timeout=UPSERT_TIMEOUT_SECONDS) as client:
        for i in range(0, len(rows), UPSERT_BATCH):
            chunk = rows[i:i + UPSERT_BATCH]
            try:
                r = await client.post(url, headers=_sb_headers(), params=params, json=chunk)
            except Exception as e:
                logger.error("Supabase POST error: %s", e)
                return

            if r.status_code >= 300:
                logger.error("Supabase upsert failed (%s): %s", r.status_code, r.text)
                return

    logger.info("Inserted (or ignored duplicates) %d rows into '%s'.", len(rows), SUPABASE_TABLE)

# ───────────────────────── BROWSER SESSION ─────────────────────────

STEALTH_INIT = r"""
// минимальные правки для снижения "headless-редиректов"
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['ru-RU','ru','en-US','en']});
Object.defineProperty(navigator, 'platform', {get: () => 'Linux x86_64'});
"""

async def open_browser(pw) -> Tuple[Browser, BrowserContext, Page]:
    browser = await pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ],
    )

    context = await browser.new_context(
        viewport={"width": 1440, "height": 810},
        locale="ru-RU",
        timezone_id="Europe/Moscow",
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    await context.add_init_script(STEALTH_INIT)

    page = await context.new_page()
    page.set_default_timeout(10_000)

    await page.goto(TARGET_URLS[0], wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(700)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)

    # Критично: закрепляем нужную пару, иначе Rapira может утащить на BTC_USDT
    await force_pair_lock(page)

    return browser, context, page

async def safe_close(browser: Optional[Browser], context: Optional[BrowserContext], page: Optional[Page]) -> None:
    try:
        if page:
            await page.close()
    except Exception:
        pass
    try:
        if context:
            await context.close()
    except Exception:
        pass
    try:
        if browser:
            await browser.close()
    except Exception:
        pass

# ───────────────────────── WORKER LOOP ─────────────────────────

async def worker() -> None:
    seen: Set[TradeKey] = set()
    seen_q: Deque[TradeKey] = deque()

    backoff = 2.0
    last_heartbeat = time.monotonic()
    last_reload = time.monotonic()
    last_tab_click = 0.0

    async with async_playwright() as pw:
        browser: Optional[Browser] = None
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None

        while True:
            try:
                if page is None:
                    logger.info("Starting browser session...")
                    try:
                        browser, context, page = await open_browser(pw)
                    except Exception as e:
                        if (not SKIP_BROWSER_INSTALL) or _should_force_install(e):
                            _playwright_install()
                            browser, context, page = await open_browser(pw)
                        else:
                            raise

                    backoff = 2.0
                    last_reload = time.monotonic()
                    last_heartbeat = time.monotonic()
                    last_tab_click = time.monotonic()

                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url)
                    last_heartbeat = time.monotonic()

                if time.monotonic() - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await page.goto(TARGET_URLS[0], wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(700)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
                    await force_pair_lock(page)
                    last_reload = time.monotonic()

                if time.monotonic() - last_tab_click >= 15:
                    await ensure_last_trades_tab(page)
                    last_tab_click = time.monotonic()

                # перед парсингом ещё раз убеждаемся, что нас не унесло
                await force_pair_lock(page)

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Reloading page... url=%s", page.url)
                    await page.goto(TARGET_URLS[0], wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(700)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
                    await force_pair_lock(page)
                    await asyncio.sleep(max(0.5, POLL_SECONDS))
                    continue

                new_rows: List[Dict[str, Any]] = []
                for t in reversed(window):
                    k = trade_key(t)
                    if k in seen:
                        continue

                    new_rows.append(t)
                    seen.add(k)
                    seen_q.append(k)

                    if len(seen_q) > SEEN_MAX:
                        old = seen_q.popleft()
                        seen.discard(old)

                if new_rows:
                    logger.info(
                        "Parsed %d new trades. Newest: %s",
                        len(new_rows),
                        json.dumps(new_rows[-1], ensure_ascii=False),
                    )
                    await supabase_upsert(new_rows)

                sleep_s = max(0.35, POLL_SECONDS + random.uniform(-0.15, 0.15))
                await asyncio.sleep(sleep_s)

            except asyncio.TimeoutError:
                logger.error(
                    "Timeout: scrape_window exceeded %.1fs. Restarting browser session...",
                    SCRAPE_TIMEOUT_SECONDS
                )
                await safe_close(browser, context, page)
                browser = context = page = None

            except Exception as e:
                logger.error("Worker error: %s", e)

                if (not SKIP_BROWSER_INSTALL) or _should_force_install(e):
                    _playwright_install()

                logger.info("Retrying after %.1fs ...", backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

                await safe_close(browser, context, page)
                browser = context = page = None

def main() -> None:
    asyncio.run(worker())

if __name__ == "__main__":
    main()
