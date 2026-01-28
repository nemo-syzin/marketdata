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
    """
    Extracts pair slug from:
      https://rapira.net/ru/exchange/USDT_RUB -> USDT_RUB
      https://rapira.net/exchange/USDT_RUB    -> USDT_RUB
    """
    try:
        path = urlparse(url).path or ""
        # find '/exchange/<slug>'
        m = re.search(r"/exchange/([^/?#]+)", path)
        if m:
            return m.group(1)
    except Exception:
        pass
    return os.getenv("PAIR_SLUG", "USDT_RUB").strip() or "USDT_RUB"

PAIR_SLUG = os.getenv("PAIR_SLUG", _extract_pair_slug(RAPIRA_URL)).strip() or "USDT_RUB"
PAIR_TEXT = os.getenv("PAIR_TEXT", "USDT/RUB").strip() or "USDT/RUB"

# two candidate URLs: /ru/ and without /ru/ (Rapira иногда уводит в не-ru)
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

# ───────────────────────── SELECTORS (Last Trades table) ─────────────────────────
#
# From your inspector:
#  tbody > tr.table-orders-row ...
#  td[0] = price, td[1] = volume, td[2] = time
#
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
    """
    Кликаем только внутри cookie-баннера, чтобы случайно не нажать что-то в терминале.
    """
    containers = [
        "div[role='dialog']:has-text('cookie')",
        "div[role='dialog']:has-text('cookies')",
        "div[role='dialog']:has-text('куки')",
        "div:has-text('cookies')",
        "div:has-text('куки')",
    ]
    buttons = ["OK", "ОК", "Принять", "Accept", "Я согласен"]

    for cont_sel in containers:
        cont = page.locator(cont_sel).first
        try:
            if await cont.count() == 0:
                continue
            if not await cont.is_visible(timeout=800):
                continue

            for b in buttons:
                btn = cont.locator(f"button:has-text('{b}')").first
                if await btn.count() and await btn.is_visible(timeout=800):
                    logger.info("Found cookies banner, clicking '%s'...", b)
                    await btn.click(timeout=2000, no_wait_after=True)
                    await page.wait_for_timeout(250)
                    return
        except Exception:
            continue

async def ensure_last_trades_tab(page: Page) -> None:
    """
    Включаем вкладку "Последние сделки" (если она есть).
    """
    try:
        # сначала по роли (стабильнее), потом fallback по тексту
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

async def ensure_correct_pair(page: Page) -> None:
    """
    Главное: Rapira SPA иногда открывает USDT_RUB и потом переключает на BTC_USDT.
    Мы не начинаем парсинг, пока:
      - URL содержит /exchange/<PAIR_SLUG>
      - На странице виден текст пары (PAIR_TEXT, по умолчанию USDT/RUB)
    """
    for attempt in range(1, 9):
        url = page.url or ""
        url_ok = (f"/exchange/{PAIR_SLUG}" in url)

        text_ok = False
        try:
            # пара обычно есть вверху рядом с тикером
            text_ok = await page.locator(f"text=/{PAIR_TEXT_RE.pattern}/i").first.is_visible(timeout=1500)
        except Exception:
            text_ok = False

        if url_ok and text_ok:
            return

        logger.warning(
            "Wrong pair detected (attempt %d). url=%s; need /exchange/%s and text %s",
            attempt, url, PAIR_SLUG, PAIR_TEXT
        )

        # пробуем обе канонические ссылки
        target = TARGET_URLS[(attempt - 1) % len(TARGET_URLS)]
        await page.goto(target, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(700)
        await accept_cookies_if_any(page)
        await ensure_last_trades_tab(page)
        await page.wait_for_timeout(200)

    raise RuntimeError(f"Cannot lock pair {PAIR_SLUG}. final_url={page.url}")

async def hard_fix_pair_if_ui_redirects(page: Page) -> None:
    """
    Иногда SPA дорисовывается позже и меняет инструмент.
    Эта проверка перед парсингом и после reload защищает от BTC_USDT.
    """
    await ensure_correct_pair(page)
    try:
        await page.wait_for_selector(TRADE_ROWS_SELECTOR, timeout=10_000)
    except Exception:
        await page.reload(wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(700)
        await accept_cookies_if_any(page)
        await ensure_last_trades_tab(page)
        await ensure_correct_pair(page)

# ───────────────────────── PARSING ─────────────────────────

def parse_row_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    По твоим данным:
      texts[0] = курс (price)
      texts[1] = объем USDT (volume_usdt)
      texts[2] = время (trade_time)
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

async def open_browser(pw) -> Tuple[Browser, BrowserContext, Page]:
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
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

    # На всякий случай убираем "последний выбранный инструмент" (SPA часто хранит его)
    await context.add_init_script(
        "try{localStorage.clear();sessionStorage.clear();}catch(e){}"
    )

    page = await context.new_page()
    page.set_default_timeout(10_000)

    # стартуем сразу с канонического URL (потом всё равно проверим)
    await page.goto(TARGET_URLS[0], wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(700)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)

    # критично: закрепляем нужную пару
    await ensure_correct_pair(page)

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

                # периодический heartbeat
                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url)
                    last_heartbeat = time.monotonic()

                # периодический reload
                if time.monotonic() - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await page.goto(TARGET_URLS[0], wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(700)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
                    await ensure_correct_pair(page)
                    last_reload = time.monotonic()

                # иногда вкладку нужно тыкать повторно
                if time.monotonic() - last_tab_click >= 15:
                    await ensure_last_trades_tab(page)
                    last_tab_click = time.monotonic()

                # ключ: прямо перед парсингом убеждаемся, что нас не унесло на BTC_USDT
                await hard_fix_pair_if_ui_redirects(page)

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Reloading page... url=%s", page.url)
                    await page.goto(TARGET_URLS[0], wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(700)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
                    await ensure_correct_pair(page)
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
