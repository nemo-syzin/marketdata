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

import httpx
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# ───────────────────────── CONFIG ─────────────────────────

# РЕКОМЕНДУЮ без /ru — меньше шансов “витринного” редиректа
RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB")
PAIR_SLUG = os.getenv("PAIR_SLUG", "USDT_RUB")   # то, что должно быть в URL
PAIR_TEXT = os.getenv("PAIR_TEXT", "USDT/RUB")   # то, что должно быть видно на странице

SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

LIMIT = int(os.getenv("LIMIT", "150"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))

SCRAPE_TIMEOUT_SECONDS = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "25"))
UPSERT_TIMEOUT_SECONDS = float(os.getenv("UPSERT_TIMEOUT_SECONDS", "25"))
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "30"))
RELOAD_EVERY_SECONDS = float(os.getenv("RELOAD_EVERY_SECONDS", "600"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

ON_CONFLICT = "source,symbol,trade_time,price,volume_usdt"

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

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
Q8 = Decimal("0.00000001")

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

# ───────────────────────── PAGE ACTIONS ─────────────────────────

async def accept_cookies_if_any(page: Page) -> None:
    for label in ["Я согласен", "Принять", "Accept", "OK", "ОК"]:
        try:
            btn = page.locator(f"text={label}")
            if await btn.count() > 0:
                logger.info("Found cookies banner, clicking '%s'...", label)
                await btn.first.click(timeout=5_000, no_wait_after=True)
                await page.wait_for_timeout(250)
                return
        except Exception:
            pass

async def ensure_last_trades_tab(page: Page) -> None:
    # если вкладка есть — нажмём (иногда таблица “лениво” активируется)
    for label in ["Последние сделки", "Last trades", "Trades"]:
        try:
            tab = page.locator(f"text={label}")
            if await tab.count() > 0:
                await tab.first.click(timeout=3_000, no_wait_after=True)
                await page.wait_for_timeout(150)
                return
        except Exception:
            pass

async def is_correct_pair(page: Page) -> bool:
    url = (page.url or "").lower()

    # допускаем /ru/exchange/..., /exchange/... и т.п.
    url_ok = (PAIR_SLUG.lower() in url) and ("/exchange/" in url or "/ru/exchange/" in url)
    if not url_ok:
        return False

    # текст на странице (обычно где-то в шапке/в блоке пары)
    try:
        t = page.locator(f"text={PAIR_TEXT}")
        if await t.count() > 0:
            return True
    except Exception:
        pass

    # иногда отображают как USDT_RUB или USDT RUB
    try:
        alt = page.locator(f"text={PAIR_SLUG}")
        if await alt.count() > 0:
            return True
    except Exception:
        pass

    return False

async def lock_pair(page: Page) -> None:
    """
    Жёстко удерживаем нужную пару.
    Идея: сайт может сам перепрыгивать на BTC_USDT/главную. Мы это ловим и возвращаем.
    """
    for attempt in range(1, 9):
        # Навигация на нужный URL
        await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
        await page.wait_for_timeout(700)
        await accept_cookies_if_any(page)
        await ensure_last_trades_tab(page)
        await page.wait_for_timeout(300)

        if await is_correct_pair(page):
            if attempt > 1:
                logger.info("Pair locked after attempt %d. url=%s", attempt, page.url)
            return

        logger.warning(
            "Wrong pair detected (attempt %d). url=%s; need /exchange/%s and text %s",
            attempt, page.url, PAIR_SLUG, PAIR_TEXT
        )

        # Если нас уже унесло на BTC_USDT — попробуем принудительно заменить location
        try:
            await page.evaluate(
                """(targetUrl, slug) => {
                    try {
                      const u = String(location.href);
                      if (!u.includes(slug)) location.replace(targetUrl);
                    } catch (e) {}
                }""",
                RAPIRA_URL,
                PAIR_SLUG,
            )
        except Exception:
            pass

        await page.wait_for_timeout(int(600 + random.uniform(0, 500)))

    raise RuntimeError(f"Cannot lock pair {PAIR_SLUG}. final_url={page.url}")

# ───────────────────────── PARSING ─────────────────────────
# Твоя структура:
# 1-й td = курс
# 2-й td = объём USDT
# 3-й td = время

ROWS_SELECTORS = [
    # твой текущий (если на сайте такой)
    "div.table-responsive.table-orders table.table-row-dashed tbody tr.table-orders-row",
    # частые варианты
    "div.table-responsive table tbody tr",
    "table tbody tr",
]

EVAL_TDS_JS = """
(rows, limit) => rows.slice(0, limit).map(row => {
  const tds = Array.from(row.querySelectorAll('td'));
  return tds.map(td => (td.innerText || '').trim());
})
"""

def parse_row_texts(texts: List[str]) -> Optional[Dict[str, Any]]:
    # нужно минимум 3 ячейки
    if not texts or len(texts) < 3:
        return None

    price = normalize_decimal(texts[0])
    vol_usdt = normalize_decimal(texts[1])
    trade_time = extract_time(texts[2])

    if price is None or vol_usdt is None or not trade_time:
        return None

    if price <= 0 or vol_usdt <= 0:
        return None

    vol_rub = price * vol_usdt

    return {
        "source": SOURCE,
        "symbol": SYMBOL,
        "price": q8_str(price),
        "volume_usdt": q8_str(vol_usdt),
        "volume_rub": q8_str(vol_rub),
        "trade_time": trade_time,
    }

async def scrape_window_fast(page: Page) -> List[Dict[str, Any]]:
    # не парсим, если пара не та
    if not await is_correct_pair(page):
        raise RuntimeError(f"Pair mismatch before scrape. url={page.url}")

    t0 = time.monotonic()

    last_err: Optional[Exception] = None
    payloads: Optional[List[List[str]]] = None

    for sel in ROWS_SELECTORS:
        try:
            await page.wait_for_selector(sel, timeout=int(SCRAPE_TIMEOUT_SECONDS * 1000))
            payloads = await page.eval_on_selector_all(sel, EVAL_TDS_JS, LIMIT)
            if payloads:
                break
        except Exception as e:
            last_err = e
            payloads = None

    if not payloads:
        if last_err:
            raise last_err
        return []

    out: List[Dict[str, Any]] = []
    for texts in payloads:
        t = parse_row_texts(texts)
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
            r = await client.post(url, headers=_sb_headers(), params=params, json=chunk)
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
        extra_http_headers={
            "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        },
    )

    # Правильный вызов add_init_script (одним аргументом)
    # Мы не знаем точные ключи localStorage сайта, поэтому ставим “самые вероятные”.
    await context.add_init_script(
        script=f"""
(() => {{
  try {{
    const slug = {json.dumps(PAIR_SLUG)};
    localStorage.setItem('pair', slug);
    localStorage.setItem('symbol', slug);
    localStorage.setItem('selectedPair', slug);
    localStorage.setItem('exchange_pair', slug);
    localStorage.setItem('last_pair', slug);
  }} catch(e) {{}}
}})();
"""
    )

    page = await context.new_page()
    page.set_default_timeout(10_000)

    # Старт + жёсткий lock пары
    await lock_pair(page)
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
    last_pair_check = 0.0

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
                    last_pair_check = time.monotonic()

                # периодический контроль “не унесло ли” на другую пару
                if time.monotonic() - last_pair_check >= 10:
                    if not await is_correct_pair(page):
                        logger.warning("Pair slipped. Re-locking...")
                        await lock_pair(page)
                    last_pair_check = time.monotonic()

                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url)
                    last_heartbeat = time.monotonic()

                if time.monotonic() - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await lock_pair(page)
                    last_reload = time.monotonic()

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Re-lock + retry...")
                    await lock_pair(page)
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
                    logger.info("Parsed %d new trades. Newest: %s", len(new_rows), json.dumps(new_rows[-1], ensure_ascii=False))
                    await supabase_upsert(new_rows)

                sleep_s = max(0.35, POLL_SECONDS + random.uniform(-0.15, 0.15))
                await asyncio.sleep(sleep_s)

            except asyncio.TimeoutError:
                logger.error("Timeout (%.1fs). Restarting browser session...", SCRAPE_TIMEOUT_SECONDS)
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
