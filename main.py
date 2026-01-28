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

import httpx
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# ───────────────────────── CONFIG ─────────────────────────

# ВАЖНО: по ответу поддержки — статическая ссылка именно такая:
PAIR_URL = os.getenv("PAIR_URL", "https://rapira.net/exchange/USDT_RUB")
PAIR_CODE = os.getenv("PAIR_CODE", "USDT_RUB")
PAIR_TEXT = os.getenv("PAIR_TEXT", "USDT/RUB")

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

ON_CONFLICT = os.getenv("ON_CONFLICT", "source,symbol,trade_time,price,volume_usdt")

SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))

LOCK_ATTEMPTS = int(os.getenv("LOCK_ATTEMPTS", "10"))

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


def url_is_pair(u: str) -> bool:
    u = (u or "").lower()
    return f"/exchange/{PAIR_CODE.lower()}" in u


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
    for label in ["OK", "Я согласен", "Принять", "Accept"]:
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
    try:
        tab = page.locator("text=Последние сделки")
        if await tab.count() > 0:
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
    except Exception:
        pass


async def verify_pair_ui(page: Page) -> bool:
    try:
        title = page.locator("span.fw-bold.fs-6").first
        txt = (await title.inner_text(timeout=4_000)).replace(" ", "").upper()
        return ("USDT" in txt and "/RUB" in txt)
    except Exception:
        return False


async def open_pair_menu(page: Page) -> None:
    await page.wait_for_selector("div.trading-pair-block-menu", timeout=15_000)
    await page.locator("div.trading-pair-block-menu").first.click(timeout=10_000, no_wait_after=True)
    await page.wait_for_timeout(200)


async def click_pair_usdt_rub(page: Page) -> None:
    await page.wait_for_selector("tr.trading-pair-row", timeout=15_000)
    rows = page.locator("tr.trading-pair-row")
    target = rows.filter(has_text="USDT").filter(has_text="/RUB").first
    if await target.count() == 0:
        target = rows.filter(has_text="USDT/RUB").first
    if await target.count() == 0:
        raise RuntimeError("USDT/RUB row not found in pair list")
    await target.click(timeout=10_000, no_wait_after=True)
    await page.wait_for_timeout(350)


async def hard_go_pair(page: Page) -> None:
    # Жёстко идём на “статическую” ссылку из поддержки
    await page.goto(PAIR_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(900)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)


async def lock_pair(page: Page) -> None:
    """
    1) Сначала всегда пробуем "статическую" ссылку PAIR_URL.
    2) Если сайт всё равно уводит/не тот UI — пробуем через меню.
    """
    for attempt in range(1, LOCK_ATTEMPTS + 1):
        try:
            await hard_go_pair(page)

            # если URL совпал — отлично
            if url_is_pair(page.url) and await verify_pair_ui(page):
                logger.info("Pair locked via static url: %s", page.url)
                return

            # fallback: выбираем через меню на текущей странице (даже если это BTC_USDT)
            await open_pair_menu(page)
            await click_pair_usdt_rub(page)

            # подождём URL либо UI
            try:
                await page.wait_for_url(f"**/{PAIR_CODE}", timeout=15_000)
            except Exception:
                pass

            await ensure_last_trades_tab(page)

            if url_is_pair(page.url) and await verify_pair_ui(page):
                logger.info("Pair locked via menu: %s", page.url)
                return

            raise RuntimeError(f"still not locked (url={page.url})")

        except Exception as e:
            logger.warning(
                "Lock attempt %d failed. url=%s err=%s",
                attempt, page.url, str(e)[:180]
            )
            await page.wait_for_timeout(600 + random.randint(0, 400))

    raise RuntimeError(f"Cannot lock pair {PAIR_CODE}. final_url={page.url}")

# ───────────────────────── PARSING ─────────────────────────

TRADE_ROWS_SELECTOR_1 = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
)

TRADE_ROWS_SELECTOR_2 = (
    "div.table-responsive.table-orders "
    "table tbody tr"
)

EVAL_JS = """
(rows, limit) => rows.slice(0, limit).map(row => {
  const tds = Array.from(row.querySelectorAll('td'));
  const texts = tds.map(td => (td.innerText || '').trim());
  const priceTd = row.querySelector('td.text-success');
  const priceHint = priceTd ? (priceTd.innerText || '').trim() : null;
  return { texts, priceHint };
})
"""

def parse_row_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        texts: List[str] = payload.get("texts") or []
        if len(texts) < 3:
            return None

        trade_time = None
        time_idx = None
        for idx, txt in enumerate(texts):
            t = extract_time(txt)
            if t:
                trade_time = t
                time_idx = idx
                break
        if not trade_time:
            return None

        price: Optional[Decimal] = None
        price_hint = payload.get("priceHint")
        if price_hint:
            price = normalize_decimal(price_hint)

        nums: List[Decimal] = []
        for idx, txt in enumerate(texts):
            if idx == time_idx:
                continue
            n = normalize_decimal(txt)
            if n is not None:
                nums.append(n)
        if len(nums) < 2:
            return None

        # эвристика цены для USDT/RUB
        if price is None:
            for n in nums:
                if Decimal("40") <= n <= Decimal("200"):
                    price = n
                    break
        if price is None:
            price = nums[0]

        volume_usdt: Optional[Decimal] = None
        for n in nums:
            if n != price:
                volume_usdt = n
                break
        if volume_usdt is None:
            volume_usdt = nums[1]

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
    except Exception:
        return None


async def _eval_rows(page: Page, selector: str) -> List[Dict[str, Any]]:
    await page.wait_for_selector(selector, timeout=int(SCRAPE_TIMEOUT_SECONDS * 1000))
    payloads = await page.eval_on_selector_all(selector, EVAL_JS, LIMIT)
    out: List[Dict[str, Any]] = []
    for p in payloads:
        t = parse_row_payload(p)
        if t:
            out.append(t)
    return out


async def scrape_window_fast(page: Page) -> List[Dict[str, Any]]:
    t0 = time.monotonic()
    try:
        out = await _eval_rows(page, TRADE_ROWS_SELECTOR_1)
    except Exception:
        out = await _eval_rows(page, TRADE_ROWS_SELECTOR_2)

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

    # корректный вызов (1 аргумент) + чистим storage, чтобы не восстанавливалось "последнее"
    await context.add_init_script(
        """() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e) {} }"""
    )

    page = await context.new_page()
    page.set_default_timeout(10_000)

    # ключ: начинаем сразу со "статической" ссылки
    await hard_go_pair(page)
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
    last_click_tab = 0.0

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

                # если куда-то “уплыли” — возвращаемся на PAIR_URL
                if not url_is_pair(page.url) or not await verify_pair_ui(page):
                    logger.warning("Pair guard failed (url=%s). Re-locking...", page.url)
                    await lock_pair(page)

                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url)
                    last_heartbeat = time.monotonic()

                if time.monotonic() - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await hard_go_pair(page)
                    await lock_pair(page)
                    last_reload = time.monotonic()

                if time.monotonic() - last_click_tab >= 15:
                    await ensure_last_trades_tab(page)
                    last_click_tab = time.monotonic()

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Re-lock pair and retry...")
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
