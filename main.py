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

RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB")
SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

# On Render, large LIMIT is often slow.
LIMIT = int(os.getenv("LIMIT", "150"))

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))

SCRAPE_TIMEOUT_SECONDS = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "25"))
UPSERT_TIMEOUT_SECONDS = float(os.getenv("UPSERT_TIMEOUT_SECONDS", "25"))
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "30"))
RELOAD_EVERY_SECONDS = float(os.getenv("RELOAD_EVERY_SECONDS", "600"))

# If no successful DB upsert for this long -> exit(1) so Render restarts the worker.
STALL_RESTART_SECONDS = float(os.getenv("STALL_RESTART_SECONDS", "3600"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

ON_CONFLICT = "source,symbol,trade_time,price,volume_usdt"

# If you want to forbid runtime install, set SKIP_BROWSER_INSTALL=1
SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))

# Optimization: block heavy resources (recommended for Render).
BLOCK_HEAVY_RESOURCES = os.getenv("BLOCK_HEAVY_RESOURCES", "1") == "1"

# ───────────────────────── LOGGING ─────────────────────────

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
    """Runtime install as a last resort. Avoid calling on non-install-related errors."""
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
    for label in ["Я согласен", "Принять", "Accept"]:
        try:
            btn = page.locator(f"text={label}")
            if await btn.count() > 0:
                logger.info("Found cookies banner, clicking '%s'...", label)
                await btn.first.click(timeout=5_000, no_wait_after=True)
                await page.wait_for_timeout(300)
                return
        except Exception:
            pass


async def ensure_last_trades_tab(page: Page) -> None:
    """
    Try to activate "Последние сделки". Site may change, so we keep this defensive.
    """
    # First try ARIA role tabs (more stable when available)
    try:
        tab = page.get_by_role("tab", name=re.compile(r"Последние сделки", re.I))
        if await tab.count() > 0:
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
            return
    except Exception:
        pass

    # Fallback: text locator
    try:
        tab = page.locator("text=Последние сделки")
        if await tab.count() > 0:
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
    except Exception:
        pass

# ───────────────────────── PARSING ─────────────────────────

TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
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

        # time
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

        # numerics except time
        nums: List[Decimal] = []
        for idx, txt in enumerate(texts):
            if idx == time_idx:
                continue
            n = normalize_decimal(txt)
            if n is not None:
                nums.append(n)
        if len(nums) < 2:
            return None

        # heuristic for price if not found by class
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


async def scrape_window_fast(page: Page) -> List[Dict[str, Any]]:
    """
    Robust scrape:
    - Wait for rows to be attached (may be hidden by tabs/overlays)
    - Wait until rows contain a HH:MM:SS pattern (content actually loaded)
    - Then one eval_on_selector_all call
    """
    t0 = time.monotonic()
    timeout_ms = int(SCRAPE_TIMEOUT_SECONDS * 1000)

    # Ensure tab periodically (caller also does it, but helps after reload)
    await ensure_last_trades_tab(page)

    # 1) rows exist in DOM
    await page.wait_for_selector(TRADE_ROWS_SELECTOR, state="attached", timeout=timeout_ms)

    # 2) content is present (at least one time pattern in innerText)
    await page.wait_for_function(
        """(sel) => {
            const rows = Array.from(document.querySelectorAll(sel));
            return rows.some(r => /\\b\\d{1,2}:\\d{2}:\\d{2}\\b/.test((r.innerText || '').trim()));
        }""",
        TRADE_ROWS_SELECTOR,
        timeout=timeout_ms,
    )

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
                raise  # bubble up -> triggers retry/backoff

            if r.status_code >= 300:
                logger.error("Supabase upsert failed (%s): %s", r.status_code, r.text)
                raise RuntimeError(f"Supabase upsert failed {r.status_code}")

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

    if BLOCK_HEAVY_RESOURCES:
        async def _route(route):
            req = route.request
            if req.resource_type in {"image", "media", "font"}:
                await route.abort()
            else:
                await route.continue_()
        await context.route("**/*", _route)

    page = await context.new_page()

    # keep default for "normal" actions; scraping has its own explicit waits
    page.set_default_timeout(10_000)

    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(800)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)
    await page.wait_for_timeout(300)
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

    # restart guard: "no successful DB delivery for too long"
    last_success_upsert = time.monotonic()

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
                        # IMPORTANT: install only when error indicates missing browser executable
                        if (not SKIP_BROWSER_INSTALL) and _should_force_install(e):
                            _playwright_install()
                            browser, context, page = await open_browser(pw)
                        else:
                            raise

                    backoff = 2.0
                    last_reload = time.monotonic()
                    last_heartbeat = time.monotonic()

                now = time.monotonic()

                # Stall watchdog: no successful upsert for too long -> exit to trigger Render restart
                if now - last_success_upsert >= STALL_RESTART_SECONDS:
                    logger.error(
                        "STALL: no successful DB upsert for %.0fs (threshold=%.0fs). Exiting(1) to trigger Render restart.",
                        now - last_success_upsert,
                        STALL_RESTART_SECONDS,
                    )
                    raise SystemExit(1)

                if now - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info(
                        "Heartbeat: alive. seen=%d, since_last_upsert=%.0fs",
                        len(seen),
                        now - last_success_upsert,
                    )
                    last_heartbeat = now

                if now - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(800)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
                    last_reload = time.monotonic()

                if now - last_click_tab >= 15:
                    await ensure_last_trades_tab(page)
                    last_click_tab = time.monotonic()

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Reloading page...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(800)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
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
                    last_success_upsert = time.monotonic()

                sleep_s = max(0.35, POLL_SECONDS + random.uniform(-0.15, 0.15))
                await asyncio.sleep(sleep_s)

            except SystemExit:
                # Let process exit so Render restarts it.
                raise

            except asyncio.TimeoutError:
                logger.error(
                    "Timeout: scrape_window exceeded %.1fs. Restarting browser session...",
                    SCRAPE_TIMEOUT_SECONDS
                )
                await safe_close(browser, context, page)
                browser = context = page = None

            except Exception as e:
                logger.error("Worker error: %s", e)

                # IMPORTANT: do NOT install browsers on every error
                if (not SKIP_BROWSER_INSTALL) and _should_force_install(e):
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
