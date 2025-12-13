import asyncio
import json
import logging
import os
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import httpx
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# ───────────────────────── CONFIG ─────────────────────────

RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB")
SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

LIMIT = int(os.getenv("LIMIT", "50"))
POLL_SEC = float(os.getenv("POLL_SEC", "3"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
# поддержим оба варианта имени ключа (как у тебя в Render)
SUPABASE_KEY = (
    os.getenv("SUPABASE_KEY", "").strip()
    or os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
)

SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

# Имя колонки времени в Supabase (у тебя trade_time)
SUPABASE_TIME_COLUMN = os.getenv("SUPABASE_TIME_COLUMN", "trade_time")

# Колонки уникального ключа (должно совпадать с dedupe-unique index/constraint)
# exchange_trades_dedupe_uq: (source, symbol, trade_time, price, volume_usdt)
ON_CONFLICT = os.getenv(
    "ON_CONFLICT",
    "source,symbol,trade_time,price,volume_usdt"
)

# Если браузер ставится на build-стадии — ставь 1, чтобы не тратить время на каждом запуске
SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

# Какой таймзоной “живёт” сайт (для корректного отображения/локали)
TIMEZONE_ID = os.getenv("TIMEZONE_ID", "Europe/Moscow")
LOCALE = os.getenv("LOCALE", "ru-RU")

# Память дедупликации в процессе (чтобы меньше долбить Supabase одинаковыми пачками)
SEEN_MAX = int(os.getenv("SEEN_MAX", "5000"))

# ───────────────────────── LOGGING ─────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("rapira-worker")

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")

TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
)

# ───────────────────────── PLAYWRIGHT INSTALL ─────────────────────────

def ensure_playwright_browsers() -> None:
    if SKIP_BROWSER_INSTALL:
        logger.info("SKIP_BROWSER_INSTALL=1, skipping playwright install.")
        return

    try:
        logger.info("Ensuring Playwright Chromium is installed ...")
        result = subprocess.run(
            ["playwright", "install", "chromium", "chromium-headless-shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Playwright Chromium is installed (or already present).")
        else:
            logger.error(
                "playwright install returned code %s\nSTDOUT:\n%s\nSTDERR:\n%s",
                result.returncode,
                result.stdout,
                result.stderr,
            )
    except FileNotFoundError:
        logger.error("playwright CLI not found in PATH.")
    except Exception as e:
        logger.error("Unexpected error while installing Playwright browsers: %s", e)

# ───────────────────────── NORMALIZERS ─────────────────────────

def normalize_decimal(text: str) -> Optional[Decimal]:
    """
    '2 894.12' / '2 894,12' -> Decimal
    """
    t = text.strip()
    if not t:
        return None
    t = t.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    try:
        return Decimal(t)
    except (InvalidOperation, ValueError):
        return None

def extract_time(text: str) -> Optional[str]:
    """
    Вытаскивает время HH:MM:SS из текста.
    """
    m = TIME_RE.search(text.replace("\xa0", " "))
    if not m:
        return None
    s = m.group(0)
    hh, mm, ss = s.split(":")
    if len(hh) == 1:
        hh = "0" + hh
    return f"{hh}:{mm}:{ss}"

def round_money(x: Decimal, ndigits: int) -> Decimal:
    q = Decimal("1").scaleb(-ndigits)  # 10^-ndigits
    return x.quantize(q, rounding=ROUND_HALF_UP)

# ───────────────────────── PAGE ACTIONS ─────────────────────────

async def accept_cookies_if_any(page: Page) -> None:
    for label in ["Я согласен", "Принять", "Accept"]:
        try:
            btn = page.locator(f"text={label}")
            if await btn.count() > 0:
                logger.info("Found cookies banner, clicking '%s'...", label)
                await btn.first.click(timeout=5_000)
                await page.wait_for_timeout(300)
                return
        except Exception:
            pass

async def ensure_last_trades_tab(page: Page) -> None:
    for label in ["Последние сделки", "История сделок", "История"]:
        try:
            tab = page.locator(f"text={label}")
            if await tab.count() > 0:
                logger.info("Clicking tab '%s' ...", label)
                await tab.first.click(timeout=5_000)
                await page.wait_for_timeout(600)
                return
        except Exception:
            pass
    logger.info("Trades tab not clicked explicitly (maybe already active).")

async def wait_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    for i in range(max_wait_seconds):
        try:
            rows = await page.query_selector_all(TRADE_ROWS_SELECTOR)
            if rows:
                return rows
        except Exception:
            pass
        await page.wait_for_timeout(1_000)
    return []

# ───────────────────────── PARSING ─────────────────────────

async def parse_row(row: Any) -> Optional[Tuple[Decimal, Decimal, str]]:
    """
    Возвращает (price, volume_usdt, trade_time) или None.
    """
    try:
        tds = await row.query_selector_all("td")
        if len(tds) < 3:
            return None

        td_texts = [(await td.inner_text()).strip() for td in tds]

        # 1) время
        trade_time = None
        time_idx = None
        for idx, txt in enumerate(td_texts):
            t = extract_time(txt)
            if t:
                trade_time = t
                time_idx = idx
                break
        if not trade_time:
            return None

        # 2) price: пробуем td.text-success
        price: Optional[Decimal] = None
        try:
            price_td = await row.query_selector("td.text-success")
            if price_td:
                price_txt = (await price_td.inner_text()).strip()
                price = normalize_decimal(price_txt)
        except Exception:
            pass

        # 3) все числа (кроме времени)
        nums: List[Decimal] = []
        for idx, txt in enumerate(td_texts):
            if idx == time_idx:
                continue
            n = normalize_decimal(txt)
            if n is not None:
                nums.append(n)

        if len(nums) < 2:
            return None

        # 4) если price не нашли — эвристика 40..200
        if price is None:
            for n in nums:
                if Decimal("40") <= n <= Decimal("200"):
                    price = n
                    break
        if price is None:
            price = nums[0]

        # 5) volume_usdt — любое другое число (не price)
        volume_usdt: Optional[Decimal] = None
        for n in nums:
            if n != price:
                volume_usdt = n
                break
        if volume_usdt is None:
            volume_usdt = nums[1]

        if price <= 0 or volume_usdt <= 0:
            return None

        return price, volume_usdt, trade_time
    except Exception:
        return None

def make_dedupe_key(source: str, symbol: str, trade_time: str, price: Decimal, volume_usdt: Decimal) -> str:
    # нормализуем к 8 знакам, как в БД numeric(18,8)
    p = round_money(price, 8)
    v = round_money(volume_usdt, 8)
    return f"{source}|{symbol}|{trade_time}|{p}|{v}"

async def parse_trades(rows: List[Any], limit: int, seen: deque) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for row in rows:
        if len(out) >= limit:
            break

        parsed = await parse_row(row)
        if not parsed:
            continue

        price, volume_usdt, trade_time = parsed

        # дедупликация на уровне процесса (чтобы меньше дергать Supabase)
        key = make_dedupe_key(SOURCE, SYMBOL, trade_time, price, volume_usdt)
        if key in seen:
            continue

        volume_rub = price * volume_usdt

        # округление (в БД 8 знаков — отправим 8, а не float-2)
        price8 = round_money(price, 8)
        volu8 = round_money(volume_usdt, 8)
        rub8 = round_money(volume_rub, 8)

        out.append(
            {
                "source": SOURCE,
                "symbol": SYMBOL,
                "price": str(price8),         # лучше строкой, чтобы не терять точность
                "volume_usdt": str(volu8),
                "volume_rub": str(rub8),
                SUPABASE_TIME_COLUMN: trade_time,  # time without time zone
            }
        )

        seen.append(key)

    return out

# ───────────────────────── SUPABASE ─────────────────────────

async def supabase_insert(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logger.info("No new rows to insert.")
        return
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set; skipping insert.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    params = {"on_conflict": ON_CONFLICT}

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        # важно: игнорировать дубликаты по on_conflict
        "Prefer": "return=minimal,resolution=ignore-duplicates",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, params=params, json=rows)

        # 201/204 — ок
        if r.status_code in (200, 201, 204):
            logger.info("Inserted (or ignored duplicates) %d rows into '%s'.", len(rows), SUPABASE_TABLE)
            return

        # если Supabase всё равно вернул 409 — считаем допустимым (дубликаты)
        if r.status_code == 409:
            logger.info("Supabase returned 409 (duplicates). Treat as OK. Body: %s", r.text)
            return

        logger.error("Supabase insert failed (%s): %s", r.status_code, r.text)

# ───────────────────────── WORKER LOOP ─────────────────────────

@dataclass
class BrowserBundle:
    browser: Browser
    context: BrowserContext
    page: Page

async def open_browser() -> BrowserBundle:
    p = await async_playwright().start()

    browser = await p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )

    context = await browser.new_context(
        viewport={"width": 1440, "height": 810},
        locale=LOCALE,
        timezone_id=TIMEZONE_ID,
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    page = await context.new_page()

    # сохраним ссылку на playwright, чтобы корректно закрыть
    page._pw = p  # type: ignore[attr-defined]
    return BrowserBundle(browser=browser, context=context, page=page)

async def close_browser(bundle: BrowserBundle) -> None:
    try:
        await bundle.page.close()
    except Exception:
        pass
    try:
        await bundle.context.close()
    except Exception:
        pass
    try:
        await bundle.browser.close()
    except Exception:
        pass
    try:
        p = getattr(bundle.page, "_pw", None)
        if p:
            await p.stop()
    except Exception:
        pass

async def ensure_page_ready(page: Page) -> None:
    logger.info("Opening Rapira page %s ...", RAPIRA_URL)
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(2_000)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)
    await page.wait_for_timeout(1_000)

async def worker_loop() -> None:
    ensure_playwright_browsers()

    seen = deque(maxlen=SEEN_MAX)
    bundle: Optional[BrowserBundle] = None

    backoff = 2.0
    backoff_max = 60.0

    while True:
        try:
            if bundle is None:
                bundle = await open_browser()
                await ensure_page_ready(bundle.page)
                backoff = 2.0

            # мягкий refresh, чтобы таблица точно обновлялась
            try:
                await bundle.page.reload(wait_until="domcontentloaded", timeout=60_000)
                await bundle.page.wait_for_timeout(1_000)
                await accept_cookies_if_any(bundle.page)
                await ensure_last_trades_tab(bundle.page)
                await bundle.page.wait_for_timeout(800)
            except Exception:
                # если reload сломался — пересоздадим браузер
                raise

            rows = await wait_trade_rows(bundle.page, max_wait_seconds=20)
            if not rows:
                logger.warning("No trade rows found this iteration.")
                await asyncio.sleep(POLL_SEC)
                continue

            trades = await parse_trades(rows, limit=LIMIT, seen=seen)

            # лог для контроля
            if trades:
                logger.info("Parsed %d new trades. First: %s", len(trades), json.dumps(trades[0], ensure_ascii=False))
            else:
                logger.info("No new trades after dedupe.")

            await supabase_insert(trades)

            await asyncio.sleep(POLL_SEC)

        except asyncio.CancelledError:
            logger.info("Worker cancelled, shutting down.")
            break

        except Exception as e:
            logger.error("Worker error: %s", e)

            # пересоздаём браузер на любой серьёзной ошибке
            if bundle is not None:
                await close_browser(bundle)
                bundle = None

            logger.info("Retrying after %.1fs ...", backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, backoff_max)

    if bundle is not None:
        await close_browser(bundle)

def main() -> None:
    asyncio.run(worker_loop())

if __name__ == "__main__":
    main()
