import asyncio
import json
import logging
import os
import random
import re
import subprocess
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

# Сколько строк брать с UI за проход (держи с запасом!)
LIMIT = int(os.getenv("LIMIT", "400"))

# Как часто опрашивать (сек). Чем меньше — тем меньше шанс пропусков.
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

# Должно совпадать с unique index:
# create unique index ... (source, symbol, trade_time, price, volume_usdt)
ON_CONFLICT = "source,symbol,trade_time,price,volume_usdt"

# Можно оставить, но код сам поставит браузер, если его нет
SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

# Сколько последних ключей помнить локально (защита от дублей/переупорядочивания)
SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("rapira-worker")

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")

# Точность под numeric(18,8)
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


def _playwright_install() -> None:
    """
    Runtime-установка браузеров. Это НЕ требует менять команды Render.
    Ставим и chromium, и chromium-headless-shell, чтобы не ловить ошибку по headless_shell.
    """
    logger.warning("Installing Playwright browsers (runtime)...")
    try:
        r = subprocess.run(
            ["python", "-m", "playwright", "install", "chromium", "chromium-headless-shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.error("playwright install failed (%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
                         r.returncode, r.stdout, r.stderr)
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
        or "ms-playwright" in s and "doesn't exist" in s
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
                await btn.first.click(timeout=5_000)
                await page.wait_for_timeout(300)
                return
        except Exception:
            pass


async def ensure_last_trades_tab(page: Page) -> None:
    """
    Важно: не кликаем подряд 'История'/'История сделок', чтобы не уводило в другой UI.
    Стараемся держаться строго на 'Последние сделки'.
    """
    try:
        tab = page.locator("text=Последние сделки")
        if await tab.count() > 0:
            await tab.first.click(timeout=5_000)
            await page.wait_for_timeout(250)
    except Exception:
        pass


# ───────────────────────── PARSING ─────────────────────────

TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
)

async def wait_trade_rows(page: Page, max_wait_seconds: int = 25) -> List[Any]:
    for _ in range(max_wait_seconds):
        try:
            rows = await page.query_selector_all(TRADE_ROWS_SELECTOR)
            if rows:
                return rows
        except Exception:
            pass
        await page.wait_for_timeout(1_000)
    return []


async def parse_row(row: Any) -> Optional[Dict[str, Any]]:
    """
    Возвращает dict для вставки в БД или None.
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

        # 2) цена (пытаемся td.text-success)
        price: Optional[Decimal] = None
        try:
            price_td = await row.query_selector("td.text-success")
            if price_td:
                price = normalize_decimal((await price_td.inner_text()).strip())
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

        # 4) если цену не нашли — эвристика 40..200
        if price is None:
            for n in nums:
                if Decimal("40") <= n <= Decimal("200"):
                    price = n
                    break
        if price is None:
            price = nums[0]

        # 5) объем USDT — другое число
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

        # В БД отправляем как строки с 8 знаками (точно под numeric(18,8))
        return {
            "source": SOURCE,
            "symbol": SYMBOL,
            "price": q8_str(price),
            "volume_usdt": q8_str(volume_usdt),
            "volume_rub": q8_str(volume_rub),
            "trade_time": trade_time,  # time without tz
        }
    except Exception:
        return None


async def scrape_window(page: Page) -> List[Dict[str, Any]]:
    rows = await wait_trade_rows(page, max_wait_seconds=25)
    if not rows:
        return []

    out: List[Dict[str, Any]] = []
    # rows на странице обычно идут "новые сверху"
    for row in rows[:LIMIT]:
        t = await parse_row(row)
        if t:
            out.append(t)
    return out


# ───────────────────────── SUPABASE ─────────────────────────

async def supabase_upsert(rows: List[Dict[str, Any]]) -> None:
    if not rows:
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
        # ключевое: не падать на unique, а игнорировать дубли
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, params=params, json=rows)
        if r.status_code >= 300:
            logger.error("Supabase upsert failed (%s): %s", r.status_code, r.text)
        else:
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
    page = await context.new_page()
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(1_200)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)
    await page.wait_for_timeout(600)
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
    seen_q: Deque[TradeKey] = deque(maxlen=SEEN_MAX)

    backoff = 2.0

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
                        # Если браузера нет — ставим и повторяем
                        if (not SKIP_BROWSER_INSTALL) or _should_force_install(e):
                            _playwright_install()
                            browser, context, page = await open_browser(pw)
                        else:
                            raise
                    backoff = 2.0

                # держим вкладку "Последние сделки"
                await ensure_last_trades_tab(page)

                window = await scrape_window(page)
                if not window:
                    logger.warning("No rows parsed. Reloading page...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(1_200)
                    await accept_cookies_if_any(page)
                    await ensure_last_trades_tab(page)
                    await asyncio.sleep(max(0.5, POLL_SECONDS))
                    continue

                # Чтобы не зависеть от last_seen (который иногда "теряется"),
                # фильтруем по множеству seen.
                # Вставляем в порядке "старые -> новые": для этого переворачиваем окно.
                new_rows: List[Dict[str, Any]] = []
                for t in reversed(window):
                    k = trade_key(t)
                    if k in seen:
                        continue
                    new_rows.append(t)
                    seen.add(k)
                    seen_q.append(k)

                # чистим set, когда deque вытесняет элементы
                # (deque maxlen сам выкидывает слева, но нам нужно синхронизировать set)
                while len(seen_q) == seen_q.maxlen and len(seen) > len(seen_q):
                    # редкий случай рассинхронизации, перегенерируем set
                    seen = set(seen_q)

                if new_rows:
                    logger.info("Parsed %d new trades. Newest: %s", len(new_rows), json.dumps(new_rows[-1], ensure_ascii=False))
                    await supabase_upsert(new_rows)
                else:
                    logger.info("No new trades.")

                # джиттер, чтобы не попадать в жесткий ритм
                sleep_s = max(0.35, POLL_SECONDS + random.uniform(-0.15, 0.15))
                await asyncio.sleep(sleep_s)

            except Exception as e:
                logger.error("Worker error: %s", e)
                # Если браузер пропал в рантайме — ставим и попробуем заново
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
