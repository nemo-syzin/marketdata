import asyncio
import json
import logging
import os
import random
import re
import subprocess
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import httpx
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# ───────────────────────── CONFIG ─────────────────────────

RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB")

SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

# Сколько строк брать с UI за один проход (должно быть с запасом!)
LIMIT = int(os.getenv("LIMIT", "200"))

# Как часто опрашивать (сек). Чем меньше — тем меньше шанс пропусков.
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "2.0"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

# Конфликт-ключ должен совпадать с твоим unique index
ON_CONFLICT = "source,symbol,trade_time,price,volume_usdt"

# Playwright install в рантайме (можно отключить, если ставишь в Build Command)
SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("rapira-worker")

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")

# Точность под numeric(18,8)
Q8 = Decimal("0.00000001")


# ───────────────────────── HELPERS ─────────────────────────

def ensure_playwright_browsers() -> None:
    """
    На Render надежнее ставить браузер на build step:
      pip install -r requirements.txt && playwright install chromium --with-deps
    Но этот runtime-установщик тоже работает (медленнее).
    """
    if SKIP_BROWSER_INSTALL:
        logger.info("SKIP_BROWSER_INSTALL=1, skipping playwright install.")
        return

    try:
        logger.info("Ensuring Playwright Chromium is installed ...")
        result = subprocess.run(
            ["playwright", "install", "chromium"],
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


def normalize_decimal(text: str) -> Optional[Decimal]:
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
    m = TIME_RE.search(text.replace("\xa0", " "))
    if not m:
        return None
    s = m.group(0)
    hh, mm, ss = s.split(":")
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
                await page.wait_for_timeout(500)
                return
        except Exception:
            pass
    logger.info("Trades tab not clicked explicitly (maybe already active).")


# ───────────────────────── PARSING ─────────────────────────

TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
)

async def wait_trade_rows(page: Page, max_wait_seconds: int = 30) -> List[Any]:
    for i in range(max_wait_seconds):
        try:
            rows = await page.query_selector_all(TRADE_ROWS_SELECTOR)
            if rows:
                return rows
        except Exception:
            pass
        await page.wait_for_timeout(1_000)
    return []


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

        # 2) цена — пробуем td.text-success
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

        # 5) volume_usdt — другое число
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


async def scrape_once(page: Page) -> List[Dict[str, Any]]:
    rows = await wait_trade_rows(page, max_wait_seconds=30)
    if not rows:
        logger.warning("No trade rows found.")
        return []

    out: List[Dict[str, Any]] = []
    for row in rows[:LIMIT]:
        parsed = await parse_row(row)
        if not parsed:
            continue

        price, volume_usdt, trade_time = parsed
        volume_rub = price * volume_usdt

        # В БД отправляем как строки с 8 знаками (точно под numeric(18,8))
        price_s = q8_str(price)
        volu_s = q8_str(volume_usdt)
        rub_s = q8_str(volume_rub)

        out.append(
            {
                "source": SOURCE,
                "symbol": SYMBOL,
                "price": price_s,
                "volume_usdt": volu_s,
                "volume_rub": rub_s,
                "trade_time": trade_time,  # time without tz
            }
        )

    return out


def trade_key(t: Dict[str, Any]) -> TradeKey:
    return TradeKey(
        source=t["source"],
        symbol=t["symbol"],
        trade_time=t["trade_time"],
        price=t["price"],
        volume_usdt=t["volume_usdt"],
    )


def filter_new_trades(trades: List[Dict[str, Any]], last_seen: Optional[TradeKey]) -> Tuple[List[Dict[str, Any]], Optional[TradeKey], bool]:
    """
    trades: отсортированы как на странице (обычно новые сверху)
    Возвращает:
      - new_trades (в правильном порядке для вставки: старые -> новые)
      - updated_last_seen (самая новая запись из текущего окна)
      - gap_detected: last_seen не найден в окне (возможен пропуск)
    """
    if not trades:
        return [], last_seen, False

    keys = [trade_key(t) for t in trades]
    newest = keys[0]

    if last_seen is None:
        # первый запуск: вставим всё окно (от старых к новым)
        return list(reversed(trades)), newest, False

    # ищем last_seen в текущем окне
    try:
        idx = keys.index(last_seen)
        # всё, что выше idx — новые
        new_part = trades[:idx]
        return list(reversed(new_part)), newest, False
    except ValueError:
        # last_seen не нашли => окно “уехало”, пропуск возможен
        return list(reversed(trades)), newest, True


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
        # ключевая часть: не падать на unique, а игнорировать дубликаты
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, params=params, json=rows)
        if r.status_code >= 300:
            logger.error("Supabase upsert failed (%s): %s", r.status_code, r.text)
        else:
            logger.info("Inserted (or ignored duplicates) %d rows into '%s'.", len(rows), SUPABASE_TABLE)


# ───────────────────────── WORKER LOOP ─────────────────────────

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
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    page = await context.new_page()
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(1_500)
    await accept_cookies_if_any(page)
    await ensure_last_trades_tab(page)
    await page.wait_for_timeout(800)
    return browser, context, page


async def worker() -> None:
    ensure_playwright_browsers()

    last_seen: Optional[TradeKey] = None
    backoff = 2.0

    async with async_playwright() as pw:
        browser: Optional[Browser] = None
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None

        while True:
            try:
                if page is None:
                    logger.info("Starting browser session...")
                    browser, context, page = await open_browser(pw)
                    backoff = 2.0

                # на всякий случай убеждаемся, что вкладка правильная
                await ensure_last_trades_tab(page)

                trades = await scrape_once(page)
                new_trades, last_seen_new, gap = filter_new_trades(trades, last_seen)

                if gap:
                    logger.warning(
                        "Possible gap detected: last_seen not found in current window. "
                        "Increase LIMIT and/or decrease POLL_SECONDS to reduce misses."
                    )

                if new_trades:
                    first = json.dumps(new_trades[-1], ensure_ascii=False)
                    logger.info("Parsed %d new trades. Last (newest): %s", len(new_trades), first)
                    await supabase_upsert(new_trades)
                else:
                    logger.info("No new trades.")

                last_seen = last_seen_new

                # небольшой джиттер, чтобы не попадать в строгие интервалы
                sleep_s = max(0.3, POLL_SECONDS + random.uniform(-0.2, 0.2))
                await asyncio.sleep(sleep_s)

            except Exception as e:
                logger.error("Worker error: %s", e)
                logger.info("Retrying after %.1fs ...", backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

                # перезапускаем браузерную сессию
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

                browser = context = page = None


def main() -> None:
    asyncio.run(worker())


if __name__ == "__main__":
    main()
