import asyncio
import json
import logging
import os
import re
import subprocess
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

import httpx
from playwright.async_api import async_playwright, Page

# ───────────────────────── CONFIG ─────────────────────────

RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB")
SOURCE = os.getenv("SOURCE", "rapira")
LIMIT = int(os.getenv("LIMIT", "20"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

# Имя колонки времени в Supabase
SUPABASE_TIME_COLUMN = os.getenv("SUPABASE_TIME_COLUMN", "trade_time")

SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger("rapira-last-trades")

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")


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


# ───────────────────────── PARSING ─────────────────────────

TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
)

async def wait_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    for i in range(max_wait_seconds):
        try:
            rows = await page.query_selector_all(TRADE_ROWS_SELECTOR)
            if rows:
                logger.info("Found %d rows using selector '%s'.", len(rows), TRADE_ROWS_SELECTOR)
                return rows
        except Exception:
            pass

        logger.info("No rows yet (attempt %d/%d), waiting 1s...", i + 1, max_wait_seconds)
        await page.wait_for_timeout(1_000)

    logger.warning("No trade rows found.")
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


async def parse_trades(rows: List[Any], limit: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Возвращает:
      - rows_for_output: с volume_rub (для логов/print)
      - rows_for_db: без volume_rub (чтобы generated column не ломался)
    """
    out: List[Dict[str, Any]] = []
    db_rows: List[Dict[str, Any]] = []

    for row in rows:
        if len(out) >= limit:
            break

        parsed = await parse_row(row)
        if not parsed:
            continue

        price, volume_usdt, trade_time = parsed
        volume_rub = price * volume_usdt

        # округление
        price2 = round_money(price, 2)
        volu2 = round_money(volume_usdt, 2)
        rub2 = round_money(volume_rub, 2)

        row_out = {
            "source": SOURCE,
            "price": float(price2),
            "volume_usdt": float(volu2),
            "volume_rub": float(rub2),  # показываем, но в БД не отправляем
            SUPABASE_TIME_COLUMN: trade_time,
        }

        row_db = {
            "source": SOURCE,
            "price": float(price2),
            "volume_usdt": float(volu2),
            SUPABASE_TIME_COLUMN: trade_time,
        }

        out.append(row_out)
        db_rows.append(row_db)

    return out, db_rows


# ───────────────────────── SUPABASE ─────────────────────────

async def supabase_insert(rows_for_db: List[Dict[str, Any]]) -> None:
    if not rows_for_db:
        logger.info("No rows to insert.")
        return
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set; skipping insert.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=headers, json=rows_for_db)
        if r.status_code >= 300:
            logger.error("Supabase insert failed (%s): %s", r.status_code, r.text)
        else:
            logger.info("Inserted %d rows into '%s'.", len(rows_for_db), SUPABASE_TABLE)


# ───────────────────────── SCRAPER ─────────────────────────

async def scrape_rapira() -> Dict[str, Any]:
    ensure_playwright_browsers()

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
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
            logger.info("Opening Rapira page %s ...", RAPIRA_URL)
            await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(2_500)

            await accept_cookies_if_any(page)
            await ensure_last_trades_tab(page)
            await page.wait_for_timeout(1_500)

            rows = await wait_trade_rows(page, max_wait_seconds=40)

            rows_for_output, rows_for_db = await parse_trades(rows, limit=LIMIT)

            return {
                "ok": True,
                "source": SOURCE,
                "count": len(rows_for_output),
                "rows": rows_for_output,
                "_rows_for_db": rows_for_db,  # внутреннее — для вставки
            }
        finally:
            try:
                await page.close()
            except Exception:
                pass
            await context.close()
            await browser.close()


async def main() -> None:
    result = await scrape_rapira()

    # insert
    if result.get("ok") and result.get("_rows_for_db"):
        await supabase_insert(result["_rows_for_db"])

    # печать (без внутреннего поля)
    result.pop("_rows_for_db", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
