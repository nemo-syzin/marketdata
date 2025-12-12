import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

import httpx
from playwright.async_api import async_playwright, Page

RAPIRA_URL = "https://rapira.net/exchange/USDT_RUB"
SOURCE = "rapira"
MAX_TRADES = int(os.getenv("MAX_TRADES", "20"))

SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").strip()
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or "").strip()
SUPABASE_TABLE = (os.getenv("SUPABASE_TABLE") or "exchange_trades").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


def ensure_playwright_browsers() -> None:
    try:
        logging.info("Ensuring Playwright Chromium is installed ...")
        result = subprocess.run(
            ["playwright", "install", "chromium", "chromium-headless-shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logging.info("Playwright Chromium is installed (or already present).")
        else:
            logging.error(
                "playwright install returned code %s\nSTDOUT:\n%s\nSTDERR:\n%s",
                result.returncode,
                result.stdout,
                result.stderr,
            )
    except Exception as e:
        logging.error("Playwright install error: %s", e)


async def close_silently(page: Page) -> None:
    try:
        await page.close()
    except Exception:
        pass


async def accept_cookies_if_any(page: Page) -> None:
    try:
        btn = page.locator("text=Я согласен")
        if await btn.count() > 0:
            logging.info("Found cookies banner, clicking 'Я согласен'...")
            await btn.first.click(timeout=5_000)
    except Exception as e:
        logging.info("Ignoring cookies click error: %s", e)


async def ensure_last_trades_tab(page: Page) -> None:
    for text in ["Последние сделки", "История сделок", "История"]:
        try:
            tab = page.locator(f"text={text}")
            if await tab.count() > 0:
                logging.info("Clicking tab '%s' ...", text)
                await tab.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception:
            pass


async def poll_for_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    selectors = [
        "div.table-responsive.table-orders table.table-row-dashed tbody tr.table-orders-row",
        "div.table-responsive.table-orders table.table-row-dashed tbody tr",
        "table.table-row-dashed tbody tr.table-orders-row",
        "table.table-row-dashed tbody tr",
        "tr.table-orders-row",
    ]

    for i in range(max_wait_seconds):
        for selector in selectors:
            rows = await page.query_selector_all(selector)
            if rows:
                logging.info("Found %d rows using selector '%s'.", len(rows), selector)
                return rows
        logging.info("No rows yet (%d/%d). Waiting...", i + 1, max_wait_seconds)
        await page.wait_for_timeout(1_000)

    return []


def _normalize_num(text: str) -> float:
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


def _looks_like_time(s: str) -> bool:
    s = s.strip()
    return ":" in s and len(s) <= 10


async def parse_trades_to_db_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Возвращает список строк для БД:
      price, volume_usdt, volume_rub, time_text, source
    """
    out: List[Dict[str, Any]] = []

    for row in rows:
        cells = await row.query_selector_all("th, td")
        if len(cells) < 3:
            continue

        # Тянем тексты всех ячеек
        texts = [(await c.inner_text()).strip() for c in cells]
        texts = [t for t in texts if t]  # убираем пустое

        if len(texts) < 3:
            continue

        price_text = texts[0]
        vol_text = texts[1]

        # Время берём как "последнюю ячейку, похожую на время", иначе просто третью
        time_text: str = texts[2]
        for t in reversed(texts):
            if _looks_like_time(t):
                time_text = t
                break

        # Пропускаем заголовки
        low = [x.lower() for x in texts[:3]]
        if "цена" in low[0] or "объ" in low[1] or "время" in low[2]:
            continue

        try:
            price = _normalize_num(price_text)
            volume_usdt = _normalize_num(vol_text)
        except Exception:
            continue

        volume_rub = price * volume_usdt

        out.append(
            {
                "source": SOURCE,
                "price": price,
                "volume_usdt": volume_usdt,
                "volume_rub": volume_rub,
                "time_text": time_text,
            }
        )

        if len(out) >= MAX_TRADES:
            break

    return out


async def insert_rows_to_supabase(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        logging.info("No rows to insert.")
        return

    if not SUPABASE_URL or not SUPABASE_KEY:
        logging.warning("SUPABASE_URL / SUPABASE_KEY not set — skipping DB insert.")
        return

    url = f"{SUPABASE_URL.rstrip('/')}/rest/v1/{SUPABASE_TABLE}"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.post(url, headers=headers, json=rows)
        if 200 <= r.status_code < 300:
            logging.info("Inserted %d rows into Supabase (%s).", len(rows), r.status_code)
            return
        logging.error("Supabase insert failed (%s): %s", r.status_code, r.text[:2000])


async def scrape_rapira_and_save() -> Dict[str, Any]:
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
        )
        page = await context.new_page()

        try:
            logging.info("Opening Rapira page %s ...", RAPIRA_URL)
            await page.goto(RAPIRA_URL, wait_until="networkidle", timeout=60_000)
            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)
            await ensure_last_trades_tab(page)
            await page.wait_for_timeout(2_000)

            rows = await poll_for_trade_rows(page, max_wait_seconds=40)
            parsed = await parse_trades_to_db_rows(rows)

            await insert_rows_to_supabase(parsed)

            return {
                "ok": True,
                "source": SOURCE,
                "count": len(parsed),
                "rows": parsed,
            }
        finally:
            await close_silently(page)
            await context.close()
            await browser.close()


async def main() -> None:
    res = await scrape_rapira_and_save()
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
