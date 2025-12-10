import asyncio
import json
import logging
from typing import Any, Dict, List

from playwright.async_api import async_playwright, Page

RAPIRA_URL = "https://rapira.net/trading/usdt-rub"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


async def close_silently(page: Page) -> None:
    try:
        await page.close()
    except Exception:
        pass


async def accept_cookies_if_any(page: Page) -> None:
    """
    На Rapira иногда показывается баннер cookies с кнопкой «Я согласен».
    Если он есть — нажимаем и игнорируем все ошибки.
    """
    try:
        btn = page.locator("text=Я согласен")
        if await btn.count() > 0:
            logging.info("Found cookies banner, clicking 'Я согласен'...")
            await btn.first.click(timeout=5_000)
        else:
            logging.info("No 'Я согласен' (cookies) button found.")
    except Exception as e:
        logging.info("Ignoring cookies click error: %s", e)


async def ensure_history_tab(page: Page) -> None:
    """
    В центральном блоке есть вкладки Книга / История.
    Пробуем кликнуть по «История», но если элемент не найден —
    ничего страшного: нам подойдёт и правая колонка «Последние сделки».
    """
    try:
        history_tab = page.locator("text=История")
        if await history_tab.count() > 0:
            logging.info("Clicking 'История' tab...")
            await history_tab.first.click(timeout=5_000)
        else:
            logging.info("History tab element not found explicitly, maybe already active.")
    except Exception as e:
        logging.info("Failed to click 'История' tab explicitly, maybe it's already active: %s", e)


async def poll_for_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    """
    Периодически пытается найти строки сделок в DOM.

    Структура по скрину:
      <div class="table-responsive table-orders ...">
        <table class="table table-row-dashed ...">
          <tbody>
            <tr class="table-orders-row ...">
              <td>Цена</td>
              <td>Объём</td>
              <td>Время</td>
    """
    selectors = [
        "div.table-responsive.table-orders table tbody tr.table-orders-row",
        "div.table-responsive.table-orders table tbody tr",
        "table.table-row-dashed tbody tr.table-orders-row",
        "table.table-row-dashed tbody tr",
        "tr.table-orders-row",
    ]

    attempts = max_wait_seconds
    for i in range(attempts):
        for selector in selectors:
            rows = await page.query_selector_all(selector)
            if rows:
                logging.info(
                    "Found %d rows using selector '%s' on attempt %d/%d.",
                    len(rows),
                    selector,
                    i + 1,
                    attempts,
                )
                return rows

        logging.info(
            "No trade rows found yet (attempt %d/%d), waiting 1 second...",
            i + 1,
            attempts,
        )
        await page.wait_for_timeout(1_000)

    logging.warning("No history table rows found by any of the candidate selectors.")
    return []


def _normalize_num(text: str) -> float:
    """
    Преобразует строку вида '191 889.47' или '78,12' в float.
    Если не получилось — выбрасывает ValueError.
    """
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


async def parse_trades_from_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []

    for row in rows:
        try:
            cells = await row.query_selector_all("td")
            if len(cells) < 3:
                continue

            price_text = (await cells[0].inner_text()).strip()
            volume_text = (await cells[1].inner_text()).strip()
            time_text = (await cells[2].inner_text()).strip()

            price = _normalize_num(price_text)
            volume = _normalize_num(volume_text)

            trades.append(
                {
                    "price": price,
                    "volume": volume,
                    "time": time_text,
                    "price_raw": price_text,
                    "volume_raw": volume_text,
                }
            )
        except Exception as e:
            logging.info("Failed to parse a trade row: %s", e)
            continue

    return trades


async def scrape_rapira_trades() -> Dict[str, Any]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
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
            logging.info("Opening Rapira page %s ...", RAPIRA_URL)
            await page.goto(RAPIRA_URL, wait_until="networkidle", timeout=60_000)

            # Подождём, пока фронт подтянет данные
            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)
            await ensure_history_tab(page)

            # Ещё пауза после переключения вкладки
            await page.wait_for_timeout(3_000)

            logging.info("Trying to detect history/last-trades table in DOM ...")
            rows = await poll_for_trade_rows(page, max_wait_seconds=40)

            if not rows:
                logging.warning("No trade rows found on Rapira page.")
                return {
                    "exchange": "rapira",
                    "symbol": "USDT/RUB",
                    "count": 0,
                    "trades": [],
                }

            trades = await parse_trades_from_rows(rows)

            logging.info("Parsed %d trades.", len(trades))

            return {
                "exchange": "rapira",
                "symbol": "USDT/RUB",
                "count": len(trades),
                "trades": trades,
            }
        finally:
            await close_silently(page)
            await context.close()
            await browser.close()


async def main() -> None:
    logging.info("Starting Rapira last-trades scraper ...")
    result = await scrape_rapira_trades()
    logging.info("Scraper finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
