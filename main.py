import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List

from playwright.async_api import async_playwright, Page

GRINEX_URL = "https://grinex.io/trading/usdta7a5"

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
    except FileNotFoundError:
        logging.error(
            "playwright CLI not found in PATH. "
            "Убедись, что Playwright установлен и доступен как 'playwright'."
        )
    except Exception as e:
        logging.error("Unexpected error while installing Playwright browsers: %s", e)


async def close_silently(page: Page) -> None:
    try:
        await page.close()
    except Exception:
        pass


def _normalize_num(text: str) -> float:
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


async def dump_html_debug(page: Page) -> None:
    """
    Печатаем в логи первые ~4000 символов HTML + простую статистику по селекторам.
    """
    for attempt in range(1, 11):
        try:
            html = await page.content()
            short = html[:4000].replace("\n", " ").replace("\r", " ")
            logging.info(
                "HTML DUMP attempt %d: first 4000 chars: %s",
                attempt,
                short,
            )

            has_tab = "#tab_trade_history_all" in html
            has_all_trades = "all-trades usdta7a5" in html

            tables_all_trades = await page.query_selector_all("table.all-trades")
            rows_market = await page.query_selector_all("tr[id^='market-trade-']")

            logging.info(
                "HTML debug flags: tab_trade_history_all present=%s, "
                "'all-trades usdta7a5' present=%s, "
                "table.all-trades count=%d, "
                "tr[id^='market-trade-'] count=%d",
                has_tab,
                has_all_trades,
                len(tables_all_trades),
                len(rows_market),
            )
            return
        except Exception as e:
            logging.info(
                "Failed to get page.content() on attempt %d/10: %s",
                attempt,
                e,
            )
            await page.wait_for_timeout(1_000)


async def scrape_grinex_trades_debug() -> Dict[str, Any]:
    ensure_playwright_browsers()

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
            viewport={"width": 1440, "height": 900},
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
            logging.info("Opening Grinex page %s ...", GRINEX_URL)
            await page.goto(GRINEX_URL, wait_until="networkidle", timeout=60_000)

            # Небольшой буфер после networkidle
            await page.wait_for_timeout(3_000)

            # Отладочный дамп того, что реально видит Playwright
            await dump_html_debug(page)

            # Пока возвращаем пустой результат — цель этого скрипта именно отладка
            return {
                "exchange": "grinex",
                "symbol": "USDT/RUB (usdta7a5)",
                "count": 0,
                "trades": [],
            }
        finally:
            await close_silently(page)
            await context.close()
            await browser.close()


async def main() -> None:
    logging.info("Starting Grinex debug scraper ...")
    result = await scrape_grinex_trades_debug()
    logging.info("Scraper finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
