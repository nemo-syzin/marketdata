import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List

from playwright.async_api import async_playwright, Page

RAPIRA_URL = "https://rapira.net/trading/usdt-rub"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# ───────────────────────── УСТАНОВКА BROWSERS ─────────────────────────


def ensure_playwright_browsers() -> None:
    """
    Гарантируем, что нужные браузеры для Playwright скачаны.

    ВАЖНО:
    - только download (без системных deps)
    - никаких su / --with-deps
    """
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
            "Убедись, что Playwright установлен (npm/pip) и доступен как 'playwright'."
        )
    except Exception as e:
        logging.error("Unexpected error while installing Playwright browsers: %s", e)


# ───────────────────────── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ─────────────────────────


async def close_silently(page: Page) -> None:
    try:
        await page.close()
    except Exception:
        pass


async def accept_cookies_if_any(page: Page) -> None:
    """
    Нажимаем 'Я согласен' на баннере cookies, если он есть.
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


async def ensure_last_trades_tab(page: Page) -> None:
    """
    Включаем вкладку 'Последние сделки' в центральном блоке.

    На всякий случай пробуем несколько вариантов текста.
    """
    candidates = [
        "Последние сделки",
        "История сделок",
        "История",
    ]

    for text in candidates:
        try:
            tab = page.locator(f"text={text}")
            if await tab.count() > 0:
                logging.info("Clicking tab '%s' ...", text)
                await tab.first.click(timeout=5_000)
                # Небольшая пауза после переключения
                await page.wait_for_timeout(1_000)
                return
        except Exception as e:
            logging.info("Failed to click tab '%s': %s", text, e)

    logging.info(
        "Last-trades tab ('Последние сделки' / 'История сделок') "
        "not found explicitly, возможно, уже активна."
    )


async def poll_for_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    """
    Ожидаем появления строк сделок в DOM.

    По твоему скрину структура такая:
      <div class="table-responsive table-orders ...">
        <table>
          <tbody>
            <tr class="table-orders-row ...">
    """

    selectors = [
        "div.table-responsive.table-orders table tbody tr.table-orders-row",
        "div.table-responsive.table-orders table tbody tr",
        "table.table-row-dashed tbody tr.table-orders-row",
        "table.table-row-dashed tbody tr",
        "tr.table-orders-row",
    ]

    for i in range(max_wait_seconds):
        for selector in selectors:
            rows = await page.query_selector_all(selector)
            if rows:
                logging.info(
                    "Found %d rows using selector '%s' on attempt %d/%d.",
                    len(rows),
                    selector,
                    i + 1,
                    max_wait_seconds,
                )
                return rows

        logging.info(
            "No trade rows found yet (attempt %d/%d), waiting 1 second...",
            i + 1,
            max_wait_seconds,
        )
        await page.wait_for_timeout(1_000)

    logging.warning("No history table rows found by any of the candidate selectors.")
    return []


def _normalize_num(text: str) -> float:
    """
    Преобразует '191 889.47' / '191 889,47' / '78,12' -> float.
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


# ───────────────────────── ОСНОВНОЙ СКРАПЕР ─────────────────────────


async def scrape_rapira_trades() -> Dict[str, Any]:
    # 1. Сначала гарантируем, что браузеры скачаны
    ensure_playwright_browsers()

    # 2. Запускаем Playwright
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

            # Даём фронту немного времени подтянуть данные
            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)
            await ensure_last_trades_tab(page)

            # Ещё пауза после переключения
            await page.wait_for_timeout(3_000)

            logging.info("Trying to detect last-trades table in DOM ...")
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


# ───────────────────────── entrypoint ─────────────────────────


async def main() -> None:
    logging.info("Starting Rapira last-trades scraper ...")
    result = await scrape_rapira_trades()
    logging.info("Scraper finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
