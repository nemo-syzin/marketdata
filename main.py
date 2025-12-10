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


# ───────────────────────── УСТАНОВКА BROWSERS ─────────────────────────


def ensure_playwright_browsers() -> None:
    """
    Гарантируем, что нужные браузеры для Playwright скачаны.

    Никаких su / --with-deps — только загрузка браузеров.
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
            "Убедись, что Playwright установлен и доступен как 'playwright'."
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
    (На Grinex текст обычно такой же, как на Rapira.)
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


def _normalize_num(text: str) -> float:
    """
    Преобразует '10 483.3136' / '10 483,3136' в float.
    Убираем пробелы / неразрывные пробелы и заменяем запятую на точку.
    """
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


async def poll_for_trade_rows_grinex(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    """
    Ожидаем появления строк истории сделок на Grinex.

    По скрину структура такая:
      <div id="tab_trade_history_all" ...>
        <table class="table all-trades usdt7a5 table-updated size-4">
          <tbody>
            <tr id="market-trade-...">
    """

    selector = "#tab_trade_history_all table.all-trades tbody tr"

    for i in range(max_wait_seconds):
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
            "No Grinex trade rows yet (attempt %d/%d), waiting 1 second...",
            i + 1,
            max_wait_seconds,
        )
        await page.wait_for_timeout(1_000)

    logging.warning("No Grinex trade rows found by selector '%s'.", selector)
    return []


async def parse_trades_from_rows_grinex(rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Парсим строки таблицы Grinex.

    Заголовок:
      Цена | Объём (USDT) | Объём (A7AS) | Дата и время

    Берём:
      price  = колонка 0 (значение из последнего <span> в ячейке)
      volume = колонка 1 (USDT)
      time   = колонка 3 (берём только 'HH:MM:SS')
    """
    trades: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            cells = await row.query_selector_all("td")
            if len(cells) < 4:
                logging.info(
                    "Row %d skipped: has only %d cells (need >= 4).",
                    idx,
                    len(cells),
                )
                continue

            # ── PRICE (ячейка 0: там валюта + цена в <span>ах) ──
            price_cell = cells[0]
            price_spans = await price_cell.query_selector_all("span")
            if price_spans:
                price_text = (await price_spans[-1].inner_text()).strip()
            else:
                price_text = (await price_cell.inner_text()).strip()

            # ── VOLUME (USDT, ячейка 1) ──
            volume_cell = cells[1]
            volume_text = (await volume_cell.inner_text()).strip()

            # ── TIME (ячейка 3: '14:02:48 10.12.2025 14:02' и т.п.) ──
            time_cell = cells[3]
            time_full = (await time_cell.inner_text()).strip()
            # Берём первую "часть", где обычно HH:MM:SS
            time_text = time_full.split()[0] if time_full else ""

            if not price_text or not volume_text or not time_text:
                logging.info(
                    "Row %d skipped: empty values: price='%s', volume='%s', time='%s'",
                    idx,
                    price_text,
                    volume_text,
                    time_text,
                )
                continue

            try:
                price = _normalize_num(price_text)
                volume = _normalize_num(volume_text)
            except Exception as conv_err:
                logging.info(
                    "Row %d skipped: cannot convert to float "
                    "(price='%s', volume='%s'): %s",
                    idx,
                    price_text,
                    volume_text,
                    conv_err,
                )
                continue

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
            logging.info("Failed to parse Grinex trade row %d: %s", idx, e)
            continue

    return trades


# ───────────────────────── ОСНОВНОЙ СКРАПЕР GRINEX ─────────────────────────


async def scrape_grinex_trades() -> Dict[str, Any]:
    """
    Скрейпер последних сделок Grinex для пары USDT/A7AS.

    Возвращает:
    {
      "exchange": "grinex",
      "symbol": "USDT/A7AS",
      "count": N,
      "trades": [
         {
            "price": float,
            "volume": float,
            "time": "HH:MM:SS",
            "price_raw": str,
            "volume_raw": str
         },
         ...
      ]
    }
    """
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
            logging.info("Opening Grinex page %s ...", GRINEX_URL)
            await page.goto(GRINEX_URL, wait_until="networkidle", timeout=60_000)

            # Даём фронту прогрузиться
            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)

            # История сделок по умолчанию обычно активна,
            # поэтому отдельно таб не трогаем. Если вдруг понадобится:
            # link = page.locator("a[href='#tab_trade_history_all']")
            # if await link.count() > 0: await link.first.click()

            logging.info("Trying to detect Grinex last-trades table in DOM ...")
            rows = await poll_for_trade_rows_grinex(page, max_wait_seconds=40)

            if not rows:
                logging.warning("No Grinex trade rows found.")
                return {
                    "exchange": "grinex",
                    "symbol": "USDT/A7AS",
                    "count": 0,
                    "trades": [],
                }

            trades = await parse_trades_from_rows_grinex(rows)
            logging.info("Parsed %d Grinex trades.", len(trades))

            return {
                "exchange": "grinex",
                "symbol": "USDT/A7AS",
                "count": len(trades),
                "trades": trades,
            }
        finally:
            await close_silently(page)
            await context.close()
            await browser.close()


# ───────────────────────── entrypoint ─────────────────────────


async def main() -> None:
    logging.info("Starting Grinex last-trades scraper ...")
    result = await scrape_grinex_trades()
    logging.info("Grinex scraper finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
