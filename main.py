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
    Нажимаем 'Я согласен' или аналогичный текст на баннере cookies, если он есть.
    """
    texts = ["Я согласен", "Согласен", "Принять", "Accept"]
    try:
        for t in texts:
            btn = page.locator(f"text={t}")
            if await btn.count() > 0:
                logging.info("Found cookies banner, clicking '%s'...", t)
                await btn.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        logging.info("No cookies button found.")
    except Exception as e:
        logging.info("Ignoring cookies click error: %s", e)


async def ensure_history_tab(page: Page) -> None:
    """
    Активируем вкладку с историей сделок (trade_history_panel).

    По DOM она живет в блоке:
      <div class="tab-pane ... history-list" id="tab_trade_history_all">
        <div class="trade_history_all_wrapper default" ...>
          <div class="trade_history_panel">...</div>
    """

    # 1. Пробуем по href/id таба
    selectors = [
        "a[href='#tab_trade_history_all']",
        "a[data-target='#tab_trade_history_all']",
        "li a[href='#tab_trade_history_all']",
    ]
    for sel in selectors:
        try:
            tab = page.locator(sel)
            if await tab.count() > 0:
                logging.info("Clicking history tab via selector '%s'...", sel)
                await tab.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception as e:
            logging.info("Failed to click history tab '%s': %s", sel, e)

    # 2. Фолбэк по тексту
    text_candidates = ["История", "История сделок", "Сделки", "History"]
    for text in text_candidates:
        try:
            tab = page.locator(f"text={text}")
            if await tab.count() > 0:
                logging.info("Clicking history tab with text '%s'...", text)
                await tab.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception as e:
            logging.info("Failed to click history tab text '%s': %s", text, e)

    logging.info(
        "History tab ('tab_trade_history_all') not found explicitly, "
        "возможно, уже активна."
    )


async def poll_for_trade_rows(page: Page, max_wait_seconds: int = 40) -> List[Any]:
    """
    Ожидаем появления строк сделок в trade_history_panel.

    По скрину структура такая:
      <div class="trade_history_panel">
        <table class="table table-updated size-4">  <-- заголовок
        <table class="table all-trades usdta7a5 table-updated size-4">
          <tbody>
            <tr id="market-trade-...">
              <td class="price text-left col-xs-6 text-up">...</td>
              <td class="volume text-left col-xs-6" ...>...</td>
              <td class="time text-left col-xs-6">...</td>
    """

    selectors = [
        "div.trade_history_panel table.all-trades tbody tr",
        "div.trade_history_panel table.table.all-trades tbody tr",
        "table.all-trades tbody tr",
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

    logging.warning("No trade rows found by any of the candidate selectors.")
    return []


def _normalize_num(text: str) -> float:
    """
    Преобразует '191 889.47' / '191 889,47' / '2 573.54' -> float.
    """
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


async def parse_trades_from_rows(rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Разбираем строки таблицы сделок Grinex.
    Берем 3 ячейки: Цена, Объём, Дата и время.
    """
    trades: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            # на всякий случай берём и th, и td
            cells = await row.query_selector_all("th, td")
            if len(cells) < 3:
                logging.info(
                    "Row %d skipped: has only %d cells (need >= 3).",
                    idx,
                    len(cells),
                )
                continue

            price_text = (await cells[0].inner_text()).strip()
            volume_text = (await cells[1].inner_text()).strip()
            time_raw = (await cells[2].inner_text()).strip()

            # time_raw у Grinex выглядит примерно так:
            # '14:02:48\n\n10.12.2025 14:02 "'
            # берём только первое "слово" — собственно время сделки
            time_parts = time_raw.split()
            time_text = time_parts[0] if time_parts else time_raw

            if not price_text or not volume_text or not time_text:
                logging.info(
                    "Row %d skipped: empty cell(s): price='%s', volume='%s', time='%s'",
                    idx,
                    price_text,
                    volume_text,
                    time_raw,
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
            logging.info("Failed to parse trade row %d: %s", idx, e)
            continue

    return trades


# ───────────────────────── ОСНОВНОЙ СКРАПЕР ─────────────────────────


async def scrape_grinex_trades() -> Dict[str, Any]:
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
            logging.info("Opening Grinex page %s ...", GRINEX_URL)
            await page.goto(GRINEX_URL, wait_until="networkidle", timeout=60_000)

            # Даём фронту немного времени подтянуть данные
            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)
            await ensure_history_tab(page)

            # Ещё пауза после переключения таба
            await page.wait_for_timeout(3_000)

            logging.info("Trying to detect Grinex history table in DOM ...")
            rows = await poll_for_trade_rows(page, max_wait_seconds=40)

            if not rows:
                logging.warning("No trade rows found on Grinex page.")
                return {
                    "exchange": "grinex",
                    "symbol": "USDT/RUB (usdta7a5)",
                    "count": 0,
                    "trades": [],
                }

            trades = await parse_trades_from_rows(rows)
            logging.info("Parsed %d trades.", len(trades))

            return {
                "exchange": "grinex",
                "symbol": "USDT/RUB (usdta7a5)",
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
    logging.info("Scraper finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
