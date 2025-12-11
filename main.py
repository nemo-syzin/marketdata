import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List

from playwright.async_api import async_playwright, Page, Frame

# ───────────────────────── КОНСТАНТЫ ─────────────────────────

RAPIRA_URL = "https://rapira.net/trading/usdt-rub"
GRINEX_URL = "https://grinex.io/trading/usdta7a5"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


# ───────────────────────── ОБЩЕЕ: УСТАНОВКА BROWSERS ─────────────────────────


def ensure_playwright_browsers() -> None:
    """
    Гарантируем, что нужные браузеры для Playwright скачаны.
    Только download, без su / --with-deps.
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


# ───────────────────────── ОБЩИЕ ВСПОМОГАТЕЛЬНЫЕ ─────────────────────────


async def close_silently(page: Page) -> None:
    try:
        await page.close()
    except Exception:
        pass


async def accept_cookies_if_any(page: Page) -> None:
    """
    Нажимаем кнопку cookies, если она есть.
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


def _normalize_num(text: str) -> float:
    """
    Преобразует строку с пробелами и запятыми в float:
    '191 889.47' / '191 889,47' / '2 573.54' -> float.
    """
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


# ───────────────────────── RAPIRA ─────────────────────────


async def ensure_last_trades_tab_rapira(page: Page) -> None:
    """
    Включаем вкладку 'Последние сделки' / 'История сделок' для Rapira.
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
                logging.info("Clicking Rapira tab '%s' ...", text)
                await tab.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception as e:
            logging.info("Failed to click Rapira tab '%s': %s", text, e)

    logging.info(
        "Rapira last-trades tab not found explicitly, возможно, уже активна."
    )


async def poll_for_trade_rows_rapira(
    page: Page, max_wait_seconds: int = 40
) -> List[Any]:
    """
    Ожидаем появления строк сделок в DOM Rapira.
    """
    selectors = [
        "div.table-responsive.table-orders table.table-row-dashed tbody tr.table-orders-row",
        "div.table-responsive.table-orders table.table-row-dashed tbody tr",
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
                    "Found %d Rapira rows using selector '%s' on attempt %d/%d.",
                    len(rows),
                    selector,
                    i + 1,
                    max_wait_seconds,
                )
                return rows

        logging.info(
            "No Rapira trade rows yet (attempt %d/%d), waiting 1 second...",
            i + 1,
            max_wait_seconds,
        )
        await page.wait_for_timeout(1_000)

    logging.warning("No Rapira history table rows found by any selector.")
    return []


async def parse_trades_from_rows_rapira(rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Разбор строк Rapira: Цена / Объём / Время.
    """
    trades: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            cells = await row.query_selector_all("th, td")
            if len(cells) < 3:
                logging.info(
                    "Rapira row %d skipped: has only %d cells (need >= 3).",
                    idx,
                    len(cells),
                )
                continue

            price_text = (await cells[0].inner_text()).strip()
            volume_text = (await cells[1].inner_text()).strip()
            time_text = (await cells[2].inner_text()).strip()

            if not price_text or not volume_text or not time_text:
                logging.info(
                    "Rapira row %d skipped: empty cell(s): price='%s', volume='%s', time='%s'",
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
                    "Rapira row %d skipped: cannot convert to float "
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
            logging.info("Failed to parse Rapira row %d: %s", idx, e)
            continue

    return trades


async def scrape_rapira_trades() -> Dict[str, Any]:
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
            logging.info("Opening Rapira page %s ...", RAPIRA_URL)
            await page.goto(RAPIRA_URL, wait_until="networkidle", timeout=60_000)

            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)
            await ensure_last_trades_tab_rapira(page)

            await page.wait_for_timeout(3_000)

            logging.info("Trying to detect Rapira last-trades table in DOM ...")
            rows = await poll_for_trade_rows_rapira(page, max_wait_seconds=40)

            if not rows:
                logging.warning("No Rapira trade rows found.")
                return {
                    "exchange": "rapira",
                    "symbol": "USDT/RUB",
                    "count": 0,
                    "trades": [],
                }

            trades = await parse_trades_from_rows_rapira(rows)
            logging.info("Parsed %d Rapira trades.", len(trades))

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


# ───────────────────────── GRINEX ─────────────────────────

async def ensure_last_trades_tab_grinex(page: Page) -> None:
    """
    Жмём таб «Последние сделки» в центральном блоке под графиком.
    """
    # 1) Нормальный путь: ролевая вкладка
    try:
        tab = page.get_by_role("tab", name="Последние сделки")
        if await tab.count() > 0:
            logging.info("Clicking Grinex tab 'Последние сделки' (role=tab)...")
            await tab.first.click(timeout=5_000)
            await page.wait_for_timeout(1_000)
            return
    except Exception as e:
        logging.info("Failed to click Grinex tab by role: %s", e)

    # 2) Фолбэк — по тексту/классам
    selectors = [
        "div.cContainerTrading .nav-tabs a:has-text('Последние сделки')",
        "a:has-text('Последние сделки')",
        "button:has-text('Последние сделки')",
    ]
    for sel in selectors:
        try:
            tab2 = page.locator(sel)
            if await tab2.count() > 0:
                logging.info("Clicking Grinex tab via selector '%s'...", sel)
                await tab2.first.click(timeout=5_000)
                await page.wait_for_timeout(1_000)
                return
        except Exception as e:
            logging.info("Failed to click Grinex tab '%s': %s", sel, e)

    logging.info(
        "Grinex tab 'Последние сделки' not found explicitly, возможно, уже активна."
    )


async def _query_rows_in_all_frames(
    page: Page, selector: str
) -> List[Any]:
    """
    Ищем элементы по селектору во ВСЕХ фреймах страницы (main frame + iframes).
    """
    targets: List[Frame] = [page.main_frame] + page.frames
    # page.main_frame уже в page.frames, но от этого хуже не будет

    for frame in targets:
        try:
            rows = await frame.query_selector_all(selector)
            if rows:
                url = getattr(frame, "url", "unknown")
                logging.info(
                    "Found %d Grinex rows in frame '%s' using selector '%s'.",
                    len(rows),
                    url,
                    selector,
                )
                return rows
        except Exception as e:
            logging.info(
                "Error while querying selector '%s' in some frame: %s",
                selector,
                e,
            )
    return []


async def poll_for_trade_rows_grinex(
    page: Page, max_wait_seconds: int = 40
) -> List[Any]:
    """
    Ожидаем появления строк в таблице 'Последние сделки' на Grinex.

    По твоему скрину структура примерно такая:
      <div class="trade_history_panel">
        <table class="table all-trades usdta7a5 table-updated size-4">
          <tbody>
            <tr id="market-trade-..."> ... </tr>
    """

    selectors = [
        "div.trade_history_panel table.all-trades tbody tr",
        "table.all-trades.usdta7a5 tbody tr",
        "table.all-trades tbody tr",
        "tr[id^='market-trade-']",
    ]

    for i in range(max_wait_seconds):
        for selector in selectors:
            rows = await _query_rows_in_all_frames(page, selector)
            if rows:
                return rows

        logging.info(
            "No Grinex trade rows yet (attempt %d/%d), waiting 1 second...",
            i + 1,
            max_wait_seconds,
        )
        await page.wait_for_timeout(1_000)

    logging.warning(
        "No Grinex trade rows found by any selector for table 'Последние сделки'."
    )
    return []


async def parse_trades_from_rows_grinex(rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Разбор строк Grinex:
      Цена / Объём (USDT) / Объём (A7A5) / Дата и время.
    """
    trades: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            cells = await row.query_selector_all("td")
            if len(cells) < 4:
                logging.info(
                    "Grinex row %d skipped: has only %d cells (need >= 4).",
                    idx,
                    len(cells),
                )
                continue

            price_text = (await cells[0].inner_text()).strip()
            vol_usdt_text = (await cells[1].inner_text()).strip()
            vol_a7a5_text = (await cells[2].inner_text()).strip()
            datetime_text = (await cells[3].inner_text()).strip()

            if not price_text or not vol_usdt_text or not vol_a7a5_text or not datetime_text:
                logging.info(
                    "Grinex row %d skipped: empty cell(s): price='%s', usdt='%s', a7a5='%s', dt='%s'",
                    idx,
                    price_text,
                    vol_usdt_text,
                    vol_a7a5_text,
                    datetime_text,
                )
                continue

            try:
                price = _normalize_num(price_text)
                vol_usdt = _normalize_num(vol_usdt_text)
                vol_a7a5 = _normalize_num(vol_a7a5_text)
            except Exception as conv_err:
                logging.info(
                    "Grinex row %d skipped: cannot convert to float "
                    "(price='%s', usdt='%s', a7a5='%s'): %s",
                    idx,
                    price_text,
                    vol_usdt_text,
                    vol_a7a5_text,
                    conv_err,
                )
                continue

            trades.append(
                {
                    "price": price,
                    "volume_usdt": vol_usdt,
                    "volume_a7a5": vol_a7a5,
                    "datetime": datetime_text,
                    "price_raw": price_text,
                    "volume_usdt_raw": vol_usdt_text,
                    "volume_a7a5_raw": vol_a7a5_text,
                }
            )
        except Exception as e:
            logging.info("Failed to parse Grinex row %d: %s", idx, e)
            continue

    return trades


async def scrape_grinex_trades() -> Dict[str, Any]:
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

            # Чуть времени на фронт
            await page.wait_for_timeout(5_000)

            await accept_cookies_if_any(page)
            await ensure_last_trades_tab_grinex(page)

            # Небольшая пауза после переключения таба
            await page.wait_for_timeout(2_000)

            logging.info("Trying to detect Grinex 'Последние сделки' table in DOM ...")
            rows = await poll_for_trade_rows_grinex(page, max_wait_seconds=40)

            if not rows:
                logging.warning("No Grinex trade rows found.")
                return {
                    "exchange": "grinex",
                    "symbol": "USDT/RUB (usdta7a5)",
                    "count": 0,
                    "trades": [],
                }

            trades = await parse_trades_from_rows_grinex(rows)
            logging.info("Parsed %d Grinex trades.", len(trades))

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
    logging.info("Starting combined scrapers (Rapira + Grinex) ...")

    rapira, grinex = await asyncio.gather(
        scrape_rapira_trades(),
        scrape_grinex_trades(),
    )

    result = {
        "rapira": rapira,
        "grinex": grinex,
    }

    logging.info("All scrapers finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
