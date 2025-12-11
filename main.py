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


# ─────────────────── УСТАНОВКА BROWSERS ДЛЯ PLAYWRIGHT ───────────────────


def ensure_playwright_browsers() -> None:
    """
    Гарантируем, что Chromium для Playwright скачан.
    Важно: без su и без '--with-deps', только download.
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


def _normalize_num(text: str) -> float:
    """
    '191 889.47' / '191 889,47' / '2 573.54' -> float
    """
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    return float(t)


async def scroll_to_history_block(page: Page) -> None:
    """
    Скроллим страницу так, чтобы блок истории сделок оказался в зоне видимости.
    Grinex рендерит таблицу в:
      <div id="tab_trade_history_all"> ... <table class="table all-trades ...">
    """
    try:
        locator = page.locator("#tab_trade_history_all")
        if await locator.count() > 0:
            await locator.first.scroll_into_view_if_needed(timeout=5_000)
        else:
            panel = page.locator("div.trade_history_panel")
            if await panel.count() > 0:
                await panel.first.scroll_into_view_if_needed(timeout=5_000)
            else:
                # Если совсем ничего не нашли — грубая прокрутка вниз
                await page.mouse.wheel(0, 3000)

        await page.wait_for_timeout(1_000)
    except Exception as e:
        logging.info("Failed to scroll to history block explicitly: %s", e)
        try:
            await page.mouse.wheel(0, 3000)
            await page.wait_for_timeout(1_000)
        except Exception:
            pass


async def ensure_grinex_last_trades_tab(page: Page) -> None:
    """
    На всякий случай пытаемся кликнуть таб «Последние сделки».
    Если он не найдётся — просто продолжаем.
    """
    try:
        tab = page.locator("text=Последние сделки")
        if await tab.count() > 0:
            logging.info("Clicking Grinex tab 'Последние сделки' ...")
            await tab.first.click(timeout=5_000)
            await page.wait_for_timeout(1_000)
        else:
            logging.info(
                "Grinex tab 'Последние сделки' not found explicitly, возможно, уже активна."
            )
    except Exception as e:
        logging.info("Failed to click Grinex tab 'Последние сделки': %s", e)


async def poll_grinex_trade_rows(
    page: Page, max_wait_seconds: int = 40
) -> List[Any]:
    """
    Ожидаем появления строк истории сделок на Grinex.

    DOM по твоему скрину:
      <div id="tab_trade_history_all">
        <div class="trade_history_all_wrapper default" data-market="usdta7a5_tab">
          <div class="trade_history_panel">
            <table class="table all-trades usdta7a5 table-updated size-4">
              <tbody>
                <tr id="market-trade-...">
    """
    selector = (
        "#tab_trade_history_all table.all-trades tbody tr, "
        "div.trade_history_panel table.all-trades tbody tr, "
        "tr[id^='market-trade-']"
    )

    for attempt in range(1, max_wait_seconds + 1):
        rows = await page.query_selector_all(selector)
        if rows:
            logging.info(
                "Found %d Grinex trade rows on attempt %d/%d.",
                len(rows),
                attempt,
                max_wait_seconds,
            )
            return rows

        logging.info(
            "No trade rows found yet (attempt %d/%d), waiting 1 second...",
            attempt,
            max_wait_seconds,
        )
        await scroll_to_history_block(page)
        await page.wait_for_timeout(1_000)

    logging.warning("No trade rows found by any of the candidate selectors.")
    return []


async def parse_grinex_trades(rows: List[Any]) -> List[Dict[str, Any]]:
    """
    Разбор строк Grinex:
      <tr id="market-trade-...">
        <td class="price ..."> ... </td>
        <td class="volume ...">объём USDT</td>
        <td class="volume ...">объём A7A5</td>
        <td class="time ...">дата и время</td>
    """
    trades: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        try:
            trade_id = await row.get_attribute("id")

            cells = await row.query_selector_all("td")
            if len(cells) < 4:
                logging.info(
                    "Row %d skipped: has only %d cells (need >= 4).",
                    idx,
                    len(cells),
                )
                continue

            # Цена
            price_span = await cells[0].query_selector("span.visible-lg-inline")
            if price_span is None:
                price_text = (await cells[0].inner_text()).strip()
            else:
                price_text = (await price_span.inner_text()).strip()

            # Объём (USDT)
            vol_usdt_span = await cells[1].query_selector("span.visible-lg-inline")
            if vol_usdt_span is None:
                vol_usdt_text = (await cells[1].inner_text()).strip()
            else:
                vol_usdt_text = (await vol_usdt_span.inner_text()).strip()

            # Объём (A7A5)
            vol_a7_span = await cells[2].query_selector("span.visible-lg-inline")
            if vol_a7_span is None:
                vol_a7_text = (await cells[2].inner_text()).strip()
            else:
                vol_a7_text = (await vol_a7_span.inner_text()).strip()

            # Дата и время — берём последнюю непустую строку
            time_raw = (await cells[3].inner_text()).strip()
            time_lines = [t.strip() for t in time_raw.splitlines() if t.strip()]
            time_text = time_lines[-1] if time_lines else time_raw

            if not price_text or not vol_usdt_text or not vol_a7_text or not time_text:
                logging.info(
                    "Row %d skipped: empty cell(s): price='%s', vol_usdt='%s', "
                    "vol_a7a5='%s', time='%s'",
                    idx,
                    price_text,
                    vol_usdt_text,
                    vol_a7_text,
                    time_text,
                )
                continue

            try:
                price = _normalize_num(price_text)
                volume_usdt = _normalize_num(vol_usdt_text)
                volume_a7a5 = _normalize_num(vol_a7_text)
            except Exception as conv_err:
                logging.info(
                    "Row %d skipped: cannot convert to float "
                    "(price='%s', vol_usdt='%s', vol_a7a5='%s'): %s",
                    idx,
                    price_text,
                    vol_usdt_text,
                    vol_a7_text,
                    conv_err,
                )
                continue

            trades.append(
                {
                    "trade_id": trade_id,
                    "price": price,
                    "volume_usdt": volume_usdt,
                    "volume_a7a5": volume_a7a5,
                    "time": time_text,
                    "price_raw": price_text,
                    "volume_usdt_raw": vol_usdt_text,
                    "volume_a7a5_raw": vol_a7_text,
                    "time_raw": time_raw,
                }
            )
        except Exception as e:
            logging.info("Failed to parse Grinex trade row %d: %s", idx, e)
            continue

    return trades


# ───────────────────────── ОСНОВНОЙ СКРАПЕР ─────────────────────────


async def scrape_grinex_trades() -> Dict[str, Any]:
    """
    Основная функция:
    - гарантирует наличие браузера;
    - открывает страницу Grinex;
    - скроллит к истории;
    - парсит сделки.
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
            await page.goto(GRINEX_URL, wait_until="load", timeout=60_000)

            # даём фронту время прогрузиться
            await page.wait_for_timeout(5_000)

            await ensure_grinex_last_trades_tab(page)
            await scroll_to_history_block(page)

            logging.info("Trying to detect Grinex 'Последние сделки' table in DOM ...")
            rows = await poll_grinex_trade_rows(page, max_wait_seconds=40)

            if not rows:
                logging.warning("No trade rows found on Grinex page.")
                return {
                    "exchange": "grinex",
                    "symbol": "USDT/RUB (usdta7a5)",
                    "count": 0,
                    "trades": [],
                }

            trades = await parse_grinex_trades(rows)
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


# ───────────────────────────── entrypoint ─────────────────────────────


async def main() -> None:
    logging.info("Starting Grinex last-trades scraper ...")
    result = await scrape_grinex_trades()
    logging.info("Scraper finished.")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
