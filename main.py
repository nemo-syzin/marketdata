import asyncio
import json
import logging
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright

# ───────────────────── ЛОГГЕР ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("grinex-last-trades")

# ───────────────────── КОНСТАНТЫ ────────────────────
GRINEX_URL = "https://grinex.io/trading/usdta7a5?lang=en"
TIMEOUT_MS = 30_000
MAX_TRADES = 20  # сколько последних сделок забирать максимум

# Прокси (для продакшена лучше вынести в переменные окружения)
PROXY_SERVER = "http://5.22.207.65:50100"
PROXY_USERNAME = "nemosyzin"
PROXY_PASSWORD = "TDbaTDHbTA"


# ───────────── УСТАНОВКА CHROMIUM ДЛЯ PLAYWRIGHT ────
def install_chromium_for_playwright() -> None:
    """
    Одноразовая установка Chromium для Playwright на Render.
    При повторных запусках быстро отдаёт "already present".
    """
    try:
        logger.info("Ensuring Playwright Chromium is installed ...")
        subprocess.run(["playwright", "install", "chromium"], check=True)
        logger.info("Playwright Chromium is installed (or already present).")
    except Exception as exc:
        logger.warning("Playwright install error: %s", exc)


# ──────────────────── ХЕЛПЕРЫ ───────────────────────
def parse_float(text: str) -> Optional[float]:
    """
    Парсинг числа вида '25 033.8898' или '79,30' → float.
    Если не получилось — возвращаем None.
    """
    if not text:
        return None
    cleaned = text.replace("\xa0", " ").strip()
    # оставляем только цифры, запятую, точку
    filtered = "".join(ch for ch in cleaned if ch.isdigit() or ch in ",.")
    if not filtered:
        return None
    # если и запятая и точка – предположим, что точка = десятичный разделитель
    if "," in filtered and "." in filtered:
        # убираем пробелы-разделители тысяч, запятую тоже можно убрать
        filtered = filtered.replace(" ", "").replace(",", "")
    else:
        # если только запятая – считаем её десятичной
        filtered = filtered.replace(",", ".")
    try:
        return float(filtered)
    except ValueError:
        return None


def is_captcha_page(html: str) -> bool:
    """
    Простейшая эвристика: определяем, что нас редиректнуло
    на страницу rotated-captcha, а не на реальную торговую.
    """
    markers = [
        "sp_rotated_captcha",
        "captcha-wrap",
        "captcha-title",
        "captcha-control-button",
    ]
    return any(m in html for m in markers)


# ─────────────── ПАРСИНГ ПОСЛЕДНИХ СДЕЛОК ───────────
async def fetch_grinex_last_trades() -> List[Dict[str, Any]]:
    """
    Пытаемся открыть https://grinex.io/trading/usdta7a5 и вытащить
    последние сделки из таблицы "Последние сделки".

    Если попадаем на страницу с капчей – логируем это и возвращаем [].
    """
    async with async_playwright() as p:
        logger.info("Launching Chromium with proxy ...")
        browser = await p.chromium.launch(
            headless=True,
            proxy={
                "server": PROXY_SERVER,
                "username": PROXY_USERNAME,
                "password": PROXY_PASSWORD,
            },
            args=["--disable-blink-features=AutomationControlled"],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        logger.info("Opening Grinex page %s ...", GRINEX_URL)
        await page.goto(GRINEX_URL, wait_until="domcontentloaded", timeout=TIMEOUT_MS)

        # Берём HTML, чтобы понять, что именно нам отдали
        html = await page.content()
        if is_captcha_page(html):
            logger.warning(
                "Looks like Grinex returned rotated-captcha page. "
                "Real trading interface is not accessible, cannot read last trades."
            )
            await browser.close()
            return []

        # Пытаемся найти таблицу последних сделок
        rows_locator = page.locator("table.all-trades.usdta7a5 tbody tr[id^='market-trade-']")

        rows_count = await rows_locator.count()
        logger.info("Detected %s trade rows in table selector.", rows_count)

        if rows_count == 0:
            # Дополнительная диагностика: проверим наличие классов и id
            has_tab = "#tab_trade_history_all" in html
            has_table = "all-trades usdta7a5" in html
            logger.info(
                "Debug flags: tab_trade_history_all=%s, table_class_present=%s",
                has_tab,
                has_table,
            )
            await browser.close()
            return []

        trades: List[Dict[str, Any]] = []

        limit = min(rows_count, MAX_TRADES)
        for i in range(limit):
            row = rows_locator.nth(i)

            price_td = row.locator("td.price")
            vol_usdt_td = row.locator("td.volume").nth(0)
            vol_a7a5_td = row.locator("td.volume").nth(1)
            time_td = row.locator("td.time")

            # Берём текст
            price_text = (await price_td.inner_text()).strip()
            vol_usdt_text = (await vol_usdt_td.inner_text()).strip()
            vol_a7a5_text = (await vol_a7a5_td.inner_text()).strip()
            time_text = (await time_td.inner_text()).strip()

            trade = {
                "row_index": i,
                "price_raw": price_text,
                "volume_usdt_raw": vol_usdt_text,
                "volume_a7a5_raw": vol_a7a5_text,
                "time_raw": time_text,
                "price": parse_float(price_text),
                "volume_usdt": parse_float(vol_usdt_text),
                "volume_a7a5": parse_float(vol_a7a5_text),
                # парсить дату/время можно отдельно, пока оставляем строкой
            }
            trades.append(trade)

        await browser.close()
        return trades


# ───────────────────── MAIN ─────────────────────────
async def main() -> None:
    install_chromium_for_playwright()

    logger.info("Starting Grinex last-trades one-shot scraper ...")
    start_ts = datetime.utcnow()

    trades = await fetch_grinex_last_trades()

    result = {
        "exchange": "grinex",
        "symbol": "USDT/RUB (usdta7a5)",
        "utc_timestamp": start_ts.isoformat() + "Z",
        "count": len(trades),
        "trades": trades,
    }

    # Логируем красивым JSON
    logger.info("Grinex last trades snapshot:\n%s", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
