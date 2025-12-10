import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import List, Optional

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError, Page, Locator

# ============= НАСТРОЙКИ ============================================

# Страница Rapira с парой USDT/RUB (поменяй при необходимости)
RAPIRA_TRADE_URL = "https://rapira.net/trading/usdt-rub"

# Как часто опрашивать таблицу "История", секунды
POLL_INTERVAL_SEC = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("marketdata-rapira")

# Маркер последней обработанной сделки (в рамках процесса)
_last_trade_key: Optional[str] = None


# ============= УСТАНОВКА PLAYWRIGHT BROWSER ==========================

def install_chromium() -> None:
    """
    Скачивает Chromium для Playwright (как в kenigswap-rates).
    """
    try:
        log.info("Installing Chromium for Playwright ...")
        subprocess.run(["playwright", "install", "chromium"], check=True)
        log.info("Chromium installed successfully.")
    except Exception as exc:
        log.warning("Playwright install error (ignored): %s", exc)


# ============= МОДЕЛЬ ДАННЫХ =======================================

@dataclass
class RapiraDomTrade:
    price: float
    volume: float
    time_str: str          # строка, как в таблице, напр. "15:17:49"
    ts: datetime           # время сделки (UTC, по сегодняшней дате)
    side: Optional[str] = None


# ============= HELPERS =============================================

def _parse_number_ru(s: str) -> Optional[float]:
    """
    '3 425.55', '1 269,99', '0.84' -> float
    """
    if s is None:
        return None
    s = s.strip().replace(" ", "")
    s = s.replace(",", ".")
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


def _build_trade_key(t: RapiraDomTrade) -> str:
    """
    Стабильный ключ сделки для дедупликации.
    """
    return f"{t.time_str}|{t.price}|{t.volume}"


def _combine_time_today(time_str: str) -> datetime:
    """
    Берём сегодняшнюю дату + время HH:MM[:SS] из строки.
    """
    time_str = time_str.strip()
    try:
        parts = [int(x) for x in time_str.split(":")]
        if len(parts) == 2:
            hh, mm = parts
            ss = 0
        else:
            hh, mm, ss = parts
        today = datetime.now(timezone.utc).date()
        return datetime.combine(today, dtime(hh, mm, ss), tzinfo=timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


# ============= ПОИСК ТАБЛИЦЫ "ИСТОРИЯ" ==============================

async def _find_history_table(page: Page) -> Locator:
    """
    Ищет на странице таблицу, в заголовке которой есть колонки
    'Цена', 'Объем/Объём', 'Время' (или 'Price', 'Volume', 'Time').

    Возвращает Locator на эту таблицу или бросает TimeoutError.
    """
    # даём странице чуть времени дорендериться
    await page.wait_for_timeout(2000)

    tables = page.locator("table")
    count = await tables.count()
    log.info("Found %d <table> elements on page", count)

    for i in range(count):
        t = tables.nth(i)
        # пытаемся считать заголовки thead
        headers = await t.locator("thead th").all_inner_texts()
        headers = [h.strip() for h in headers if h.strip()]
        if len(headers) < 3:
            continue

        headers_joined = " | ".join(headers)
        log.debug("Table %d headers: %s", i, headers_joined)

        has_price = any("цен" in h.lower() or "price" in h.lower() for h in headers)
        has_vol = any(
            "объем" in h.lower()
            or "объём" in h.lower()
            or "volume" in h.lower()
            for h in headers
        )
        has_time = any("врем" in h.lower() or "time" in h.lower() for h in headers)

        if has_price and has_vol and has_time:
            log.info("History table detected at index %d", i)
            return t

    # если ни одна таблица не подошла
    raise PlaywrightTimeoutError("History trades table not found by headers")


# ============= PLAYWRIGHT / DOM-ПАРСИНГ ============================

async def _prepare_page():
    """
    Запускаем браузер, открываем страницу и включаем вкладку 'История'.
    """
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = await browser.new_page(
        viewport={"width": 1280, "height": 720},
    )

    log.info("Opening Rapira page %s ...", RAPIRA_TRADE_URL)
    await page.goto(RAPIRA_TRADE_URL, wait_until="networkidle")

    # принять cookies
    try:
        await page.get_by_text("Я согласен").click(timeout=5000)
    except PlaywrightTimeoutError:
        pass

    # перейти на вкладку 'История'
    try:
        await page.get_by_text("История").click(timeout=10000)
    except PlaywrightTimeoutError:
        log.info("Tab 'История' might already be active")

    await page.wait_for_timeout(1500)
    return p, browser, page


async def _parse_trades_from_page(page: Page) -> List[RapiraDomTrade]:
    """
    Находит таблицу 'История' и парсит из неё сделки.
    """
    table = await _find_history_table(page)

    rows = table.locator("tbody tr")
    row_count = await rows.count()
    trades: List[RapiraDomTrade] = []

    log.info("History table rows: %d", row_count)

    for i in range(row_count):
        row = rows.nth(i)
        cells = row.locator("td")
        cell_count = await cells.count()
        if cell_count < 3:
            continue

        price_text = (await cells.nth(0).inner_text()).strip()
        vol_text = (await cells.nth(1).inner_text()).strip()
        time_text = (await cells.nth(2).inner_text()).strip()

        price = _parse_number_ru(price_text)
        volume = _parse_number_ru(vol_text)
        if price is None or volume is None:
            continue

        ts = _combine_time_today(time_text)

        trade = RapiraDomTrade(
            price=price,
            volume=volume,
            time_str=time_text,
            ts=ts,
            side=None,
        )
        trades.append(trade)

    trades.sort(key=lambda t: t.ts)
    return trades


async def fetch_rapira_new_trades(page: Page) -> List[RapiraDomTrade]:
    """
    Возвращает только новые сделки по сравнению с предыдущим вызовом.
    """
    global _last_trade_key

    try:
        trades = await _parse_trades_from_page(page)
    except PlaywrightTimeoutError as e:
        log.warning("History table not found: %s", e)
        return []

    if not trades:
        return []

    keys = [_build_trade_key(t) for t in trades]
    new_trades: List[RapiraDomTrade] = []

    if _last_trade_key is None:
        new_trades = trades
    else:
        try:
            idx = keys.index(_last_trade_key)
            new_trades = trades[idx + 1 :]
        except ValueError:
            # если не нашли прошлую сделку — считаем новыми все
            new_trades = trades

    _last_trade_key = _build_trade_key(trades[-1])
    return new_trades


# ============= MAIN-ЦИКЛ ==========================================

async def main():
    install_chromium()  # как в рабочем проекте

    playwright = None
    browser = None
    try:
        playwright, browser, page = await _prepare_page()
        log.info("Rapira page is ready, starting polling loop ...")

        while True:
            new_trades = await fetch_rapira_new_trades(page)

            if new_trades:
                total_turnover = sum(t.price * t.volume for t in new_trades)
                log.info(
                    "New trades: %d, total turnover: %.2f RUB",
                    len(new_trades),
                    total_turnover,
                )
                for t in new_trades:
                    log.info(
                        "Trade %s | price=%.2f | volume=%.4f",
                        t.time_str,
                        t.price,
                        t.volume,
                    )
                # здесь можно писать в БД / Supabase и т.п.

            await asyncio.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        log.info("Stopped by KeyboardInterrupt")
    finally:
        if browser is not None:
            await browser.close()
        if playwright is not None:
            await playwright.stop()


if __name__ == "__main__":
    asyncio.run(main())
