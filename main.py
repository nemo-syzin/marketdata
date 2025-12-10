import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import List, Optional

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ============= НАСТРОЙКИ ============================================

# Страница Rapira с парой USDT/RUB (если URL другой — поменяй тут)
RAPIRA_TRADE_URL = "https://rapira.net/trading/usdt-rub"

# Как часто опрашивать таблицу "История", секунды
POLL_INTERVAL_SEC = 10

# Логгер
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("marketdata-rapira")

# Глобальный маркер последней обработанной сделки (в рамках запущенного процесса)
_last_trade_key: Optional[str] = None


# ============= УСТАНОВКА BROWSER PLAYWRIGHT ==========================

def install_chromium() -> None:
    """
    Качаем Chromium для Playwright.
    Вызывается один раз при старте (как в kenigswap-rates).
    """
    try:
        log.info("Installing Chromium for Playwright ...")
        # если будут ругаться на зависимости — можно заменить на "--with-deps", "chromium"
        subprocess.run(["playwright", "install", "chromium"], check=True)
        log.info("Chromium installed successfully.")
    except Exception as exc:
        # Игнорируем ошибку, чтобы не падать, но лог пишем
        log.warning("Playwright install error (ignored): %s", exc)


# ============= МОДЕЛЬ ДАННЫХ =======================================

@dataclass
class RapiraDomTrade:
    price: float
    volume: float
    time_str: str          # строка вида "15:17:49"
    ts: datetime           # время сделки (UTC, на основе сегодняшней даты)
    side: Optional[str] = None  # можно доработать по цвету/классу строки


# ============= HELPERS =============================================

def _parse_number_ru(s: str) -> Optional[float]:
    """
    '3 425.55' / '1 269,99' / '0.84' -> float
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
    Стабильный ключ для сделок (время + цена + объём).
    """
    return f"{t.time_str}|{t.price}|{t.volume}"


def _combine_time_today(time_str: str) -> datetime:
    """
    Берём сегодняшнюю дату + время HH:MM[:SS] из строки.
    Если не получилось распарсить — возвращаем текущий момент.
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


# ============= PLAYWRIGHT / DOM-ПАРСИНГ ============================

async def _prepare_page():
    """
    Запускаем браузер, открываем страницу Rapira и переключаемся на вкладку 'История'.
    Возвращаем (playwright, browser, page).
    """
    p = await async_playwright().start()
    browser = await p.chromium.launch(headless=True)
    page = await browser.new_page()

    log.info("Opening Rapira page %s ...", RAPIRA_TRADE_URL)
    await page.goto(RAPIRA_TRADE_URL, wait_until="networkidle")

    # Принять cookies (если есть попап)
    try:
        await page.get_by_text("Я согласен").click(timeout=5000)
    except PlaywrightTimeoutError:
        pass

    # Переключиться на вкладку "История"
    try:
        await page.get_by_text("История").click(timeout=7000)
    except PlaywrightTimeoutError:
        log.info("Tab 'История' might already be active")

    await page.wait_for_timeout(1000)
    return p, browser, page


async def _parse_trades_from_page(page) -> List[RapiraDomTrade]:
    """
    Парсим таблицу 'История' на уже открытой странице.
    Возвращаем список сделок (от старых к новым).
    """
    # Находим таблицу, где в заголовках есть 'Цена', 'Объем/Объём' и 'Время'
    table_locator = page.locator(
        "//table[.//th[contains(., 'Цен')]"  # 'Цена'
        "       and (.//th[contains(., 'Объем')] or .//th[contains(., 'Объём')])"
        "       and .//th[contains(., 'Время')]]"
    )

    await table_locator.wait_for(state="visible", timeout=15000)

    rows = table_locator.locator("tbody tr")
    row_count = await rows.count()
    trades: List[RapiraDomTrade] = []

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

        # При желании можно определить BUY/SELL по классу/цвету строки
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


async def fetch_rapira_new_trades(page) -> List[RapiraDomTrade]:
    """
    Возвращает только новые сделки по сравнению с прошлым вызовом
    (в рамках одного процесса), используя глобальный _last_trade_key.
    """
    global _last_trade_key

    trades = await _parse_trades_from_page(page)
    if not trades:
        return []

    keys = [_build_trade_key(t) for t in trades]
    new_trades: List[RapiraDomTrade] = []

    if _last_trade_key is None:
        # Первый запуск — считаем новыми все сделки
        new_trades = trades
    else:
        try:
            idx = keys.index(_last_trade_key)
            new_trades = trades[idx + 1 :]
        except ValueError:
            # Если старую сделку не нашли (таблица сбросилась) — считаем новыми все
            new_trades = trades

    _last_trade_key = _build_trade_key(trades[-1])
    return new_trades


# ============= MAIN-ЦИКЛ ==========================================

async def main():
    # 1. Ставим Chromium (как в kenigswap-rates)
    install_chromium()

    # 2. Запускаем Playwright и открываем страницу
    playwright = None
    browser = None
    try:
        playwright, browser, page = await _prepare_page()
        log.info("Rapira page is ready, starting polling loop ...")

        while True:
            try:
                new_trades = await fetch_rapira_new_trades(page)
            except PlaywrightTimeoutError:
                log.warning("Timeout while reading trades table, retrying ...")
                new_trades = []

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

                # здесь можно вместо логов писать в БД / Supabase и т.п.

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
