import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import List, Optional

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

# ============= НАСТРОЙКИ ============================================

RAPIRA_TRADE_URL = "https://rapira.net/trading/usdt-rub"  # URL пары USDT/RUB
POLL_INTERVAL_SEC = 10  # как часто опрашивать таблицу, сек

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rapira-trades")

# Глобальный маркер последней обработанной сделки
_last_trade_key: Optional[str] = None


# ============= МОДЕЛЬ ДАННЫХ =======================================

@dataclass
class RapiraDomTrade:
    price: float
    volume: float
    time_str: str          # строка вида "15:17:49"
    ts: datetime           # время сделки (UTC, на основе сегодняшней даты)
    side: Optional[str] = None  # можно доработать по цвету/классу строки


# ============= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ============================

def _parse_number_ru(s: str) -> Optional[float]:
    """
    Преобразует строки вида '3 425.55', '1 269,99', '0.84' в float.
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
    Стабильный ключ для идентификации сделки: время + цена + объем.
    """
    return f"{t.time_str}|{t.price}|{t.volume}"


def _combine_time_today(time_str: str) -> datetime:
    """
    Берём сегодняшнюю дату + время HH:MM[:SS] из строки.
    Если парсинг не удался — возвращаем текущее время.
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


# ============= ЛОГИКА PARSE И ОПРОСА ==============================

async def _prepare_page():
    """
    Запускает браузер, открывает страницу Rapira и переключается на вкладку 'История'.
    Возвращает (playwright, browser, page).
    Вызывающий код ОБЯЗАН потом закрыть и browser, и playwright.
    """
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=True)
    page = await browser.new_page()

    log.info("Открываю страницу %s ...", RAPIRA_TRADE_URL)
    await page.goto(RAPIRA_TRADE_URL, wait_until="networkidle")

    # принять cookies (если диалог есть)
    try:
        await page.get_by_text("Я согласен").click(timeout=5000)
    except PlaywrightTimeoutError:
        pass

    # перейти на вкладку 'История'
    try:
        await page.get_by_text("История").click(timeout=7000)
    except PlaywrightTimeoutError:
        log.info("Не смог нажать на вкладку 'История' — возможно, уже активна")

    # подождать, чтобы таблица успела отрисоваться
    await page.wait_for_timeout(1000)

    return playwright, browser, page


async def _parse_trades_from_page(page) -> List[RapiraDomTrade]:
    """
    Парсит таблицу 'История' на уже открытой странице Rapira.
    Возвращает список сделок (обычно от старых к новым).
    """
    # таблица, в заголовке которой есть 'Цена', 'Объем/Объём' и 'Время'
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

        # если нужно определять BUY/SELL по классу/цвету строки — доработай здесь
        trade = RapiraDomTrade(
            price=price,
            volume=volume,
            time_str=time_text,
            ts=ts,
            side=None,
        )
        trades.append(trade)

    # на всякий случай сортируем по времени (если Rapira отдаёт новые сверху)
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
        # первый запуск — считаем новыми все (можно ограничить хвостом, если нужно)
        new_trades = trades
    else:
        try:
            idx = keys.index(_last_trade_key)
            new_trades = trades[idx + 1 :]
        except ValueError:
            # прошлую сделку не нашли: таблица могла сброситься — считаем новыми все
            new_trades = trades

    # обновляем маркер на самую свежую сделку
    _last_trade_key = _build_trade_key(trades[-1])
    return new_trades


# ============= MAIN-ЦИКЛ =========================================

async def main():
    playwright = None
    browser = None

    try:
        playwright, browser, page = await _prepare_page()
        log.info("Страница Rapira готова, начинаю опрос таблицы сделок ...")

        while True:
            try:
                new_trades = await fetch_rapira_new_trades(page)
            except PlaywrightTimeoutError:
                log.warning("Не удалось прочитать таблицу за отведенное время, пробую ещё раз ...")
                new_trades = []

            if new_trades:
                total_turnover = sum(t.price * t.volume for t in new_trades)
                log.info(
                    "Новых сделок: %d, суммарный оборот: %.2f RUB",
                    len(new_trades),
                    total_turnover,
                )
                for t in new_trades:
                    log.info(
                        "Сделка: %s | цена=%.2f | объём=%.4f",
                        t.time_str,
                        t.price,
                        t.volume,
                    )

            await asyncio.sleep(POLL_INTERVAL_SEC)

    except KeyboardInterrupt:
        log.info("Остановка по Ctrl+C")
    finally:
        if browser is not None:
            await browser.close()
        if playwright is not None:
            await playwright.stop()


if __name__ == "__main__":
    asyncio.run(main())
