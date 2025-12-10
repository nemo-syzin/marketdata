import asyncio
import logging
import subprocess
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import List, Optional

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Page,
)

# ================== НАСТРОЙКИ =======================================

RAPIRA_TRADE_URL = "https://rapira.net/trading/usdt-rub"  # URL пары
POLL_INTERVAL_SEC = 10                                    # период опроса в секундах

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("marketdata-rapira")

# маркер последней сделки, чтобы отсекать дубли
_last_trade_key: Optional[str] = None


# ================== УСТАНОВКА PLAYWRIGHT BROWSER ====================

def install_chromium() -> None:
    """
    Качаем Chromium для Playwright (как в kenigswap-rates).
    Если уже установлен — просто пройдёт без ошибки.
    """
    try:
        log.info("Installing Chromium for Playwright ...")
        subprocess.run(["playwright", "install", "chromium"], check=True)
        log.info("Chromium installed successfully.")
    except Exception as exc:
        log.warning("Playwright install error (ignored): %s", exc)


# ================== МОДЕЛЬ ДАННЫХ ===================================

@dataclass
class RapiraDomTrade:
    price: float
    volume: float
    time_str: str   # строка времени, напр. "16:25:10"
    ts: datetime    # datetime по сегодняшней дате
    side: Optional[str] = None  # BUY/SELL, если захочешь использовать


# ================== HELPERS =========================================

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
    Ключ для сделки — для отсечения дублей.
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


# ================== ПОДГОТОВКА СТРАНИЦЫ =============================

async def _try_click_history(page: Page) -> None:
    """
    Пытаемся активировать вкладку 'История сделок' несколькими способами.
    Ошибки не пробрасываем — просто логируем и идём дальше.
    """
    # 1) по роли вкладки
    try:
        await page.get_by_role("tab", name="История сделок").click(timeout=5000)
        log.info("Clicked tab by role(name='История сделок').")
        return
    except PlaywrightTimeoutError:
        pass

    # 2) по точному тексту
    for txt in ("История сделок", "История"):
        try:
            await page.get_by_text(txt, exact=False).click(timeout=5000)
            log.info("Clicked tab by text '%s'.", txt)
            return
        except PlaywrightTimeoutError:
            continue

    # 3) CSS-локатор по data-state / id (если повезёт)
    try:
        # кнопка, которая управляет rp-5-0-content-history
        await page.locator("[aria-controls='rp-5-0-content-history']").click(timeout=5000)
        log.info("Clicked tab via [aria-controls='rp-5-0-content-history'].")
        return
    except PlaywrightTimeoutError:
        pass

    log.info("Failed to click 'История' tab explicitly, maybe it's already active.")


async def _prepare_page():
    """
    Запускаем браузер, открываем страницу, принимаем cookies, пытаемся
    включить вкладку 'История', даём странице время подгрузиться.
    Никаких wait_for_selector, чтобы не падать по timeout.
    """
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = await browser.new_page(viewport={"width": 1280, "height": 720})

    log.info("Opening Rapira page %s ...", RAPIRA_TRADE_URL)
    await page.goto(RAPIRA_TRADE_URL, wait_until="domcontentloaded")

    # принять cookies, если есть
    try:
        await page.get_by_text("Я согласен", exact=False).click(timeout=5000)
        log.info("Accepted cookies.")
    except PlaywrightTimeoutError:
        log.info("No 'Я согласен' button (cookies) or not clickable.")

    # вкладка "История"
    await _try_click_history(page)

    # просто ждём несколько секунд, чтобы JS подтянул таблицы
    log.info("Waiting a bit for page JS to load data ...")
    await page.wait_for_timeout(5000)

    try:
        title = await page.title()
        log.info("Page title: %s", title)
    except Exception:
        pass

    return p, browser, page


# ================== ПАРСИНГ ТАБЛИЦЫ ================================

async def _parse_trades_from_page(page: Page) -> List[RapiraDomTrade]:
    """
    Парсим таблицу истории сделок.
    Ищем таблицу внутри div.table-responsive.table-orders и там вытаскиваем
    строки tbody tr.table-orders-row.
    Если строк нет — просто возвращаем [].
    """
    tables = page.locator("div.table-responsive.table-orders table")
    count_tables = await tables.count()
    if count_tables == 0:
        log.info("No div.table-responsive.table-orders table found yet.")
        return []

    table = tables.first

    # заголовки просто для информации
    try:
        header_cells = table.locator("thead tr").first.locator("th")
        header_count = await header_cells.count()
        headers = []
        for i in range(header_count):
            txt = (await header_cells.nth(i).inner_text()).strip()
            if txt:
                headers.append(txt)
        if headers:
            log.info("History-like table headers: %s", " | ".join(headers))
    except PlaywrightTimeoutError:
        log.info("Table has no thead (or header not found).")

    rows = table.locator("tbody tr.table-orders-row")
    row_count = await rows.count()
    log.info("table-orders-row count: %d", row_count)

    if row_count == 0:
        return []

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

        if i == 0:
            log.info(
                "First row raw: price='%s', volume='%s', time='%s'",
                price_text,
                vol_text,
                time_text,
            )

        price = _parse_number_ru(price_text)
        volume = _parse_number_ru(vol_text)
        if price is None or volume is None:
            continue

        ts = _combine_time_today(time_text)

        # side по цвету строки (если нужен)
        row_classes = (await row.get_attribute("class") or "").lower()
        side = None
        if "text-danger" in row_classes:
            side = "SELL"
        elif "text-success" in row_classes:
            side = "BUY"

        trades.append(
            RapiraDomTrade(
                price=price,
                volume=volume,
                time_str=time_text,
                ts=ts,
                side=side,
            )
        )

    trades.sort(key=lambda t: t.ts)
    return trades


async def fetch_rapira_new_trades(page: Page) -> List[RapiraDomTrade]:
    """
    Возвращает только новые сделки по сравнению с предыдущим вызовом.
    Никаких исключений наружу не кидает.
    """
    global _last_trade_key

    try:
        trades = await _parse_trades_from_page(page)
    except Exception as e:
        log.warning("Error while parsing trades: %s", e)
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
            new_trades = trades

    _last_trade_key = _build_trade_key(trades[-1])
    return new_trades


# ================== MAIN-ЦИКЛ ======================================

async def main():
    install_chromium()

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
                    "New trades: %d, total turnover: %.2f (price*volume sum)",
                    len(new_trades),
                    total_turnover,
                )
                for t in new_trades:
                    log.info(
                        "Trade %s | price=%.2f | volume=%.4f | side=%s",
                        t.time_str,
                        t.price,
                        t.volume,
                        t.side or "-",
                    )
                # здесь потом добавим запись в БД / Supabase
            else:
                log.info("No trades parsed this cycle, will retry.")

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
    
