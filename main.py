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
POLL_INTERVAL_SEC = 10                                    # период опроса

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("marketdata-rapira")

_last_trade_key: Optional[str] = None  # маркер последней сделки


# ================== УСТАНОВКА PLAYWRIGHT BROWSER ====================

def install_chromium() -> None:
    """
    Качаем Chromium для Playwright (как в рабочем проекте kenigswap-rates).
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
    side: Optional[str] = None  # BUY/SELL, если захочешь добавить позже


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
    Стабильный ключ для сделки — для отсечения дублей.
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

async def _prepare_page():
    """
    Запускаем браузер, открываем страницу, переключаемся на вкладку 'История'
    и ждём появления строк истории сделок (tbody tr.table-orders-row).
    """
    p = await async_playwright().start()
    browser = await p.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = await browser.new_page(viewport={"width": 1280, "height": 720})

    log.info("Opening Rapira page %s ...", RAPIRA_TRADE_URL)
    # domcontentloaded — чтобы не ждать бесконечные веб-сокеты
    await page.goto(RAPIRA_TRADE_URL, wait_until="domcontentloaded")

    # принять cookies, если есть
    try:
        await page.get_by_text("Я согласен").click(timeout=5000)
    except PlaywrightTimeoutError:
        pass

    # вкладка "История" (на всякий случай)
    try:
        await page.get_by_text("История").click(timeout=10000)
    except PlaywrightTimeoutError:
        log.info("Tab 'История' might already be active")

    # Ждём ПРЯМО строки истории, без привязки к id
    # это те самые <tr class="table-orders-row border-0 fs-7 lh-1">...
    log.info("Waiting for history rows (tbody tr.table-orders-row) ...")
    await page.wait_for_selector("tbody tr.table-orders-row", timeout=30000)
    log.info("History rows are present, page is ready.")

    # небольшая пауза, чтобы DOM устаканился
    await page.wait_for_timeout(1000)

    return p, browser, page


# ================== ПАРСИНГ КОНКРЕТНОЙ ТАБЛИЦЫ =====================

async def _parse_trades_from_page(page: Page) -> List[RapiraDomTrade]:
    """
    Парсим таблицу истории сделок.
    Ищем таблицу, у которой есть строки tr.table-orders-row, и вытаскиваем
    первые три ячейки: цена, объём, время.
    """
    # самая «узкая» привязка к таблице истории
    table = page.locator("div.table-responsive.table-orders table").first

    # заголовки — просто для логов
    try:
        headers = await table.locator("thead tr").first.locator("th").all_inner_texts()
        headers = [h.strip() for h in headers if h.strip()]
        if headers:
            log.info("History table headers: %s", " | ".join(headers))
        else:
            log.info("History table: no thead headers detected")
    except PlaywrightTimeoutError:
        log.info("History table: no thead found")

    # строки сделок
    rows = table.locator("tbody tr.table-orders-row")
    row_count = await rows.count()
    log.info("History table rows: %d", row_count)

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

        # можно вытащить side по классу (text-danger / text-success), если нужно
        row_classes = (await row.get_attribute("class") or "").lower()
        side = None
        if "text-danger" in row_classes:
            side = "SELL"
        elif "text-success" in row_classes:
            side = "BUY"

        trade = RapiraDomTrade(
            price=price,
            volume=volume,
            time_str=time_text,
            ts=ts,
            side=side,
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
        log.warning("parse error: %s", e)
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
                    "New trades: %d, total turnover: %.2f RUB",
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
                # тут потом добавим запись в БД / Supabase

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
