# main.py
import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any

from playwright.async_api import (
    async_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Page,
    Browser,
    Playwright,
)

# ───────────────────────────────── LOGGING ─────────────────────────────────

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("rapira-trades")

# ───────────────────────────── CONFIG ──────────────────────────────

RAPIRA_URL = os.environ.get("RAPIRA_URL", "https://rapira.net/trading/usdt-rub")
HEADLESS = os.environ.get("HEADLESS", "true").lower() != "false"
MAX_TRADES = int(os.environ.get("MAX_TRADES", "50"))
PLAYWRIGHT_INSTALL_ON_START = os.environ.get("PLAYWRIGHT_INSTALL_ON_START", "true").lower() != "false"


# ───────────────────────────── MODELS ──────────────────────────────

@dataclass
class Trade:
    exchange: str
    symbol: str
    side: Optional[str]  # "buy"/"sell" или None, если не смогли распарсить
    price: Optional[float]
    amount: Optional[float]
    total: Optional[float]
    time_str: Optional[str]   # строка времени как есть в таблице
    raw_columns: List[str]    # все ячейки строки, на всякий случай
    scraped_at: str           # ISO-время, когда мы это спарсили


# ──────────────────── PLAYWRIGHT INSTALL HELPER ────────────────────

def ensure_playwright_browsers() -> None:
    """
    Идемпотентно ставит браузер Chromium для Playwright.
    ВАЖНО: без '--with-deps', чтобы не пытаться дергать 'su' на Render.
    """
    if not PLAYWRIGHT_INSTALL_ON_START:
        log.info("Skipping playwright install (PLAYWRIGHT_INSTALL_ON_START=false).")
        return

    log.info("Ensuring Playwright Chromium is installed ...")
    try:
        # Здесь используется CLI 'playwright', который идет вместе с зависимостями.
        subprocess.run(
            ["playwright", "install", "chromium"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log.info("Playwright Chromium is installed (or already present).")
    except FileNotFoundError:
        log.warning(
            "Playwright CLI not found in PATH. "
            "Assuming browsers are already installed. If not — "
            "add 'playwright install chromium' to Render build command."
        )
    except subprocess.CalledProcessError as e:
        log.error(
            "Failed to install Playwright browsers via CLI: %s\nstdout:\n%s\nstderr:\n%s",
            e,
            e.stdout.decode("utf-8", errors="ignore") if e.stdout else "",
            e.stderr.decode("utf-8", errors="ignore") if e.stderr else "",
        )
        # Не падаем — возможно, браузеры уже стоят в кэше.


# ───────────────────────── BROWSER LIFECYCLE ───────────────────────

async def create_browser() -> tuple[Playwright, Browser, Page]:
    ensure_playwright_browsers()

    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=HEADLESS,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
        ],
    )
    page = await browser.new_page(viewport={"width": 1280, "height": 800})
    return pw, browser, page


async def close_browser(pw: Playwright, browser: Browser) -> None:
    try:
        await browser.close()
    finally:
        await pw.stop()


# ────────────────────────── HELPERS (PAGE) ─────────────────────────

async def open_rapira_trading(page: Page) -> None:
    log.info("Opening Rapira page %s ...", RAPIRA_URL)
    await page.goto(RAPIRA_URL, wait_until="networkidle", timeout=60_000)

    # Пытаемся принять cookies (если есть).
    try:
        # Вариант 1: по тексту
        btn = await page.query_selector("text=Я согласен")
        if btn:
            await btn.click()
            log.info("Clicked 'Я согласен' (cookies).")
        else:
            log.info("No 'Я согласен' (cookies) button found.")
    except PlaywrightTimeoutError:
        log.info("Timeout while trying to click cookies button (ignored).")
    except Exception as e:
        log.info("No cookies button or not clickable: %s", e)

    # Пытаемся активировать вкладку «История» / «History».
    try:
        # Пробуем несколько вариантов локаторов.
        history_locators = [
            "text=История",
            "text=History",
            "button:has-text('История')",
            "button:has-text('History')",
            "[role=tab]:has-text('История')",
            "[role=tab]:has-text('History')",
        ]
        clicked = False
        for sel in history_locators:
            el = await page.query_selector(sel)
            if el:
                await el.click()
                log.info("Clicked history tab via selector: %s", sel)
                clicked = True
                break

        if not clicked:
            log.info("History tab element not found explicitly, maybe already active.")
    except Exception as e:
        log.info("Failed to click 'История' tab explicitly, maybe already active: %s", e)

    # Немного ждем и логируем заголовок.
    await page.wait_for_timeout(5_000)
    try:
        title = await page.title()
        log.info("Page title after open: %s", title)
    except Exception:
        pass


async def find_history_rows(page: Page) -> List[Any]:
    """
    Пытается найти строки таблицы истории сделок по нескольким селекторам.
    Возвращает список элементов <tr>.
    """
    # Кандидаты селекторов: сделаны «с запасом», чтобы покрыть возможные варианты верстки.
    candidate_row_selectors = [
        "div.table-responsive table tbody tr",          # общий случай таблицы
        "table.table-orders tbody tr",                  # если есть класс table-orders
        "tbody tr.table-orders-row",                    # как ты изначально пробовал
        "div.table-orders tbody tr",                    # если таблица внутри div.table-orders
    ]

    # Сначала ждем появления хотя бы ОДНОЙ таблицы.
    log.info("Trying to detect history table ...")
    for sel in candidate_row_selectors:
        try:
            await page.wait_for_selector(sel, timeout=10_000)
            rows = await page.query_selector_all(sel)
            if rows:
                log.info("Found %d rows via selector: %s", len(rows), sel)
                return rows
            else:
                log.info("Selector %s appeared but no rows found (yet).", sel)
        except PlaywrightTimeoutError:
            log.info("Selector %s not found within timeout, trying next one.", sel)
        except Exception as e:
            log.info("Error while trying selector %s: %s", sel, e)

    log.warning("No history table rows found by any of the candidate selectors.")
    return []


async def parse_trade_row(row_el) -> Trade:
    """
    Универсальный парсер строки торгов.
    Мы не делаем жесткую привязку к позициям колонок, чтобы не падать при любом изменении верстки.
    """
    cells = await row_el.query_selector_all("td")
    texts: List[str] = []
    for cell in cells:
        txt = (await cell.inner_text()).strip()
        texts.append(" ".join(txt.split()))  # нормализуем пробелы

    # По-умолчанию все как None, а в raw_columns кладем весь текст.
    side: Optional[str] = None
    price: Optional[float] = None
    amount: Optional[float] = None
    total: Optional[float] = None
    time_str: Optional[str] = None

    # Мини-логика распознавания:
    #   - ищем колонку с "BUY"/"SELL"/"КУПЛЯ"/"ПРОДАЖА"
    #   - первую числовую берем как price, вторую как amount, третью как total
    #   - колонку, похожую на время, кладем в time_str
    numbers: List[float] = []

    def to_float_safe(val: str) -> Optional[float]:
        val = val.replace(" ", "").replace("\xa0", "")
        # меняем запятую на точку
        val = val.replace(",", ".")
        try:
            return float(val)
        except ValueError:
            return None

    for t in texts:
        low = t.lower()
        if any(x in low for x in ["buy", "покуп", "купля"]):
            side = "buy"
        if any(x in low for x in ["sell", "продаж"]):
            side = "sell"

    for t in texts:
        num = to_float_safe(t)
        if num is not None:
            numbers.append(num)

    if numbers:
        price = numbers[0]
    if len(numbers) >= 2:
        amount = numbers[1]
    if len(numbers) >= 3:
        total = numbers[2]

    # Время — первая колонка, которая визуально похожа на дату/время.
    for t in texts:
        if any(c in t for c in [":", ".", "-"]) and any(ch.isdigit() for ch in t):
            time_str = t
            break

    trade = Trade(
        exchange="rapira",
        symbol="USDT/RUB",
        side=side,
        price=price,
        amount=amount,
        total=total,
        time_str=time_str,
        raw_columns=texts,
        scraped_at=datetime.utcnow().isoformat(),
    )
    return trade


async def fetch_rapira_trades_once(page: Page, limit: int = 50) -> List[Trade]:
    """
    Открывает страницу Rapira и парсит последние сделки один раз.
    (Без бесконечного цикла — удобно для cron/job и дебага.)
    """
    await open_rapira_trading(page)

    rows = await find_history_rows(page)
    if not rows:
        log.warning("No trade rows found on Rapira page.")
        return []

    trades: List[Trade] = []
    for idx, row_el in enumerate(rows):
        if idx >= limit:
            break
        try:
            trade = await parse_trade_row(row_el)
            trades.append(trade)
        except Exception as e:
            log.warning("Failed to parse row %d: %s", idx, e)

    log.info("Parsed %d trades from Rapira.", len(trades))
    return trades


# ─────────────────────────────── MAIN ──────────────────────────────

async def main() -> None:
    log.info("Starting Rapira last-trades scraper ...")
    pw: Optional[Playwright] = None
    browser: Optional[Browser] = None
    try:
        pw, browser, page = await create_browser()
        trades = await fetch_rapira_trades_once(page, limit=MAX_TRADES)

        # Здесь два варианта:
        # 1) Просто вывести в stdout как JSON (удобно для Render cron/logs)
        # 2) Отправить в базу (Supabase, Postgres и т.д.)
        #
        # Ниже — вариант (1). Подключение БД ты можешь добавить поверх,
        # используя уже готовую структуру Trade.

        output: Dict[str, Any] = {
            "exchange": "rapira",
            "symbol": "USDT/RUB",
            "count": len(trades),
            "trades": [asdict(t) for t in trades],
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))

    except Exception as e:
        log.exception("Unhandled error in main(): %s", e)
    finally:
        if pw and browser:
            await close_browser(pw, browser)
        log.info("Scraper finished.")


if __name__ == "__main__":
    asyncio.run(main())
