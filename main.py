import asyncio
import logging
import os
from typing import List, Dict, Any, Tuple

from playwright.async_api import async_playwright, Browser, Page, Playwright
import subprocess
import textwrap

RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/trading/usdt-rub")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger("rapira-scraper")


# ─────────────────────────── INSTALL BROWSERS ───────────────────────────


def ensure_playwright_browsers() -> None:
    """
    Устанавливаем Chromium для Playwright, если он ещё не установлен.
    На Render среда эфемерная, поэтому делаем установку при каждом запуске
    – но в рамках одного процесса этот код вызовется один раз.
    """
    if os.getenv("PLAYWRIGHT_BROWSERS_INSTALLED") == "1":
        log.info("Playwright browsers already marked as installed, skipping.")
        return

    log.info("Installing Chromium for Playwright ...")
    # Минимальный набор – chromium. --with-deps Render позволяет
    try:
        subprocess.run(
            ["playwright", "install", "chromium", "--with-deps"],
            check=True,
        )
        os.environ["PLAYWRIGHT_BROWSERS_INSTALLED"] = "1"
        log.info("Chromium installed successfully.")
    except Exception as e:
        log.exception("Failed to install Playwright browsers: %s", e)
        raise


# ─────────────────────────── PLAYWRIGHT SETUP ───────────────────────────


async def create_browser() -> Tuple[Playwright, Browser, Page]:
    """
    Стартуем Playwright, создаём контекст/страницу, чуть маскируем headless,
    чтобы уменьшить шанс анти-бота.
    """
    ensure_playwright_browsers()

    playwright = await async_playwright().start()
    chromium = playwright.chromium

    browser = await chromium.launch(
        headless=True,
        args=[
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-blink-features=AutomationControlled",
        ],
    )

    context = await browser.new_context(
        viewport={"width": 1280, "height": 720},
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    # Убираем navigator.webdriver
    await context.add_init_script(
        """
        Object.defineProperty(navigator, 'webdriver', {
            get: () => undefined
        });
        """
    )

    page = await context.new_page()
    return playwright, browser, page


# ─────────────────────────── RAPIRA HELPERS ───────────────────────────


async def open_rapira(page: Page) -> None:
    log.info("Opening Rapira page %s ...", RAPIRA_URL)
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60000)

    # Пытаемся закрыть куки, если есть
    try:
        btn = page.locator("text='Я согласен'")
        if await btn.is_visible():
            await btn.click()
            log.info("Clicked cookies consent button.")
        else:
            log.info("No 'Я согласен' button (cookies) or not clickable.")
    except Exception:
        log.info("No 'Я согласен' button (cookies) or not clickable.")

    # Пытаемся нажать на вкладку "История", но не ругаемся, если не получилось
    try:
        log.info("Trying to click 'История' tab explicitly ...")
        history_tab = page.locator("text='История'")
        if await history_tab.is_visible():
            await history_tab.click()
            log.info("Clicked 'История' tab.")
        else:
            log.info("Tab 'История' might already be active or not visible.")
    except Exception:
        log.info("Failed to click 'История' tab explicitly, maybe it's already active.")

    # Даём фронтенду Rapira немного времени прогрузиться
    log.info("Waiting a bit for page JS to load data ...")
    await page.wait_for_timeout(8000)

    title = await page.title()
    log.info("Page title: %s", title)


async def _rows_from_locator(page: Page, selector: str) -> List[List[str]]:
    """
    Вытаскиваем текст ячеек из найденных tr через Playwright Locator.
    """
    locator = page.locator(selector)
    count = await locator.count()
    if count == 0:
        return []

    rows: List[List[str]] = []
    for i in range(min(count, 100)):  # ограничим до 100 строк за один проход
        tr = locator.nth(i)
        cells = tr.locator("th,td")
        ccount = await cells.count()
        if ccount == 0:
            continue
        texts: List[str] = []
        for j in range(ccount):
            txt = (await cells.nth(j).inner_text()).strip()
            texts.append(txt)
        # отбрасываем полностью пустые строки
        if any(texts):
            rows.append(texts)

    return rows


async def detect_any_history_table(page: Page) -> List[List[str]]:
    """
    Универсальный поиск таблицы/строк истории.
    Пробуем несколько стратегий и возвращаем первую, которая дала строки.
    """
    strategies = [
        ("tbody tr.table-orders-row", "tbody tr.table-orders-row"),
        ("tr.table-orders-row", "tr.table-orders-row"),
        ("div.table-orders table tbody tr", "div.table-orders table tbody tr"),
        ("div.table-responsive table tbody tr", "div.table-responsive table tbody tr"),
        # fallback – вообще любые tr внутри таблиц
        ("table tbody tr", "table tbody tr (generic)"),
    ]

    for label, selector in strategies:
        try:
            rows = await _rows_from_locator(page, selector)
            if rows:
                log.info(
                    "Found %d rows using strategy '%s' with selector '%s'.",
                    len(rows),
                    label,
                    selector,
                )
                return rows
            else:
                log.info("Strategy '%s' found 0 rows.", label)
        except Exception as e:
            log.warning("Strategy '%s' failed: %s", label, e)

    # Если ничего не нашли – вернём пустой список
    return []


def rows_to_trades(raw_rows: List[List[str]]) -> List[Dict[str, Any]]:
    """
    Простейшая нормализация: упаковываем строку таблицы в словарь с индексами колонок.
    Поскольку мы не знаем точную структуру, не придумываем поля.
    """
    trades: List[Dict[str, Any]] = []
    for row in raw_rows:
        trades.append(
            {
                "columns": row,
            }
        )
    return trades


async def dump_dom_snippet(page: Page, max_chars: int = 3000) -> None:
    """
    На случай, если ничего не нашли, сохраняем кусок DOM в лог,
    чтобы глазами посмотреть, какие там реальные классы/теги.
    """
    try:
        html = await page.content()
        snippet = html[:max_chars]
        log.warning(
            "DOM snippet (first %d chars):\n%s",
            max_chars,
            textwrap.indent(snippet, "    "),
        )
    except Exception as e:
        log.warning("Failed to dump DOM snippet: %s", e)


# ─────────────────────────── MAIN LOOP ───────────────────────────


async def main() -> None:
    playwright, browser, page = await create_browser()
    try:
        await open_rapira(page)

        log.info("Rapira page is ready, starting polling loop ...")

        # Основной цикл опроса
        while True:
            try:
                raw_rows = await detect_any_history_table(page)
                if not raw_rows:
                    log.info("No history rows parsed this cycle, will retry.")
                    # Один раз за запуск попробуем вывести DOM-сниппет
                    await dump_dom_snippet(page)
                else:
                    trades = rows_to_trades(raw_rows)
                    # Здесь сейчас просто логируем. Сюда можно вставить запись в Supabase.
                    log.info("Parsed %d trades:", len(trades))
                    for t in trades[:10]:  # первые 10 строк для логов
                        log.info("  %s", t["columns"])

                # Пауза между циклами опроса
                await asyncio.sleep(10)

            except Exception as cycle_error:
                log.exception("Error in polling cycle: %s", cycle_error)
                await asyncio.sleep(15)

    finally:
        await browser.close()
        await playwright.stop()


if __name__ == "__main__":
    asyncio.run(main())
