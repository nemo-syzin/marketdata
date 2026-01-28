import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import httpx
from playwright.async_api import async_playwright, Browser, BrowserContext, Page

# ───────────────────────── CONFIG ─────────────────────────

# ВАЖНО: без /ru/ (у Rapira это часто "витрина", а SPA потом перекидывает куда хочет)
RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/exchange/USDT_RUB").strip()
PAIR_PATH_NEEDLE = "/exchange/USDT_RUB"  # для проверки URL
PAIR_TEXT = os.getenv("PAIR_TEXT", "USDT/RUB")  # текст пары в UI

SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

LIMIT = int(os.getenv("LIMIT", "150"))
POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))

SCRAPE_TIMEOUT_SECONDS = float(os.getenv("SCRAPE_TIMEOUT_SECONDS", "25"))
UPSERT_TIMEOUT_SECONDS = float(os.getenv("UPSERT_TIMEOUT_SECONDS", "25"))
HEARTBEAT_SECONDS = float(os.getenv("HEARTBEAT_SECONDS", "30"))
RELOAD_EVERY_SECONDS = float(os.getenv("RELOAD_EVERY_SECONDS", "600"))

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")
ON_CONFLICT = os.getenv("ON_CONFLICT", "source,symbol,trade_time,price,volume_usdt")

# Дедуп по окну
SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))

# Авто-установка браузеров Playwright (Render)
SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

# ───────────────────────── LOGGING ─────────────────────────

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    force=True,
)
logger = logging.getLogger("rapira-worker")

TIME_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}\b")
Q8 = Decimal("0.00000001")

# ───────────────────────── HELPERS ─────────────────────────

def normalize_decimal(text: str) -> Optional[Decimal]:
    t = (text or "").strip()
    if not t:
        return None
    t = t.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    try:
        return Decimal(t)
    except (InvalidOperation, ValueError):
        return None

def extract_time(text: str) -> Optional[str]:
    m = TIME_RE.search((text or "").replace("\xa0", " "))
    if not m:
        return None
    hh, mm, ss = m.group(0).split(":")
    if len(hh) == 1:
        hh = "0" + hh
    return f"{hh}:{mm}:{ss}"

def q8_str(x: Decimal) -> str:
    return str(x.quantize(Q8, rounding=ROUND_HALF_UP))

_last_install_ts = 0.0

def _playwright_install() -> None:
    global _last_install_ts
    now = time.time()
    if now - _last_install_ts < 600:
        logger.warning("Playwright install was attempted recently; skipping (cooldown).")
        return
    _last_install_ts = now

    logger.warning("Installing Playwright browsers (runtime)...")
    try:
        r = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "chromium", "chromium-headless-shell"],
            check=False,
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            logger.error(
                "playwright install failed (%s)\nSTDOUT:\n%s\nSTDERR:\n%s",
                r.returncode, r.stdout, r.stderr
            )
        else:
            logger.info("Playwright browsers installed.")
    except Exception as e:
        logger.error("Cannot run playwright install: %s", e)

def _should_force_install(err: Exception) -> bool:
    s = str(err)
    return (
        "Executable doesn't exist" in s
        or "playwright install" in s
        or "chromium_headless_shell" in s
        or ("ms-playwright" in s and "doesn't exist" in s)
    )

@dataclass(frozen=True)
class TradeKey:
    source: str
    symbol: str
    trade_time: str
    price: str
    volume_usdt: str

def trade_key(t: Dict[str, Any]) -> TradeKey:
    return TradeKey(
        source=t["source"],
        symbol=t["symbol"],
        trade_time=t["trade_time"],
        price=t["price"],
        volume_usdt=t["volume_usdt"],
    )

# ───────────────────────── UI ACTIONS ─────────────────────────

async def accept_cookies_if_any(page: Page) -> None:
    # В логах у тебя было "OK"
    for label in ["OK", "ОК", "Я согласен", "Принять", "Accept", "Согласен"]:
        try:
            btn = page.locator(f"text={label}")
            if await btn.count() > 0 and await btn.first.is_visible():
                logger.info("Found cookies banner, clicking '%s'...", label)
                await btn.first.click(timeout=5_000, no_wait_after=True)
                await page.wait_for_timeout(300)
                return
        except Exception:
            pass

async def ensure_last_trades_tab(page: Page) -> None:
    # В UI таб называется "Последние сделки"
    try:
        tab = page.locator("text=Последние сделки")
        if await tab.count() > 0 and await tab.first.is_visible():
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
    except Exception:
        pass

async def _find_top_pair_button_text(page: Page) -> Optional[str]:
    """
    Находит текст кнопки пары вверху (самый верхний матч вида AAA/BBB).
    Мы берем элемент с минимальным y (topmost), чтобы не спутать с таблицей сделок.
    """
    js = r"""
    () => {
      const re = /^[A-Z0-9]{2,10}\/[A-Z0-9]{2,10}$/;
      const nodes = Array.from(document.querySelectorAll("a,button,div,span"))
        .filter(el => {
          const t = (el.innerText || "").trim();
          if (!re.test(t)) return false;
          const r = el.getBoundingClientRect();
          // отсекаем таблицу: берём верхнюю зону страницы
          return r.width > 30 && r.height > 10 && r.top >= 0 && r.top < 220 && r.left < 600;
        })
        .map(el => {
          const r = el.getBoundingClientRect();
          return { text: (el.innerText || "").trim(), top: r.top, left: r.left };
        })
        .sort((a,b) => (a.top - b.top) || (a.left - b.left));
      return nodes.length ? nodes[0].text : null;
    }
    """
    try:
        return await page.evaluate(js)
    except Exception:
        return None

async def force_select_pair_ui(page: Page) -> None:
    """
    Жёстко выбираем пару PAIR_TEXT через UI, даже если нас редиректнуло.
    Алгоритм:
      1) находим верхний переключатель пары (текст вида BTC/USDT и т.п.)
      2) кликаем
      3) в открывшемся меню ищем и кликаем USDT/RUB (или через поиск, если есть input)
    """
    current_pair = await _find_top_pair_button_text(page)
    if not current_pair:
        # fallback: иногда виден прямо нужный текст — просто кликаем по нему вверху
        current_pair = "BTC/USDT"

    # 1) клик по текущей паре вверху
    try:
        locator = page.locator(f"xpath=//*[self::a or self::button or self::div or self::span][normalize-space(text())='{current_pair}']")
        if await locator.count() > 0:
            await locator.first.click(timeout=8_000, no_wait_after=True)
            await page.wait_for_timeout(350)
    except Exception:
        pass

    # 2) если есть поле поиска — используем его
    try:
        search_input = page.locator("xpath=//input[@type='text' or @type='search']")
        if await search_input.count() > 0 and await search_input.first.is_visible():
            await search_input.first.fill(PAIR_TEXT, timeout=3_000)
            await page.wait_for_timeout(250)
    except Exception:
        pass

    # 3) кликаем по пункту USDT/RUB в выпадающем меню
    # Берём первый видимый матч
    try:
        item = page.locator(f"text={PAIR_TEXT}")
        if await item.count() > 0:
            # иногда первый матч в таблице — поэтому выбираем видимый и ближе к верху
            # (в большинстве случаев item.first — как раз в меню)
            await item.first.click(timeout=8_000, no_wait_after=True)
            await page.wait_for_timeout(800)
    except Exception:
        pass

async def lock_pair(page: Page, attempts: int = 6) -> None:
    """
    Гарантируем, что мы на USDT_RUB.
    Если Rapira SPA нас уводит на BTC_USDT, возвращаем через UI выбор пары.
    """
    last_url = ""
    for i in range(1, attempts + 1):
        url = page.url or ""
        last_url = url

        ok_url = (PAIR_PATH_NEEDLE in url)
        ok_text = False
        try:
            # Проверяем, что в верхней зоне есть нужный текст пары
            ok_text = bool(await page.locator(f"xpath=//*[normalize-space(text())='{PAIR_TEXT}']").first.is_visible(timeout=1500))
        except Exception:
            ok_text = False

        if ok_url and ok_text:
            return

        logger.warning(
            "Wrong pair detected (attempt %d). url=%s; need %s and text %s",
            i, url, PAIR_PATH_NEEDLE, PAIR_TEXT
        )

        # Пытаемся "прибить" SPA: очистить storage, заново перейти по URL, затем выбрать пару UI
        try:
            await page.evaluate("() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e) {} }")
        except Exception:
            pass

        try:
            await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
        except Exception:
            # если goto упал — просто продолжим UI-попытку
            pass

        await page.wait_for_timeout(900)
        await accept_cookies_if_any(page)
        await force_select_pair_ui(page)
        await ensure_last_trades_tab(page)
        await page.wait_for_timeout(300)

    raise RuntimeError(f"Cannot lock pair USDT_RUB. final_url={last_url}")

# ───────────────────────── PARSING ─────────────────────────

# По твоему DevTools: 1-й td = курс, 2-й = объём USDT, 3-й = время
# Селектор делаем мягче, чтобы пережить мелкие правки классов:
TRADE_ROWS_SELECTOR = "table tbody tr.table-orders-row"

EVAL_JS = """
(rows, limit) => rows.slice(0, limit).map(row => {
  const tds = Array.from(row.querySelectorAll('td'));
  const texts = tds.map(td => (td.innerText || '').trim());
  return { texts };
})
"""

def parse_row_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    texts: List[str] = (payload.get("texts") or [])
    if len(texts) < 3:
        return None

    price = normalize_decimal(texts[0])
    volume_usdt = normalize_decimal(texts[1])
    trade_time = extract_time(texts[2])

    if price is None or volume_usdt is None or trade_time is None:
        return None
    if price <= 0 or volume_usdt <= 0:
        return None

    volume_rub = price * volume_usdt

    return {
        "source": SOURCE,
        "symbol": SYMBOL,
        "price": q8_str(price),
        "volume_usdt": q8_str(volume_usdt),
        "volume_rub": q8_str(volume_rub),
        "trade_time": trade_time,
    }

async def scrape_window_fast(page: Page) -> List[Dict[str, Any]]:
    await page.wait_for_selector(TRADE_ROWS_SELECTOR, timeout=int(SCRAPE_TIMEOUT_SECONDS * 1000))
    payloads = await page.eval_on_selector_all(TRADE_ROWS_SELECTOR, EVAL_JS, LIMIT)

    out: List[Dict[str, Any]] = []
    for p in payloads:
        t = parse_row_payload(p)
        if t:
            out.append(t)
    return out

# ───────────────────────── SUPABASE ─────────────────────────

def _sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates,return=minimal",
    }

async def supabase_upsert(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("SUPABASE_URL or SUPABASE_KEY not set; skipping insert.")
        return

    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}"
    params = {"on_conflict": ON_CONFLICT}

    async with httpx.AsyncClient(timeout=UPSERT_TIMEOUT_SECONDS) as client:
        for i in range(0, len(rows), UPSERT_BATCH):
            chunk = rows[i:i + UPSERT_BATCH]
            r = await client.post(url, headers=_sb_headers(), params=params, json=chunk)
            if r.status_code >= 300:
                logger.error("Supabase upsert failed (%s): %s", r.status_code, r.text)
                return

    logger.info("Inserted (or ignored duplicates) %d rows into '%s'.", len(rows), SUPABASE_TABLE)

# ───────────────────────── BROWSER SESSION ─────────────────────────

async def open_browser(pw) -> Tuple[Browser, BrowserContext, Page]:
    browser = await pw.chromium.launch(
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )

    context = await browser.new_context(
        viewport={"width": 1440, "height": 810},
        locale="ru-RU",
        timezone_id="Europe/Moscow",
        user_agent=(
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ),
    )

    # Правильная сигнатура add_init_script в Playwright Python:
    # add_init_script(script) или add_init_script(path=...)
    # Мы чистим storage до запуска SPA (уменьшает шанс «помнит BTC/USDT»)
    try:
        await context.add_init_script(
            "() => { try { localStorage.clear(); sessionStorage.clear(); } catch(e) {} }"
        )
    except Exception:
        pass

    page = await context.new_page()
    page.set_default_timeout(10_000)

    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(1000)
    await accept_cookies_if_any(page)

    # Критично: лочим пару (если Rapira уводит на BTC/USDT — вернём через UI)
    await lock_pair(page, attempts=6)

    await ensure_last_trades_tab(page)
    await page.wait_for_timeout(400)
    return browser, context, page

async def safe_close(browser: Optional[Browser], context: Optional[BrowserContext], page: Optional[Page]) -> None:
    try:
        if page:
            await page.close()
    except Exception:
        pass
    try:
        if context:
            await context.close()
    except Exception:
        pass
    try:
        if browser:
            await browser.close()
    except Exception:
        pass

# ───────────────────────── WORKER LOOP ─────────────────────────

async def worker() -> None:
    seen: Set[TradeKey] = set()
    seen_q: Deque[TradeKey] = deque()

    backoff = 2.0
    last_heartbeat = time.monotonic()
    last_reload = time.monotonic()
    last_pair_check = 0.0

    async with async_playwright() as pw:
        browser: Optional[Browser] = None
        context: Optional[BrowserContext] = None
        page: Optional[Page] = None

        while True:
            try:
                if page is None:
                    logger.info("Starting browser session...")
                    try:
                        browser, context, page = await open_browser(pw)
                    except Exception as e:
                        if (not SKIP_BROWSER_INSTALL) or _should_force_install(e):
                            _playwright_install()
                            browser, context, page = await open_browser(pw)
                        else:
                            raise
                    backoff = 2.0
                    last_reload = time.monotonic()
                    last_heartbeat = time.monotonic()
                    last_pair_check = time.monotonic()

                now = time.monotonic()

                if now - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url)
                    last_heartbeat = now

                if now - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(1000)
                    await accept_cookies_if_any(page)
                    await lock_pair(page, attempts=6)
                    await ensure_last_trades_tab(page)
                    last_reload = now

                # ВАЖНО: Rapira может «самопроизвольно» переключать пару.
                # Поэтому раз в ~20 сек проверяем и при необходимости возвращаем USDT/RUB.
                if now - last_pair_check >= 20:
                    await lock_pair(page, attempts=3)
                    last_pair_check = now

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Reloading page...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(1000)
                    await accept_cookies_if_any(page)
                    await lock_pair(page, attempts=6)
                    await ensure_last_trades_tab(page)
                    await asyncio.sleep(max(0.5, POLL_SECONDS))
                    continue

                new_rows: List[Dict[str, Any]] = []
                for t in reversed(window):
                    k = trade_key(t)
                    if k in seen:
                        continue
                    new_rows.append(t)

                    seen.add(k)
                    seen_q.append(k)
                    if len(seen_q) > SEEN_MAX:
                        old = seen_q.popleft()
                        seen.discard(old)

                if new_rows:
                    logger.info(
                        "Parsed %d new trades. Newest: %s",
                        len(new_rows),
                        json.dumps(new_rows[-1], ensure_ascii=False),
                    )
                    await supabase_upsert(new_rows)

                sleep_s = max(0.35, POLL_SECONDS + random.uniform(-0.15, 0.15))
                await asyncio.sleep(sleep_s)

            except asyncio.TimeoutError:
                logger.error("Timeout during scrape. Restarting browser session...")
                await safe_close(browser, context, page)
                browser = context = page = None

            except Exception as e:
                logger.error("Worker error: %s", e)

                if (not SKIP_BROWSER_INSTALL) or _should_force_install(e):
                    _playwright_install()

                logger.info("Retrying after %.1fs ...", backoff)
                await asyncio.sleep(backoff)
                backoff = min(60.0, backoff * 2)

                await safe_close(browser, context, page)
                browser = context = page = None

def main() -> None:
    asyncio.run(worker())

if __name__ == "__main__":
    main()
