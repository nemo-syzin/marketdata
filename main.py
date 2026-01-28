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
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# ───────────────────────── CONFIG ─────────────────────────

SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

# Целевая пара и страницы
PAIR_CODE = os.getenv("PAIR_CODE", "USDT_RUB")  # часть URL
RAPIRA_BASE = os.getenv("RAPIRA_BASE", "https://rapira.net").rstrip("/")
RAPIRA_EXCHANGE_RU = f"{RAPIRA_BASE}/ru/exchange"
RAPIRA_PAIR_URL = os.getenv("RAPIRA_URL", f"{RAPIRA_EXCHANGE_RU}/{PAIR_CODE}")

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

SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"
SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))

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

# ───────────────────────── PAGE ACTIONS ─────────────────────────

async def accept_cookies_if_any(page: Page) -> None:
    # Rapira часто показывает баннер, он мешает кликам.
    for label in ["OK", "Я согласен", "Принять", "Accept"]:
        try:
            btn = page.get_by_text(label, exact=True)
            if await btn.count() > 0:
                logger.info("Found cookies banner, clicking '%s'...", label)
                await btn.first.click(timeout=5_000, no_wait_after=True)
                await page.wait_for_timeout(350)
                return
        except Exception:
            pass


async def ensure_last_trades_tab(page: Page) -> None:
    # На странице есть вкладки "Стакан / Последние сделки"
    try:
        tab = page.get_by_text("Последние сделки", exact=False)
        if await tab.count() > 0:
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(250)
    except Exception:
        pass


def _is_wrong_pair_url(url: str) -> bool:
    u = (url or "").lower()
    # В логах у тебя именно это: /exchange/BTC_USDT и иногда /
    if "/exchange/btc_usdt" in u:
        return True
    # иногда выкидывает на корень
    if u.rstrip("/") == RAPIRA_BASE.lower():
        return True
    return False


async def _page_shows_usdt_rub(page: Page) -> bool:
    """
    Проверяем не только URL, но и то, что UI реально показывает USDT/RUB.
    Ты дал разметку:
      <span class="fw-bold fs-6"><span>USDT</span><span class="fw-normal">/RUB</span></span>
    """
    try:
        loc = page.locator("span.fw-bold.fs-6")
        if await loc.count() == 0:
            return False
        # Берём видимый заголовок пары (как в твоём DOM)
        txt = (await loc.first.inner_text(timeout=2_000)).strip().replace(" ", "")
        return "USDT/RUB" in txt
    except Exception:
        return False


async def open_pair_menu(page: Page) -> None:
    # Меню выбора пары: div.trading-pair-block-menu (видно на твоём скрине)
    candidates = [
        "div.trading-pair-block-menu",
        "div.trading-pair-block-menu.d-flex",
        "div.trading-pair-block-menu.cursor-pointer",
    ]
    for sel in candidates:
        try:
            m = page.locator(sel)
            if await m.count() > 0:
                await m.first.click(timeout=5_000, no_wait_after=True)
                await page.wait_for_timeout(250)
                return
        except Exception:
            pass
    # Fallback: клик по заголовку USDT/RUB (если уже отображается)
    try:
        title = page.locator("span.fw-bold.fs-6")
        if await title.count() > 0:
            await title.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(250)
    except Exception:
        pass


async def click_usdt_rub_in_list(page: Page) -> None:
    """
    На странице /ru/exchange при открытом меню есть таблица пар:
      tr.cursor-pointer.trading-pair-row ...
    Мы кликаем строку, где есть текст USDT и /RUB.
    """
    # ждём, пока список пар появился
    await page.wait_for_selector("tr.trading-pair-row", timeout=15_000)

    # самый устойчивый вариант — искать по тексту строки
    row = page.locator("tr.trading-pair-row").filter(has_text="USDT").filter(has_text="/RUB").first
    if await row.count() == 0:
        # иногда текст склеивается в USDT/RUB
        row = page.locator("tr.trading-pair-row").filter(has_text="USDT/RUB").first

    if await row.count() == 0:
        raise RuntimeError("Cannot find USDT/RUB row in pair list")

    await row.click(timeout=8_000, no_wait_after=True)


async def lock_pair_usdt_rub(page: Page, attempts: int = 10) -> None:
    """
    Принудительно "закрепляем" пару USDT/RUB через UI.
    Это важнее, чем просто page.goto(/USDT_RUB), потому что SPA может перекинуть обратно.
    """
    for i in range(1, attempts + 1):
        url = page.url or ""
        ok_url = (PAIR_CODE.lower() in url.lower()) or (f"/ru/exchange/{PAIR_CODE}".lower() in url.lower())
        ok_ui = await _page_shows_usdt_rub(page)

        if ok_url and ok_ui and not _is_wrong_pair_url(url):
            return

        logger.warning(
            "Wrong pair detected (attempt %d). url=%s; need %s and UI USDT/RUB",
            i, url, RAPIRA_PAIR_URL
        )

        # 1) Идём на общий exchange, 2) открываем меню, 3) кликаем USDT/RUB
        try:
            await page.goto(RAPIRA_EXCHANGE_RU, wait_until="domcontentloaded", timeout=60_000)
        except Exception:
            pass

        await page.wait_for_timeout(700)
        await accept_cookies_if_any(page)

        await open_pair_menu(page)
        await click_usdt_rub_in_list(page)

        # ждём, что роутер реально переключил страницу на нужный URL
        try:
            await page.wait_for_url(f"**/{PAIR_CODE}", timeout=20_000)
        except Exception:
            # если роутер не успел — не страшно, проверим дальше
            pass

        await page.wait_for_timeout(900)

    raise RuntimeError(f"Cannot lock pair {PAIR_CODE}. final_url={page.url}")

# ───────────────────────── PARSING ─────────────────────────

TRADE_ROWS_SELECTOR = (
    "div.table-responsive.table-orders "
    "table.table-row-dashed tbody tr.table-orders-row"
)

EVAL_JS = """
(rows, limit) => rows.slice(0, limit).map(row => {
  const tds = Array.from(row.querySelectorAll('td'));
  const texts = tds.map(td => (td.innerText || '').trim());
  const priceTd = row.querySelector('td.text-success') || row.querySelector('td.text-danger');
  const priceHint = priceTd ? (priceTd.innerText || '').trim() : null;
  return { texts, priceHint };
})
"""


def parse_row_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        texts: List[str] = payload.get("texts") or []
        if len(texts) < 3:
            return None

        trade_time = None
        time_idx = None
        for idx, txt in enumerate(texts):
            t = extract_time(txt)
            if t:
                trade_time = t
                time_idx = idx
                break
        if not trade_time:
            return None

        price: Optional[Decimal] = None
        price_hint = payload.get("priceHint")
        if price_hint:
            price = normalize_decimal(price_hint)

        nums: List[Decimal] = []
        for idx, txt in enumerate(texts):
            if idx == time_idx:
                continue
            n = normalize_decimal(txt)
            if n is not None:
                nums.append(n)
        if len(nums) < 2:
            return None

        if price is None:
            # Цена для USDT/RUB выглядит как ~70-120 в интерфейсе, но у тебя в БД было ~90000:
            # это признак того, что ты парсил не то поле/не ту таблицу.
            # Здесь мы берем priceHint (td.text-success) в приоритете.
            for n in nums:
                if Decimal("30") <= n <= Decimal("300"):
                    price = n
                    break
        if price is None:
            price = nums[0]

        volume_usdt: Optional[Decimal] = None
        for n in nums:
            if n != price:
                volume_usdt = n
                break
        if volume_usdt is None:
            volume_usdt = nums[1]

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
    except Exception:
        return None


async def scrape_window_fast(page: Page) -> List[Dict[str, Any]]:
    t0 = time.monotonic()
    await page.wait_for_selector(TRADE_ROWS_SELECTOR, timeout=int(SCRAPE_TIMEOUT_SECONDS * 1000))

    payloads = await page.eval_on_selector_all(
        TRADE_ROWS_SELECTOR,
        EVAL_JS,
        LIMIT,
    )

    out: List[Dict[str, Any]] = []
    for p in payloads:
        t = parse_row_payload(p)
        if t:
            out.append(t)

    dt = time.monotonic() - t0
    if dt > SCRAPE_TIMEOUT_SECONDS:
        raise asyncio.TimeoutError(f"scrape_window_fast took {dt:.2f}s")

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
            try:
                r = await client.post(url, headers=_sb_headers(), params=params, json=chunk)
            except Exception as e:
                logger.error("Supabase POST error: %s", e)
                return

            if r.status_code >= 300:
                logger.error("Supabase upsert failed (%s): %s", r.status_code, r.text)
                return

        logger.info("Inserted (or ignored duplicates) %d rows into '%s'.", len(rows), SUPABASE_TABLE)

# ───────────────────────── BROWSER SESSION ─────────────────────────

INIT_JS_GUARD = f"""
(() => {{
  const good = "{RAPIRA_EXCHANGE_RU}/{PAIR_CODE}";
  const bad1 = "/exchange/BTC_USDT";
  const bad2 = "/exchange/btc_usdt";

  const fixIfBad = () => {{
    try {{
      const p = (location.pathname || "");
      if (p.includes(bad1) || p.includes(bad2) || location.href === "{RAPIRA_BASE}/" || location.href === "{RAPIRA_BASE}") {{
        location.replace(good);
      }}
    }} catch(e) {{}}
  }};

  // SPA может сама пушить роут. Мы не ломаем роутер, но постоянно “сторожим” переход.
  setInterval(fixIfBad, 200);
}})();
"""


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
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    # ВАЖНО: правильный вызов add_init_script
    await context.add_init_script(script=INIT_JS_GUARD)

    page = await context.new_page()
    page.set_default_timeout(10_000)

    # Заходим на exchange (не на конкретную пару) и кликом “фиксируем” USDT/RUB
    await page.goto(RAPIRA_EXCHANGE_RU, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(900)
    await accept_cookies_if_any(page)

    await lock_pair_usdt_rub(page, attempts=10)
    await ensure_last_trades_tab(page)
    await page.wait_for_timeout(300)

    logger.info("Opened %s final_url=%s", RAPIRA_EXCHANGE_RU, page.url)
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
    last_lock = 0.0

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
                    last_lock = time.monotonic()

                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url if page else None)
                    last_heartbeat = time.monotonic()

                # периодический reload (и повторный lock)
                if time.monotonic() - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await page.goto(RAPIRA_EXCHANGE_RU, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(900)
                    await accept_cookies_if_any(page)
                    await lock_pair_usdt_rub(page, attempts=10)
                    await ensure_last_trades_tab(page)
                    last_reload = time.monotonic()
                    last_lock = time.monotonic()

                # guard: если сайт снова увёл — возвращаемся
                if page and (_is_wrong_pair_url(page.url) or not await _page_shows_usdt_rub(page)):
                    if time.monotonic() - last_lock > 2.0:
                        await lock_pair_usdt_rub(page, attempts=10)
                        await ensure_last_trades_tab(page)
                        last_lock = time.monotonic()

                window = await scrape_window_fast(page)

                if not window:
                    logger.warning("No rows parsed. Re-locking & reloading...")
                    await lock_pair_usdt_rub(page, attempts=10)
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
                logger.error("Timeout. Restarting browser session...")
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
