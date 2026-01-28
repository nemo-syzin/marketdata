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
from playwright.async_api import Browser, BrowserContext, Page, async_playwright

# ───────────────────────── CONFIG ─────────────────────────

# Важно: у Rapira часто нормализация /ru -> без /ru.
# Поэтому держим оба URL и используем "lock_pair".
RAPIRA_URL = os.getenv("RAPIRA_URL", "https://rapira.net/ru/exchange/USDT_RUB").strip()
RAPIRA_URL_ALT = os.getenv("RAPIRA_URL_ALT", "https://rapira.net/exchange/USDT_RUB").strip()

PAIR_URL_TOKEN = "USDT_RUB"
PAIR_TEXT = os.getenv("PAIR_TEXT", "USDT/RUB").strip()

SOURCE = os.getenv("SOURCE", "rapira").strip()
SYMBOL = os.getenv("SYMBOL", "USDT/RUB").strip()

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

SEEN_MAX = int(os.getenv("SEEN_MAX", "20000"))
UPSERT_BATCH = int(os.getenv("UPSERT_BATCH", "200"))

SKIP_BROWSER_INSTALL = os.getenv("SKIP_BROWSER_INSTALL", "0") == "1"

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
    # Rapira может отдавать неразрывные пробелы/разделители тысяч
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
    for label in ["Я согласен", "Принять", "Accept", "OK", "ОК"]:
        try:
            btn = page.locator(f"text={label}")
            if await btn.count() > 0:
                await btn.first.click(timeout=5_000, no_wait_after=True)
                await page.wait_for_timeout(250)
                return
        except Exception:
            pass


async def ensure_last_trades_tab(page: Page) -> None:
    # В твоём DOM вкладка называется "Последние сделки"
    try:
        tab = page.locator("text=Последние сделки")
        if await tab.count() > 0:
            await tab.first.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(200)
    except Exception:
        pass


async def _page_has_pair_text(page: Page) -> bool:
    try:
        loc = page.locator(f"text={PAIR_TEXT}")
        return (await loc.count()) > 0
    except Exception:
        return False


def _url_has_pair(url: str) -> bool:
    u = (url or "").lower()
    return ("exchange/usdt_rub" in u) or ("exchange/usdt_rub".replace("/", "_") in u) or (PAIR_URL_TOKEN.lower() in u)


async def _set_storage_pair_hints(page: Page) -> None:
    # Мы не знаем точные ключи Rapira, поэтому выставляем типовые “last symbol” варианты.
    # Важно: это выполняется ДО старта приложения через add_init_script (см. open_browser).
    await page.evaluate(
        """(pairUnderscore, pairSlash) => {
          try {
            const pairs = [pairUnderscore, pairSlash];
            const keys = [
              'lastSymbol', 'lastsymbol',
              'symbol', 'Symbol',
              'pair', 'Pair',
              'market', 'Market',
              'instrument', 'Instrument',
              'selectedSymbol', 'selectedPair',
              'tv_symbol', 'tvSymbol', 'tv:symbol',
              'terminal:lastSymbol', 'terminal:lastPair'
            ];
            for (const k of keys) {
              for (const v of pairs) {
                try { localStorage.setItem(k, v); } catch(e) {}
                try { sessionStorage.setItem(k, v); } catch(e) {}
              }
            }
          } catch(e) {}
        }""",
        PAIR_URL_TOKEN,
        PAIR_TEXT,
    )


async def _clear_storage(page: Page) -> None:
    await page.evaluate(
        """() => {
          try { localStorage.clear(); } catch(e) {}
          try { sessionStorage.clear(); } catch(e) {}
        }"""
    )


async def _ui_try_select_pair(page: Page) -> bool:
    """
    Универсальная попытка:
    1) клик по элементу с текущей парой (часто BTC/USDT)
    2) найти поле поиска (input) и ввести USDT/RUB
    3) клик по результату с текстом USDT/RUB
    """
    # 1) пробуем кликнуть по "BTC/USDT" или вообще по любому "/USDT" рядом
    candidates = [
        "text=BTC/USDT",
        "text=BTC_USDT",
        "text=/USDT",
        "text=/RUB",
    ]

    clicked = False
    for sel in candidates:
        try:
            el = page.locator(sel).first
            if await el.count() > 0:
                # клик по ближайшему кликабельному предку
                await el.evaluate(
                    """(node) => {
                      function isClickable(n){
                        if(!n) return false;
                        const tag = (n.tagName || '').toLowerCase();
                        if(tag === 'button' || tag === 'a') return true;
                        const role = (n.getAttribute && n.getAttribute('role')) || '';
                        if(role.toLowerCase() === 'button' || role.toLowerCase() === 'tab') return true;
                        return false;
                      }
                      let cur = node;
                      for(let i=0;i<6;i++){
                        if(isClickable(cur)) { cur.click(); return; }
                        cur = cur.parentElement;
                        if(!cur) break;
                      }
                      node.click();
                    }"""
                )
                await page.wait_for_timeout(350)
                clicked = True
                break
        except Exception:
            pass

    if not clicked:
        return False

    # 2) поиск инпута
    search_inputs = [
        "input[placeholder*='Поиск']",
        "input[placeholder*='иск']",
        "input[type='text']",
    ]

    inp = None
    for s in search_inputs:
        try:
            loc = page.locator(s).filter(has_not=page.locator("[disabled]")).first
            if await loc.count() > 0:
                # должен быть видимым
                if await loc.is_visible():
                    inp = loc
                    break
        except Exception:
            pass

    if inp is None:
        return False

    try:
        await inp.click(timeout=3_000)
        await inp.fill(PAIR_TEXT, timeout=3_000)
        await page.wait_for_timeout(300)
    except Exception:
        return False

    # 3) выбор результата
    try:
        res = page.locator(f"text={PAIR_TEXT}").first
        if await res.count() > 0:
            await res.click(timeout=5_000, no_wait_after=True)
            await page.wait_for_timeout(800)
            return True
    except Exception:
        pass

    return False


async def lock_pair(page: Page) -> None:
    """
    Делает состояние "пара USDT/RUB активна" устойчивым.
    Проблема: SPA может переписать URL на BTC_USDT после domcontentloaded.
    Решение: несколько стратегий подряд.
    """
    for attempt in range(1, 8):
        await accept_cookies_if_any(page)
        await ensure_last_trades_tab(page)

        url = page.url
        has_pair_text = await _page_has_pair_text(page)
        if _url_has_pair(url) and has_pair_text:
            logger.info("Pair locked OK. url=%s", url)
            return

        logger.warning(
            "Wrong pair detected (attempt %d). url=%s; need /exchange/%s and text %s",
            attempt, url, PAIR_URL_TOKEN, PAIR_TEXT
        )

        # A) альтернативный URL
        if attempt in (1, 2):
            try:
                await page.goto(RAPIRA_URL_ALT, wait_until="domcontentloaded", timeout=60_000)
                await page.wait_for_timeout(800)
                continue
            except Exception:
                pass

        # B) очистка storage + подсказки пары + reload
        if attempt in (3, 4):
            try:
                await _clear_storage(page)
                await _set_storage_pair_hints(page)
                await page.reload(wait_until="domcontentloaded", timeout=60_000)
                await page.wait_for_timeout(800)
                continue
            except Exception:
                pass

        # C) UI-переключение пары
        if attempt in (5, 6):
            ok = await _ui_try_select_pair(page)
            if ok:
                await accept_cookies_if_any(page)
                await ensure_last_trades_tab(page)
                await page.wait_for_timeout(600)
                # проверим ещё раз
                url2 = page.url
                has_pair_text2 = await _page_has_pair_text(page)
                if _url_has_pair(url2) and has_pair_text2:
                    logger.info("Pair locked via UI. url=%s", url2)
                    return

        # D) жёсткий переход на основной URL
        try:
            await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(900)
        except Exception:
            await page.wait_for_timeout(900)

    raise RuntimeError("Cannot lock pair USDT/RUB (site keeps switching).")

# ───────────────────────── PARSING ─────────────────────────

# Более гибкий селектор под твой DOM:
# div.table-responsive.table-orders-scroll -> table -> tbody -> tr.table-orders-row
TRADE_ROWS_SELECTOR = "div.table-responsive table tbody tr.table-orders-row"

EVAL_TRADES_JS = """
(rows, limit) => rows.slice(0, limit).map(row => {
  const tds = Array.from(row.querySelectorAll('td'));
  const a = (tds[0]?.innerText || '').trim();
  const b = (tds[1]?.innerText || '').trim();
  const c = (tds[2]?.innerText || '').trim();
  return { price: a, volume: b, time: c };
})
"""

def parse_trade_payload(p: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    price_txt = (p.get("price") or "").strip()
    vol_txt = (p.get("volume") or "").strip()
    time_txt = (p.get("time") or "").strip()

    trade_time = extract_time(time_txt)
    if not trade_time:
        return None

    price = normalize_decimal(price_txt)
    vol = normalize_decimal(vol_txt)
    if price is None or vol is None:
        return None
    if price <= 0 or vol <= 0:
        return None

    volume_usdt = vol
    volume_rub = price * volume_usdt

    return {
        "source": SOURCE,
        "symbol": SYMBOL,
        "price": q8_str(price),
        "volume_usdt": q8_str(volume_usdt),
        "volume_rub": q8_str(volume_rub),
        "trade_time": trade_time,
    }


async def scrape_trades(page: Page) -> List[Dict[str, Any]]:
    await page.wait_for_selector(TRADE_ROWS_SELECTOR, timeout=int(SCRAPE_TIMEOUT_SECONDS * 1000))
    payloads = await page.eval_on_selector_all(TRADE_ROWS_SELECTOR, EVAL_TRADES_JS, LIMIT)

    out: List[Dict[str, Any]] = []
    for p in payloads:
        t = parse_trade_payload(p)
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
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
    )

    # Критично: подсказки storage ДО загрузки SPA.
    await context.add_init_script(
        """(pairUnderscore, pairSlash) => {
          try {
            // подчистим мусор, который мог фиксировать BTC_USDT
            try { localStorage.clear(); } catch(e) {}
            try { sessionStorage.clear(); } catch(e) {}

            const pairs = [pairUnderscore, pairSlash];
            const keys = [
              'lastSymbol', 'lastsymbol',
              'symbol', 'Symbol',
              'pair', 'Pair',
              'market', 'Market',
              'instrument', 'Instrument',
              'selectedSymbol', 'selectedPair',
              'tv_symbol', 'tvSymbol', 'tv:symbol',
              'terminal:lastSymbol', 'terminal:lastPair'
            ];
            for (const k of keys) {
              for (const v of pairs) {
                try { localStorage.setItem(k, v); } catch(e) {}
                try { sessionStorage.setItem(k, v); } catch(e) {}
              }
            }
          } catch(e) {}
        }""",
        PAIR_URL_TOKEN,
        PAIR_TEXT,
    )

    page = await context.new_page()
    page.set_default_timeout(10_000)

    # Важно: сначала идём на основной URL, потом lock_pair всё выравнивает.
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
    await page.wait_for_timeout(900)
    await lock_pair(page)

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

                if time.monotonic() - last_heartbeat >= HEARTBEAT_SECONDS:
                    logger.info("Heartbeat: alive. seen=%d url=%s", len(seen), page.url)
                    last_heartbeat = time.monotonic()

                if time.monotonic() - last_reload >= RELOAD_EVERY_SECONDS:
                    logger.warning("Maintenance reload...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(900)
                    await lock_pair(page)
                    last_reload = time.monotonic()

                # На всякий случай: если SPA снова переключила пару — возвращаем обратно
                if not (_url_has_pair(page.url) and await _page_has_pair_text(page)):
                    await lock_pair(page)

                window = await scrape_trades(page)

                if not window:
                    logger.warning("No rows parsed. Reloading + lock_pair...")
                    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(900)
                    await lock_pair(page)
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
