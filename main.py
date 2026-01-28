import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx
from playwright.async_api import async_playwright, Page, TimeoutError as PwTimeoutError

# ───────────────────────── LOGGING ─────────────────────────

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("rapira_worker")

# ───────────────────────── CONFIG ─────────────────────────

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", os.getenv("SUPABASE_KEY", ""))  # allow both env names
SUPABASE_TABLE = os.getenv("SUPABASE_TABLE", "exchange_trades")

SOURCE = os.getenv("SOURCE", "rapira")
SYMBOL = os.getenv("SYMBOL", "USDT/RUB")

PAIR_SLUG = os.getenv("PAIR_SLUG", "USDT_RUB")
# ВАЖНО: использовать /exchange/..., без /ru/
RAPIRA_URL = os.getenv("RAPIRA_URL", f"https://rapira.net/exchange/{PAIR_SLUG}")

POLL_SECONDS = float(os.getenv("POLL_SECONDS", "1.5"))
NAV_TIMEOUT_MS = int(float(os.getenv("NAV_TIMEOUT_SECONDS", "45")) * 1000)
WAIT_TIMEOUT_MS = int(float(os.getenv("WAIT_TIMEOUT_SECONDS", "25")) * 1000)

LIMIT_ROWS = int(os.getenv("LIMIT_ROWS", "80"))  # сколько строк таблицы читать за проход
HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"

# Если у тебя в Supabase уникальный индекс по (source,symbol,trade_time,price,volume_usdt) — оставь так:
ON_CONFLICT = os.getenv("ON_CONFLICT", "source,symbol,trade_time,price,volume_usdt")

# ───────────────────────── HELPERS ─────────────────────────

PAIR_TEXT_REGEX = re.compile(r"USDT\s*/\s*RUB", re.IGNORECASE)

def _clean_num(s: str) -> str:
    """
    Приводит строку вида "1 039.31" / "1 039,31" / "1039.31" к "1039.31"
    """
    s = (s or "").strip()
    s = s.replace("\u00a0", " ").replace("\u202f", " ")  # NBSP / narrow NBSP
    s = s.replace(" ", "")
    # если десятичный разделитель запятая
    if s.count(",") == 1 and s.count(".") == 0:
        s = s.replace(",", ".")
    # оставляем цифры, точку и минус
    s = re.sub(r"[^0-9.\-]", "", s)
    return s

def to_decimal(s: str) -> Decimal:
    v = _clean_num(s)
    if v == "" or v == "-" or v == ".":
        raise InvalidOperation(f"empty numeric: {s!r}")
    return Decimal(v)

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

@dataclass(frozen=True)
class Trade:
    source: str
    symbol: str
    price: Decimal
    volume_usdt: Decimal
    volume_rub: Decimal
    trade_time: str  # "HH:MM:SS"

    def key(self) -> Tuple[str, str, str, str, str]:
        # строковый ключ под уникальность
        return (
            self.source,
            self.symbol,
            self.trade_time,
            format(self.price, "f"),
            format(self.volume_usdt, "f"),
        )

# ───────────────────────── SUPABASE ─────────────────────────

def _supabase_headers() -> Dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY not set")
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

async def supabase_upsert(trades: List[Trade]) -> None:
    if not trades:
        return
    url = f"{SUPABASE_URL}/rest/v1/{SUPABASE_TABLE}?on_conflict={httpx.QueryParams({'': ON_CONFLICT})['']}"
    payload = [
        {
            "source": t.source,
            "symbol": t.symbol,
            "price": str(t.price),
            "volume_usdt": str(t.volume_usdt),
            "volume_rub": str(t.volume_rub),
            "trade_time": t.trade_time,
        }
        for t in trades
    ]
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(url, headers=_supabase_headers(), content=json.dumps(payload))
        if r.status_code not in (200, 201, 204):
            raise RuntimeError(f"Supabase upsert failed: {r.status_code} {r.text}")
    log.info(f"Inserted (or merged) {len(trades)} rows into '{SUPABASE_TABLE}'.")

# ───────────────────────── PLAYWRIGHT: PAGE CONTROL ─────────────────────────

COOKIE_CONTAINER_CANDIDATES = [
    "div[role='dialog']:has-text('cookie')",
    "div[role='dialog']:has-text('cookies')",
    "div[role='dialog']:has-text('куки')",
    "div:has-text('cookies')",
    "div:has-text('куки')",
]
COOKIE_BUTTON_TEXTS = ["OK", "ОК", "Принять", "Accept", "Согласен"]

async def close_cookie_banner(page: Page) -> None:
    """
    Кликаем кнопку только внутри найденного cookie-баннера.
    Это защищает от клика по “OK” в другом модальном окне, который может менять пару.
    """
    for cont_sel in COOKIE_CONTAINER_CANDIDATES:
        cont = page.locator(cont_sel).first
        try:
            if await cont.count() == 0:
                continue
            if not await cont.is_visible(timeout=800):
                continue

            for txt in COOKIE_BUTTON_TEXTS:
                btn = cont.locator(f"button:has-text('{txt}')").first
                if await btn.count() and await btn.is_visible(timeout=800):
                    log.info("Found cookies banner, clicking 'OK'...")
                    await btn.click(timeout=1500)
                    await page.wait_for_timeout(400)
                    return
        except Exception:
            continue

async def ensure_trades_tab(page: Page) -> None:
    """
    У Rapira вкладки: "Стакан" и "Последние сделки".
    Нам нужна "Последние сделки".
    """
    # Самый стабильный — клик по тексту
    tab = page.locator("li:has-text('Последние сделки')").first
    try:
        if await tab.count() and await tab.is_visible(timeout=1500):
            await tab.click(timeout=2000)
            await page.wait_for_timeout(300)
            return
    except Exception:
        pass

    # Фоллбек: по id, если есть rp-*-trigger-history
    tab2 = page.locator("[id*='trigger-history']").first
    try:
        if await tab2.count() and await tab2.is_visible(timeout=1500):
            await tab2.click(timeout=2000)
            await page.wait_for_timeout(300)
    except Exception:
        pass

async def lock_pair_usdt_rub(page: Page) -> None:
    """
    Гарантируем, что мы реально на USDT_RUB (а не на BTC_USDT),
    проверяя и URL, и наличие текста "USDT/RUB" на странице.
    """
    for attempt in range(8):
        url = page.url or ""
        # проверка текста пары
        has_pair_text = False
        try:
            # любой элемент, где встречается USDT/RUB
            locator = page.locator("text=/USDT\\s*\\/\\s*RUB/i").first
            has_pair_text = await locator.is_visible(timeout=1500)
        except Exception:
            has_pair_text = False

        if (PAIR_SLUG in url) and has_pair_text:
            return

        log.warning(f"Wrong pair state. url={url} has_pair_text={has_pair_text}. Forcing goto {RAPIRA_URL} ...")
        await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
        await page.wait_for_timeout(800)
        await close_cookie_banner(page)
        await ensure_trades_tab(page)

    raise RuntimeError(f"Could not lock pair {PAIR_SLUG}. final_url={page.url}")

async def open_page(page: Page) -> None:
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
    await page.wait_for_timeout(800)
    await close_cookie_banner(page)
    await ensure_trades_tab(page)
    await lock_pair_usdt_rub(page)

# ───────────────────────── PARSING ─────────────────────────

def _row_selector() -> str:
    # максимально мягко (классы могут отличаться, но table-orders-row стабилен)
    return "tbody tr.table-orders-row"

async def wait_rows(page: Page) -> None:
    # ждём появление хотя бы одной строки
    await page.wait_for_selector(_row_selector(), timeout=WAIT_TIMEOUT_MS, state="visible")

async def parse_trades_from_dom(page: Page, limit: int) -> List[Trade]:
    """
    По твоей структуре:
      td[0] = курс
      td[1] = объем (USDT)
      td[2] = время
    """
    await wait_rows(page)

    rows = page.locator(_row_selector())
    n = min(await rows.count(), limit)
    trades: List[Trade] = []

    for i in range(n):
        row = rows.nth(i)
        tds = row.locator("td")
        if await tds.count() < 3:
            continue

        price_s = (await tds.nth(0).inner_text()).strip()
        vol_s = (await tds.nth(1).inner_text()).strip()
        time_s = (await tds.nth(2).inner_text()).strip()

        # Время должно быть HH:MM:SS
        if not re.match(r"^\d{2}:\d{2}:\d{2}$", time_s):
            continue

        try:
            price = to_decimal(price_s)
            vol_usdt = to_decimal(vol_s)
        except Exception:
            continue

        # Отсекаем очевидный мусор: если вдруг всё равно BTC_USDT пролезет (90000+)
        # Для USDT/RUB цена обычно двухзначная (70-120). Подстрой при необходимости.
        if price > Decimal("1000"):
            continue

        vol_rub = (price * vol_usdt)

        trades.append(
            Trade(
                source=SOURCE,
                symbol=SYMBOL,
                price=price,
                volume_usdt=vol_usdt,
                volume_rub=vol_rub,
                trade_time=time_s,
            )
        )

    return trades

# ───────────────────────── RUNTIME / LOOP ─────────────────────────

async def run_worker() -> None:
    seen: Set[Tuple[str, str, str, str, str]] = set()
    last_heartbeat = time.time()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS, args=["--no-sandbox"])
        context = await browser.new_context(
            viewport={"width": 1400, "height": 900},
            locale="ru-RU",
        )
        page = await context.new_page()

        log.info("Starting browser session...")
        await open_page(page)
        log.info(f"Opened {RAPIRA_URL} final_url={page.url}")

        while True:
            try:
                # жёстко контролируем, что мы на нужной паре
                await lock_pair_usdt_rub(page)

                # читаем сделки
                trades = await parse_trades_from_dom(page, LIMIT_ROWS)

                # берём только новые
                new_trades: List[Trade] = []
                for t in trades:
                    k = t.key()
                    if k in seen:
                        continue
                    seen.add(k)
                    new_trades.append(t)

                if new_trades:
                    newest = new_trades[0]
                    log.info(
                        f"Parsed {len(new_trades)} new trades. Newest: "
                        f'{{"price":"{newest.price}","volume_usdt":"{newest.volume_usdt}","trade_time":"{newest.trade_time}"}}'
                    )
                    await supabase_upsert(new_trades)

                # heartbeat
                if time.time() - last_heartbeat > 30:
                    last_heartbeat = time.time()
                    log.info(f"Heartbeat: alive. seen={len(seen)} url={page.url}")

                await asyncio.sleep(POLL_SECONDS)

            except PwTimeoutError as e:
                log.error(f"Timeout error: {e}. Reloading...")
                try:
                    await page.reload(wait_until="domcontentloaded", timeout=NAV_TIMEOUT_MS)
                    await page.wait_for_timeout(800)
                    await close_cookie_banner(page)
                    await ensure_trades_tab(page)
                except Exception:
                    pass
                await asyncio.sleep(2.0)

            except Exception as e:
                log.error(f"Worker error: {e}. Re-opening page...")
                try:
                    await open_page(page)
                except Exception as e2:
                    log.error(f"Re-open failed: {e2}")
                await asyncio.sleep(2.0)

async def main() -> None:
    # простая проверка env
    if not SUPABASE_URL or not SUPABASE_KEY:
        log.warning("SUPABASE_URL / SUPABASE_KEY not set. Worker will run but inserts will fail.")

    log.info(f"Config: RAPIRA_URL={RAPIRA_URL} PAIR_SLUG={PAIR_SLUG} SYMBOL={SYMBOL} HEADLESS={HEADLESS}")
    await run_worker()

if __name__ == "__main__":
    asyncio.run(main())
