import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)

# ───────────────────── ЛОГГЕР ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("three-exchanges")

# ───────────────────── НАСТРОЙКИ ────────────────────
POLL_SEC = float(os.getenv("POLL_SEC", "10"))
LIMIT = int(os.getenv("LIMIT", "20"))
HEADLESS = os.getenv("HEADLESS", "1") != "0"

# Proxy (опционально)
PROXY_SERVER = os.getenv("PROXY_SERVER")  # пример: http://host:port
PROXY_USERNAME = os.getenv("PROXY_USERNAME")
PROXY_PASSWORD = os.getenv("PROXY_PASSWORD")

# ABCEX креды
ABCEX_EMAIL = os.getenv("ABCEX_EMAIL")
ABCEX_PASSWORD = os.getenv("ABCEX_PASSWORD")

# URLs/маркет
GRINEX_MARKET = os.getenv("GRINEX_MARKET", "usdta7a5")
GRINEX_TRADE_URL = f"https://grinex.io/trading/{GRINEX_MARKET}?lang=ru"
GRINEX_API_URL = f"https://grinex.io/api/v2/trades?market={GRINEX_MARKET}&limit={LIMIT}&order_by=desc"

RAPIRA_SYMBOL = os.getenv("RAPIRA_SYMBOL", "USDT_RUB")
RAPIRA_URL = f"https://rapira.net/exchange/{RAPIRA_SYMBOL}"

ABCEX_SYMBOL = os.getenv("ABCEX_SYMBOL", "USDTRUB")
ABCEX_URL = f"https://abcex.io/client/spot/{ABCEX_SYMBOL}"

# Storage state для ABCEX
ABCEX_STATE_PATH = Path(os.getenv("ABCEX_STATE_PATH", "abcex_state.json"))

UA = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
)

# Таймауты
GOTO_TIMEOUT_MS = int(os.getenv("GOTO_TIMEOUT_MS", "60000"))
EXCHANGE_TIMEOUT_MS = int(os.getenv("EXCHANGE_TIMEOUT_MS", "60000"))


# ───────────────────── УТИЛИТЫ ──────────────────────
NBSP = "\u00A0"

PRICE_RE = re.compile(r"^\d{1,6}([.,]\d+)?$")         # 79.91
QTY_RE = re.compile(r"^\d{1,3}([ \u00A0]\d{3})*([.,]\d+)?$")  # 12 717.65
TIME_RE = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")     # 12:34 или 12:34:56


def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def compact_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace(NBSP, " ")).strip()


def parse_float(s: str) -> Optional[float]:
    """
    Преобразование строк вида:
    "12 717.65", "12\u00A0717,65", "79.90" -> float
    """
    if s is None:
        return None
    s = compact_spaces(str(s))
    if not s:
        return None
    # убираем пробелы-разделители тысяч
    s = s.replace(" ", "")
    # меняем запятую на точку
    s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def looks_like_time(s: str) -> bool:
    s = compact_spaces(s)
    return bool(TIME_RE.match(s))


def extract_first_time(cells: List[str]) -> Optional[str]:
    for c in cells:
        if looks_like_time(c):
            return compact_spaces(c)
    return None


def score_price(v: float) -> int:
    # для USDT/RUB логично ожидать десятки/сотни
    if 1 <= v <= 10000:
        return 2
    if 0 < v < 1:
        return 1
    return 0


def choose_price_qty_total(nums: List[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Эвристика:
    - выбираем тройку (price, qty, total), где total ≈ price*qty.
    - если total отсутствует — вернем (price, qty, None) с минимальной ошибкой по имеющимся.
    """
    if len(nums) < 2:
        return None, None, None

    best = (None, None, None)
    best_err = float("inf")

    # перебор candidates
    for i, p in enumerate(nums):
        if score_price(p) == 0:
            continue
        for j, q in enumerate(nums):
            if j == i:
                continue
            # qty обычно не сверх-малый
            if q <= 0:
                continue
            # пытаемся найти total
            for k, t in enumerate(nums):
                if k == i or k == j:
                    continue
                if t <= 0:
                    continue
                err = abs((p * q) - t)
                # нормируем ошибку
                denom = max(1.0, t)
                rel = err / denom
                if rel < best_err:
                    best_err = rel
                    best = (p, q, t)

    if best[0] is not None:
        return best

    # если total не нашли, просто берем (price, qty)
    # price = наиболее "похожий" на цену
    price = None
    for v in nums:
        if score_price(v) > 0:
            if price is None or abs(v - 80) < abs(price - 80):
                price = v
    if price is None:
        return None, None, None

    # qty = самый крупный из оставшихся
    rest = [v for v in nums if v != price]
    if not rest:
        return price, None, None
    qty = max(rest)
    return price, qty, None


def compute_metrics(trades: List[Dict[str, Any]], quote_ccy: str) -> Dict[str, Any]:
    sum_qty = 0.0
    turnover = 0.0
    pv = 0.0
    for t in trades:
        p = t.get("price")
        q = t.get("qty")
        if isinstance(p, (int, float)) and isinstance(q, (int, float)):
            sum_qty += float(q)
            turnover += float(p) * float(q)
            pv += float(p) * float(q)

    vwap = (pv / sum_qty) if sum_qty > 0 else None
    return {
        "count": len(trades),
        "sum_qty_usdt": sum_qty,
        "turnover_quote": turnover,
        "vwap": vwap,
        "quote_ccy": quote_ccy,
        "trades": trades,
    }


def ensure_playwright_chromium() -> None:
    """
    Для Render часто нужно явно поставить браузер.
    Если уже стоит — команда пройдет быстро.
    """
    env = os.environ.copy()
    # чтобы браузеры ставились в проект (удобнее на Render)
    env.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(Path(".cache/ms-playwright").resolve()))
    log.info("Ensuring Playwright Chromium is installed ... (PLAYWRIGHT_BROWSERS_PATH=%s)", env["PLAYWRIGHT_BROWSERS_PATH"])
    subprocess.run(
        ["python", "-m", "playwright", "install", "chromium"],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log.info("Playwright Chromium is installed (or already present).")


def proxy_settings() -> Optional[Dict[str, Any]]:
    if not PROXY_SERVER:
        return None
    p: Dict[str, Any] = {"server": PROXY_SERVER}
    if PROXY_USERNAME:
        p["username"] = PROXY_USERNAME
    if PROXY_PASSWORD:
        p["password"] = PROXY_PASSWORD
    return p


# ───────────────────── GRINEX ────────────────────────
async def grinex_fetch(context: BrowserContext, limit: int) -> Dict[str, Any]:
    logger = logging.getLogger("grinex")

    # 1) прогрев страницы, чтобы получить cookies/clearance в контексте
    page = await context.new_page()
    await page.set_extra_http_headers({"User-Agent": UA})
    await page.goto(GRINEX_TRADE_URL, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)
    await page.wait_for_timeout(1200)

    # 2) запрос API из context.request (в этом же контексте)
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Referer": page.url,
        "User-Agent": UA,
    }

    async def request_once() -> str:
        resp = await context.request.get(
            GRINEX_API_URL.replace(f"limit={LIMIT}", f"limit={limit}"),
            headers=headers,
            timeout=EXCHANGE_TIMEOUT_MS,
        )
        text = await resp.text()
        logger.info("HTTP %s %s", resp.status, GRINEX_API_URL)
        return text

    text = await request_once()
    if "<!DOCTYPE html" in text or "<html" in text.lower():
        logger.warning("Grinex returned HTML; reloading trading page and retrying once ...")
        await page.reload(wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)
        await page.wait_for_timeout(1500)
        text = await request_once()

    if "<!DOCTYPE html" in text or "<html" in text.lower():
        raise RuntimeError(f"Grinex API still HTML/challenge. First 200 chars: {text[:200]!r}")

    data = json.loads(text)

    # Нормализация trades (в Grinex формат может отличаться — делаем устойчиво)
    raw_trades = data
    if isinstance(data, dict):
        # возможные ключи
        for k in ("trades", "data", "result"):
            if k in data and isinstance(data[k], list):
                raw_trades = data[k]
                break

    if not isinstance(raw_trades, list):
        raise RuntimeError(f"Unexpected Grinex response shape: {type(raw_trades)}")

    trades: List[Dict[str, Any]] = []
    for it in raw_trades[:limit]:
        if not isinstance(it, dict):
            continue
        price = parse_float(it.get("price") or it.get("p"))
        qty = parse_float(it.get("amount") or it.get("qty") or it.get("q") or it.get("volume"))
        ts = it.get("created_at") or it.get("time") or it.get("ts") or it.get("timestamp")
        time_str = None
        if isinstance(ts, (int, float)):
            # seconds or ms
            tsv = float(ts)
            if tsv > 1e12:
                tsv /= 1000.0
            time_str = datetime.fromtimestamp(tsv, tz=timezone.utc).strftime("%H:%M:%S")
        elif isinstance(ts, str):
            # iso or already time
            s = ts.strip()
            if looks_like_time(s):
                time_str = s
            else:
                # попробуем iso
                try:
                    # минимальный парсинг
                    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                    time_str = dt.astimezone(timezone.utc).strftime("%H:%M:%S")
                except Exception:
                    time_str = s[:8]

        if price is None or qty is None:
            continue

        trades.append(
            {
                "price": float(price),
                "qty": float(qty),
                "time": time_str,
                "price_raw": it.get("price"),
                "qty_raw": it.get("amount") or it.get("qty") or it.get("volume"),
            }
        )

    payload = {
        "exchange": "grinex",
        "symbol": "USDT/A7A5",  # в логах у тебя usdta7a5
        "url": GRINEX_API_URL,
        **compute_metrics(trades, quote_ccy="A7A5"),
    }
    await page.close()
    return payload


# ───────────────────── RAPIRA ────────────────────────
async def rapira_accept_cookies(page: Page) -> None:
    # разные варианты текста
    candidates = ["text=Я согласен", "text=Принять", "text=Согласен", "text=Accept"]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.count():
                await loc.click(timeout=2000)
                return
        except Exception:
            pass


async def rapira_click_last_trades(page: Page) -> None:
    candidates = [
        "text=Последние сделки",
        "text=Сделки",
        "role=tab[name='Последние сделки']",
        "role=tab[name='Сделки']",
    ]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.count():
                await loc.click(timeout=5000)
                return
        except Exception:
            pass


async def rapira_fetch(context: BrowserContext, limit: int) -> Dict[str, Any]:
    logger = logging.getLogger("rapira")
    page = await context.new_page()
    await page.set_extra_http_headers({"User-Agent": UA})

    logger.info("Opening Rapira page %s ...", RAPIRA_URL)
    await page.goto(RAPIRA_URL, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)

    # cookies
    await rapira_accept_cookies(page)

    # вкладка "Последние сделки"
    await rapira_click_last_trades(page)

    rows_sel = "div.table-responsive.table-orders table.table-row-dashed tbody tr.table-orders-row"
    await page.wait_for_selector(rows_sel, timeout=EXCHANGE_TIMEOUT_MS)

    rows = page.locator(rows_sel)
    rc = await rows.count()
    logger.info("Found %d rows on Rapira.", rc)

    trades: List[Dict[str, Any]] = []
    for i in range(min(rc, limit)):
        row = rows.nth(i)
        cells = row.locator("td")
        cc = await cells.count()
        texts: List[str] = []
        for j in range(min(cc, 8)):
            try:
                texts.append(compact_spaces(await cells.nth(j).inner_text()))
            except Exception:
                texts.append("")

        # time: если есть нормальный формат
        time_str = extract_first_time(texts)

        # числовые значения (кроме явного времени)
        nums = []
        nums_raw = []
        for t in texts:
            if looks_like_time(t):
                continue
            v = parse_float(t)
            if v is not None:
                nums.append(v)
                nums_raw.append(t)

        price, qty, total = choose_price_qty_total(nums)

        # если вдруг qty и price перепутались — поправим по диапазонам
        if price is not None and qty is not None and price > qty and price < 10000 and qty < 100:
            # чаще qty меньше price? для USDT qty может быть и меньше, но это редкость.
            pass

        if price is None or qty is None:
            continue

        trades.append(
            {
                "price": float(price),
                "qty": float(qty),
                "time": time_str,
                "price_raw": None,
                "qty_raw": None,
            }
        )

    payload = {
        "exchange": "rapira",
        "symbol": "USDT/RUB",
        "url": RAPIRA_URL,
        **compute_metrics(trades, quote_ccy="RUB"),
    }
    await page.close()
    return payload


# ───────────────────── ABCEX ─────────────────────────
async def abcex_handle_cookies(page: Page) -> None:
    candidates = ["text=Принять", "text=Я согласен", "text=Accept", "text=Согласен"]
    for sel in candidates:
        try:
            loc = page.locator(sel).first
            if await loc.count():
                await loc.click(timeout=3000)
                return
        except Exception:
            pass


async def abcex_login_if_needed(context: BrowserContext, page: Page) -> None:
    logger = logging.getLogger("abcex")

    # Делаем попытку понять, видим ли мы форму логина
    # (у ABCEХ верстка может меняться — используем набор селекторов)
    email_sel = "input[type='email'], input[name='email'], input[placeholder*='mail' i]"
    pass_sel = "input[type='password'], input[name='password'], input[placeholder*='парол' i], input[placeholder*='password' i]"
    btn_sel = "button[type='submit'], button:has-text('Войти'), button:has-text('Sign in'), button:has-text('Login')"

    async def has_login_form() -> bool:
        try:
            return (await page.locator(pass_sel).count()) > 0
        except Exception:
            return False

    if not await has_login_form():
        return

    if not ABCEX_EMAIL or not ABCEX_PASSWORD:
        raise RuntimeError("ABCEX requires login, but ABCEX_EMAIL/ABCEX_PASSWORD env vars are not set.")

    logger.info("Login detected. Performing sign-in ...")

    # иногда email поле скрыто/появляется после клика
    await page.wait_for_timeout(500)

    # fill email (если есть)
    if await page.locator(email_sel).count():
        await page.locator(email_sel).first.fill(ABCEX_EMAIL, timeout=5000)

    # fill password
    if await page.locator(pass_sel).count():
        await page.locator(pass_sel).first.fill(ABCEX_PASSWORD, timeout=5000)
    else:
        raise RuntimeError("Не смог найти поле password на ABCEX.")

    # submit
    if await page.locator(btn_sel).count():
        await page.locator(btn_sel).first.click(timeout=5000)
    else:
        # fallback: Enter
        await page.keyboard.press("Enter")

    # успех: исчезло password поле или url изменился
    try:
        await page.wait_for_timeout(1000)
        await page.wait_for_function(
            """() => {
                const p = document.querySelector("input[type='password']");
                return !p || p.value === "";
            }""",
            timeout=EXCHANGE_TIMEOUT_MS,
        )
        logger.info("Login successful (password disappeared).")
    except Exception:
        # не обязательно плохо — могло перейти без исчезновения
        logger.warning("Login check inconclusive; continuing.")

    # сохранить state
    try:
        await context.storage_state(path=str(ABCEX_STATE_PATH))
        logger.info("Saved session state to %s", ABCEX_STATE_PATH)
    except Exception as e:
        logger.warning("Failed to save storage state: %s", e)


async def abcex_find_trades_table(page: Page) -> Any:
    """
    Автодетект таблицы сделок:
    ищем table, где в строках есть признаки time/price/qty.
    """
    logger = logging.getLogger("abcex")

    tables = page.locator("table")
    n = await tables.count()
    best_tbl = None
    best_score = -1

    for i in range(n):
        tbl = tables.nth(i)
        rows = tbl.locator("tbody tr")
        rc = await rows.count()
        if rc <= 0:
            continue

        score = 0
        # анализируем первые 10 строк
        for r in range(min(rc, 10)):
            row = rows.nth(r)
            cells = row.locator("td")
            cc = await cells.count()
            if cc < 3:
                continue
            vals = []
            for k in range(min(cc, 8)):
                try:
                    vals.append(compact_spaces(await cells.nth(k).inner_text()))
                except Exception:
                    vals.append("")
            if any(looks_like_time(v) for v in vals):
                score += 1
            # цена — маленькое число с десятичной частью
            if any(parse_float(v) is not None and score_price(parse_float(v) or 0) > 0 for v in vals):
                score += 1
            # количество — обычно с тысячами/десятичными
            if any(parse_float(v) is not None and (parse_float(v) or 0) > 0 for v in vals):
                score += 1

        if score > best_score:
            best_score = score
            best_tbl = tbl

    logger.info("Trades table autodetect: tables=%d best_score=%d", n, best_score)

    if best_tbl is None or best_score < 5:
        # debug dump
        await page.screenshot(path="abcex_no_table.png", full_page=True)
        html = await page.content()
        Path("abcex_no_table.html").write_text(html, encoding="utf-8")
        raise RuntimeError("Не смог автодетектить таблицу сделок ABCEX. См. abcex_no_table.*")

    return best_tbl


async def abcex_fetch(play: Playwright, browser: Browser, limit: int) -> Dict[str, Any]:
    logger = logging.getLogger("abcex")

    # отдельный контекст, чтобы хранить state
    context_kwargs: Dict[str, Any] = {
        "user_agent": UA,
        "viewport": {"width": 1280, "height": 720},
    }
    p = proxy_settings()
    if p:
        context_kwargs["proxy"] = p

    if ABCEX_STATE_PATH.exists():
        logger.info("Using saved session state: %s", ABCEX_STATE_PATH)
        context_kwargs["storage_state"] = str(ABCEX_STATE_PATH)

    context = await browser.new_context(**context_kwargs)
    page = await context.new_page()

    logger.info("Opening ABCEX: %s", ABCEX_URL)
    await page.goto(ABCEX_URL, wait_until="domcontentloaded", timeout=GOTO_TIMEOUT_MS)

    await abcex_handle_cookies(page)
    await abcex_login_if_needed(context, page)

    # Даем UI догрузиться
    await page.wait_for_timeout(1500)

    # Автодетект таблицы
    tbl = await abcex_find_trades_table(page)

    # Парсим строки
    rows = tbl.locator("tbody tr")
    rc = await rows.count()

    trades: List[Dict[str, Any]] = []
    for i in range(min(rc, limit)):
        row = rows.nth(i)
        cells = row.locator("td")
        cc = await cells.count()
        texts: List[str] = []
        for j in range(min(cc, 10)):
            try:
                texts.append(compact_spaces(await cells.nth(j).inner_text()))
            except Exception:
                texts.append("")

        time_str = extract_first_time(texts)

        nums = []
        for t in texts:
            if looks_like_time(t):
                continue
            v = parse_float(t)
            if v is not None:
                nums.append(v)

        price, qty, total = choose_price_qty_total(nums)
        if price is None or qty is None:
            continue

        trades.append(
            {
                "price": float(price),
                "qty": float(qty),
                "time": time_str,
                "price_raw": None,
                "qty_raw": None,
            }
        )

    payload = {
        "exchange": "abcex",
        "symbol": "USDT/RUB",
        "url": ABCEX_URL,
        **compute_metrics(trades, quote_ccy="RUB"),
    }

    await context.close()
    return payload


# ───────────────────── ОРКЕСТРАЦИЯ ───────────────────
@dataclass
class Result:
    ok: bool
    error: Optional[str]
    data: Dict[str, Any]


async def run_once(play: Playwright, browser: Browser, limit: int) -> Dict[str, Any]:
    ts = utc_now_str()

    # общий контекст для grinex+rapira (быстрее), abcex отдельно из-за state
    ctx_kwargs: Dict[str, Any] = {
        "user_agent": UA,
        "viewport": {"width": 1280, "height": 720},
    }
    p = proxy_settings()
    if p:
        ctx_kwargs["proxy"] = p

    context = await browser.new_context(**ctx_kwargs)

    async def wrap(name: str, coro):
        try:
            data = await asyncio.wait_for(coro, timeout=EXCHANGE_TIMEOUT_MS / 1000)
            return Result(ok=True, error=None, data=data)
        except Exception as e:
            return Result(ok=False, error=f"{type(e).__name__}: {e}", data={"exchange": name})

    grinex_task = wrap("grinex", grinex_fetch(context, limit))
    rapira_task = wrap("rapira", rapira_fetch(context, limit))
    abcex_task = wrap("abcex", abcex_fetch(play, browser, limit))

    results = await asyncio.gather(grinex_task, rapira_task, abcex_task)

    await context.close()

    # compare только по успешным
    ok_payloads = [r.data for r in results if r.ok and isinstance(r.data, dict)]
    by_turnover = sorted(
        [
            {
                "exchange": p.get("exchange"),
                "symbol": p.get("symbol"),
                "count": p.get("count"),
                "sum_qty_usdt": p.get("sum_qty_usdt"),
                "turnover_quote": p.get("turnover_quote"),
                "quote_ccy": p.get("quote_ccy"),
                "vwap": p.get("vwap"),
            }
            for p in ok_payloads
        ],
        key=lambda x: float(x.get("turnover_quote") or 0.0),
        reverse=True,
    )
    by_qty = sorted(
        [
            {
                "exchange": p.get("exchange"),
                "symbol": p.get("symbol"),
                "count": p.get("count"),
                "sum_qty_usdt": p.get("sum_qty_usdt"),
                "turnover_quote": p.get("turnover_quote"),
                "quote_ccy": p.get("quote_ccy"),
                "vwap": p.get("vwap"),
            }
            for p in ok_payloads
        ],
        key=lambda x: float(x.get("sum_qty_usdt") or 0.0),
        reverse=True,
    )

    out = {
        "ts_utc": ts,
        "poll_sec": POLL_SEC,
        "limit": limit,
        "results": [r.__dict__ for r in results],
        "compare": {
            "by_turnover_quote_desc": by_turnover,
            "by_sum_qty_usdt_desc": by_qty,
        },
    }
    return out


async def main() -> None:
    log.info("Starting unified 3-exchange scraper ...")
    ensure_playwright_chromium()

    async with async_playwright() as play:
        launch_kwargs: Dict[str, Any] = {"headless": HEADLESS}
        p = proxy_settings()
        if p:
            launch_kwargs["proxy"] = p

        browser = await play.chromium.launch(**launch_kwargs)

        while True:
            start = datetime.now(timezone.utc).timestamp()
            payload = await run_once(play, browser, LIMIT)
            print(json.dumps(payload, ensure_ascii=False))
            # sleep остаток периода
            elapsed = datetime.now(timezone.utc).timestamp() - start
            to_sleep = max(0.0, POLL_SEC - elapsed)
            await asyncio.sleep(to_sleep)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
