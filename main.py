# ───────────────────── IMPORTS ──────────────────────
import asyncio
import html
import logging
import os
import subprocess
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import httpx
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from supabase import Client, create_client
from telegram.ext import ApplicationBuilder, CommandHandler

# ───────────────────── CONFIG ───────────────────────
TOKEN = os.getenv("TG_BOT_TOKEN", "7128150617:AAHEMrzGrSOZrLAMYDf8F8MwklSvPDN2IVk")
PASSWORD = os.getenv("TG_BOT_PASS", "7128150617")
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://jetfadpysjsvtqdgnsjp.supabase.co")
SUPABASE_KEY = os.getenv(
    "SUPABASE_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImpldGZhZHB5c2pzdnRxZGduc2pwIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTAxNjY1OTEsImV4cCI6MjA2NTc0MjU5MX0."
    "WNUax6bkFNW8NMWKxpRQ9SIFE_M2BaTxcNt2eevQT34",
)

CHAT_ID = "@KaliningradCryptoKenigSwap"
KALININGRAD_TZ = timezone(timedelta(hours=2))

KENIG_ASK_OFFSET = 1.0  # +к продаже
KENIG_BID_OFFSET = -0.5  # +к покупке

MAX_RETRIES = 3
RETRY_DELAY = 5

AUTHORIZED_USERS: set[int] = set()

# ───────────────────── LOGGER ───────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ────────────────── SUPABASE ────────────────────────
sb: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


async def upsert_rate(source: str, sell: float, buy: float) -> None:
    """Пишем/обновляем запись в таблице kenig_rates (по полю source)."""
    payload = {
        "source": source,
            "sell": round(sell, 2),
        "buy": round(buy, 2),
        "updated_at": datetime.utcnow().isoformat(),
    }
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: sb.table("kenig_rates").upsert(payload, on_conflict="source").execute(),
        )
        logger.info("Supabase upsert OK: %s", source)
    except Exception as e:
        logger.warning("Supabase upsert failed (%s): %s", source, e)


# ────────────────── PLAYWRIGHT SETUP ────────────────
def install_chromium_for_playwright() -> None:
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
    except Exception as exc:
        logger.warning("Playwright install error: %s", exc)


# ───────────────────── SCRAPERS ─────────────────────
GRINEX_URL = "https://grinex.io/trading/usdta7a5?lang=en"
TIMEOUT_MS = 30_000


def _parse_number(text: str) -> Optional[float]:
    """
    Приводит '79.30', '25 033.8898', '1 985 187,46' и т.п. к float.
    """
    if text is None:
        return None
    t = text.replace("\xa0", " ").replace(" ", "")
    t = t.replace(",", ".")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        logger.warning("Cannot parse number from %r", text)
        return None


async def fetch_grinex_rate() -> Tuple[Optional[float], Optional[float]]:
    """
    Старый рабочий парсер стакана Grinex (Ask/Bid).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"
                    )
                )
                page = await context.new_page()
                await page.goto(GRINEX_URL, wait_until="domcontentloaded", timeout=TIMEOUT_MS)

                # закрыть возможный cookie-баннер
                try:
                    btn = page.locator("button:has-text('Accept'), button:has-text('Я согласен')")
                    if await btn.count() > 0:
                        await btn.first.click(timeout=3_000)
                except Exception:
                    pass

                ask_sel = "tbody.usdta7a5_ask.asks tr[data-price]"
                bid_sel = "tbody.usdta7a5_bid.bids tr[data-price]"

                await page.wait_for_selector(ask_sel, timeout=TIMEOUT_MS)
                await page.wait_for_selector(bid_sel, timeout=TIMEOUT_MS)

                ask_node = await page.query_selector(ask_sel)
                bid_node = await page.query_selector(bid_sel)

                ask_attr = await ask_node.get_attribute("data-price") if ask_node else None
                bid_attr = await bid_node.get_attribute("data-price") if bid_node else None

                ask = float(ask_attr) if ask_attr else None
                bid = float(bid_attr) if bid_attr else None

                await browser.close()
                logger.info("Grinex orderbook OK: ask=%s bid=%s", ask, bid)
                return ask, bid
        except Exception as e:
            logger.warning("Grinex orderbook attempt %s/%s failed: %s", attempt, MAX_RETRIES, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    return None, None


async def fetch_grinex_last_trades(limit: int = 50) -> list[dict]:
    """
    Новый парсер: последние сделки с Grinex из блока trade_history_panel.

    Возвращает список словарей:
    {
        "price": float | None,
        "price_raw": str,
        "volume_usdt": float | None,
        "volume_usdt_raw": str,
        "volume_a7a5": float | None,
        "volume_a7a5_raw": str,
        "time": str,
    }
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/123.0.0.0 Safari/537.36"
                    )
                )
                page = await context.new_page()
                await page.goto(
                    GRINEX_URL,
                    wait_until="domcontentloaded",
                    timeout=TIMEOUT_MS,
                )

                # закрыть возможный cookie-баннер
                try:
                    btn = page.locator("button:has-text('Accept'), button:has-text('Я согласен')")
                    if await btn.count() > 0:
                        await btn.first.click(timeout=3_000)
                except Exception:
                    pass

                # капча — не обходим, просто логируем и выходим
                try:
                    content = await page.content()
                    if "sp_rotated_captcha" in content:
                        logger.warning(
                            "Grinex responded with rotated captcha page, no trades scraped."
                        )
                        await browser.close()
                        return []
                except Exception:
                    pass

                # клик по вкладке "Последние сделки" (на всякий случай)
                try:
                    history_tab = page.locator(
                        "a[href='#tab_trade_history_all'], "
                        "a[data-target='#tab_trade_history_all']"
                    )
                    if await history_tab.count() > 0:
                        await history_tab.first.click(timeout=5_000)
                except Exception:
                    pass

                # даём фронту дорисовать таблицу
                await page.wait_for_timeout(2_000)

                rows = await page.query_selector_all(
                    "div.trade_history_panel "
                    "table.all-trades.usdta7a5 tbody tr[id^='market-trade-']"
                )

                if not rows:
                    raise RuntimeError("No trade rows found in trade_history_panel")

                trades: list[dict] = []

                for row in rows[:limit]:
                    try:
                        # ---- цена ----
                        price_el = await row.query_selector("td.price .visible-lg-inline") \
                                   or await row.query_selector("td.price")
                        price_raw = (await price_el.inner_text()).strip() if price_el else ""
                        price = _parse_number(price_raw)

                        # ---- объёмы ----
                        volume_tds = await row.query_selector_all("td.volume")
                        volume_usdt_raw = ""
                        volume_a7a5_raw = ""
                        volume_usdt = None
                        volume_a7a5 = None

                        if len(volume_tds) >= 1:
                            volume_usdt_raw = (await volume_tds[0].inner_text()).strip()
                            volume_usdt = _parse_number(volume_usdt_raw)

                        if len(volume_tds) >= 2:
                            volume_a7a5_raw = (await volume_tds[1].inner_text()).strip()
                            volume_a7a5 = _parse_number(volume_a7a5_raw)

                        # ---- время ----
                        time_el = await row.query_selector("td.time")
                        time_raw = ""
                        if time_el:
                            t_raw = (await time_el.inner_text()).strip()
                            parts = [p.strip() for p in t_raw.splitlines() if p.strip()]
                            time_raw = " ".join(parts)

                        trades.append(
                            {
                                "price": price,
                                "price_raw": price_raw,
                                "volume_usdt": volume_usdt,
                                "volume_usdt_raw": volume_usdt_raw,
                                "volume_a7a5": volume_a7a5,
                                "volume_a7a5_raw": volume_a7a5_raw,
                                "time": time_raw,
                            }
                        )
                    except Exception as row_err:
                        logger.warning("Grinex trade row parse error: %s", row_err)
                        continue

                await browser.close()

                # ── ЛОГИРУЕМ СДЕЛКИ ──────────────────────
                logger.info("Fetched %d Grinex trades", len(trades))
                for t in trades[:10]:
                    logger.info(
                        "Grinex trade: time=%s price=%s RUB vol_usdt=%s vol_a7a5=%s",
                        t.get("time"),
                        t.get("price_raw"),
                        t.get("volume_usdt_raw"),
                        t.get("volume_a7a5_raw"),
                    )

                return trades

        except Exception as e:
            logger.warning(
                "Grinex last-trades attempt %s/%s failed: %s",
                attempt,
                MAX_RETRIES,
                e,
            )
            if attempt < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)

    return []


async def fetch_bestchange_sell() -> Optional[float]:
    url = "https://www.bestchange.com/cash-ruble-to-tether-trc20-in-klng.html"
    for a in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient() as c:
                res = await c.get(url, timeout=15)
                soup = BeautifulSoup(res.text, "html.parser")
                div = soup.find("div", class_="fs")
                if div:
                    return float(
                        "".join(ch for ch in div.text if ch.isdigit() or ch in ",.").replace(",", ".")
                    )
        except Exception as e:
            logger.warning("BestChange sell attempt %s/%s: %s", a, MAX_RETRIES, e)
            if a < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    return None


async def fetch_bestchange_buy() -> Optional[float]:
    url = "https://www.bestchange.com/tether-trc20-to-cash-ruble-in-klng.html"
    for a in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as c:
                res = await c.get(url, timeout=15)
                soup = BeautifulSoup(res.text, "html.parser")
                table = soup.find("table", id="content_table")
                row = table.find("tr", onclick=True)
                price_td = next(
                    (td for td in row.find_all("td", class_="bi") if "RUB Cash" in td.text), None
                )
                if price_td:
                    return float(
                        "".join(ch for ch in price_td.text if ch.isdigit() or ch in ",.").replace(
                            ",", "."
                        )
                    )
        except Exception as e:
            logger.warning("BestChange buy attempt %s/%s: %s", a, MAX_RETRIES, e)
            if a < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    return None


async def fetch_energo() -> Tuple[Optional[float], Optional[float], Optional[float]]:
    url = "https://ru.myfin.by/bank/energotransbank/currency/kaliningrad"
    for a in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as c:
                res = await c.get(url, timeout=15)
                soup = BeautifulSoup(res.text, "html.parser")
                table = soup.find("table", class_="table-best white_bg")
                usd_td = table.find("td", class_="title")
                buy_td = usd_td.find_next("td")
                sell_td = buy_td.find_next("td")
                cbr_td = sell_td.find_next("td")
                return (
                    float(sell_td.text.replace(",", ".")),
                    float(buy_td.text.replace(",", ".")),
                    float(cbr_td.text.replace(",", ".")),
                )
        except Exception as e:
            logger.warning("Energo attempt %s/%s: %s", a, MAX_RETRIES, e)
            if a < MAX_RETRIES:
                await asyncio.sleep(RETRY_DELAY)
    return None, None, None


# ───────────────── TELEGRAM HANDLERS ────────────────
def is_authorized(uid: int) -> bool:
    return uid in AUTHORIZED_USERS


async def auth(update, context):
    if len(context.args) != 1:
        await update.message.reply_text("Введите пароль: /auth <пароль>")
        return
    if context.args[0] == PASSWORD:
        AUTHORIZED_USERS.add(update.effective_user.id)
        await update.message.reply_text("Доступ разрешён.")
    else:
        await update.message.reply_text("Неверный пароль.")


async def start(update, context):
    await update.message.reply_text("Бот активен. Используйте /auth <пароль>.")


async def help_command(update, context):
    await update.message.reply_text("/start /auth /check /change /show_offsets /trades /help")


async def check(update, context):
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Нет доступа. /auth <пароль>")
        return
    await send_rates_message(context.application)
    await update.message.reply_text("Курсы отправлены.")


async def change_offsets(update, context):
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Нет доступа.")
        return
    try:
        global KENIG_ASK_OFFSET, KENIG_BID_OFFSET
        KENIG_ASK_OFFSET, KENIG_BID_OFFSET = map(float, context.args[:2])
        await update.message.reply_text(f"Ask +{KENIG_ASK_OFFSET}  Bid {KENIG_BID_OFFSET}")
    except Exception:
        await update.message.reply_text("Пример: /change 1.0 -0.5")


async def show_offsets(update, context):
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Нет доступа.")
        return
    await update.message.reply_text(f"Ask +{KENIG_ASK_OFFSET}  Bid {KENIG_BID_OFFSET}")


async def trades_command(update, context):
    """
    /trades — показать последние сделки Grinex USDT/A7A5 (через бота).
    """
    if not is_authorized(update.effective_user.id):
        await update.message.reply_text("Нет доступа. /auth <пароль>")
        return

    trades = await fetch_grinex_last_trades(limit=10)
    if not trades:
        await update.message.reply_text("Нет данных по последним сделкам Grinex.")
        return

    lines: list[str] = ["Последние сделки Grinex USDT/A7A5", ""]
    for t in trades:
        time_str = t.get("time") or "—"
        price_raw = t.get("price_raw") or "?"
        vol_usdt_raw = t.get("volume_usdt_raw") or "?"
        vol_a7a5_raw = t.get("volume_a7a5_raw") or "?"
        lines.append(f"{time_str}  {price_raw} RUB  {vol_usdt_raw} USDT  {vol_a7a5_raw} A7A5")

    msg = "<pre>" + html.escape("\n".join(lines)) + "</pre>"
    await update.message.reply_text(msg, parse_mode="HTML")


# ───────────────── SEND RATES MSG ───────────────────
async def send_rates_message(app):
    bc_sell = await fetch_bestchange_sell()
    bc_buy = await fetch_bestchange_buy()
    en_sell, en_buy, en_cbr = await fetch_energo()
    gr_ask, gr_bid = await fetch_grinex_rate()

    # ── НОВОЕ: логируем последние сделки в каждом цикле ──
    trades = await fetch_grinex_last_trades(limit=10)
    if trades:
        logger.info("Grinex last trades snapshot (first 10):")
        for t in trades:
            logger.info(
                "Trade: time=%s price=%s RUB vol_usdt=%s vol_a7a5=%s",
                t.get("time"),
                t.get("price_raw"),
                t.get("volume_usdt_raw"),
                t.get("volume_a7a5_raw"),
            )
    else:
        logger.warning("No Grinex last trades in current cycle.")

    ts = datetime.now(KALININGRAD_TZ).strftime("%d.%m.%Y %H:%M:%S")
    lines = [ts, ""]

    # KenigSwap
    lines += ["KenigSwap rate USDT/RUB"]
    if gr_ask and gr_bid:
        lines.append(
            f"Продажа: {gr_ask + KENIG_ASK_OFFSET:.2f} ₽, "
            f"Покупка: {gr_bid + KENIG_BID_OFFSET:.2f} ₽"
        )
    else:
        lines.append("Нет данных с Grinex.")
    lines.append("")

    # BestChange
    lines += ["BestChange rate USDT/RUB"]
    if bc_sell and bc_buy:
        lines.append(f"Продажа: {bc_sell:.2f} ₽, Покупка: {bc_buy:.2f} ₽")
    else:
        lines.append("Нет данных с BestChange.")
    lines.append("")

    # Energo
    lines += ["EnergoTransBank rate USD/RUB"]
    if en_sell and en_buy and en_cbr:
        lines.append(
            f"Продажа: {en_sell:.2f} ₽, "
            f"Покупка: {en_buy:.2f} ₽, ЦБ: {en_cbr:.2f} ₽"
        )
    else:
        lines.append("Нет данных с EnergoTransBank.")

    msg = "<pre>" + html.escape("\n".join(lines)) + "</pre>"

    # Telegram
    try:
        await app.bot.send_message(
            chat_id=CHAT_ID,
            text=msg,
            parse_mode="HTML",
            disable_web_page_preview=True,
        )
    except Exception as e:
        logger.error("Send error: %s", e)

    # Supabase (параллельно)
    tasks = []
    if gr_ask and gr_bid:
        tasks.append(
            upsert_rate(
                "kenig", gr_ask + KENIG_ASK_OFFSET, gr_bid + KENIG_BID_OFFSET
            )
        )
    if bc_sell and bc_buy:
        tasks.append(upsert_rate("bestchange", bc_sell, bc_buy))
    if en_sell and en_buy:
        tasks.append(upsert_rate("energo", en_sell, en_buy))

    if tasks:
        await asyncio.gather(*tasks)


# ───────────────────── MAIN ─────────────────────────
def main() -> None:
    install_chromium_for_playwright()

    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("auth", auth))
    app.add_handler(CommandHandler("check", check))
    app.add_handler(CommandHandler("change", change_offsets))
    app.add_handler(CommandHandler("show_offsets", show_offsets))
    app.add_handler(CommandHandler("trades", trades_command))

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        send_rates_message,
        "interval",
        minutes=2,
        seconds=30,
        timezone=KALININGRAD_TZ,
        args=[app],
    )
    scheduler.start()

    logger.info("Bot started.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
