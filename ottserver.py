import os
import time
import warnings
import logging
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
import smtplib
import urllib.request
import urllib.parse
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

warnings.filterwarnings('ignore')

# Try to use Numba for JIT compilation; fall back silently if not available
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func):
        return func

# ---------------------- LOGGING ----------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ott_alerts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------------- MARKET HOURS ----------------------

IST = pytz.timezone('Asia/Kolkata')

def is_market_open(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=0, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_open <= now <= market_close

# ---------------------- DATA FETCH ----------------------

def get_stock_data(symbol: str, period: str = "5d", interval: str = "30m", retries: int = 2) -> Optional[pd.DataFrame]:
    for attempt in range(retries + 1):
        try:
            data = yf.Ticker(symbol).history(period=period, interval=interval)
            if data.empty or len(data) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                return None
            return data.dropna()
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                logger.error(f"Error fetching data for {symbol} after {retries + 1} attempts: {e}")
    return None

# ---------------------- OTT CALCULATION ----------------------

@njit
def _calc_var_loop(src: np.ndarray, valpha: float, vCMO: np.ndarray) -> np.ndarray:
    VAR = np.empty(len(src))
    VAR[0] = src[0]
    for i in range(1, len(src)):
        alpha = valpha * abs(vCMO[i])
        VAR[i] = alpha * src[i] + (1 - alpha) * VAR[i - 1]
    return VAR

@njit
def _calc_ott_loop(MAvg: np.ndarray, longStop: np.ndarray, shortStop: np.ndarray):
    n = len(MAvg)
    long_adj = longStop.copy()
    short_adj = shortStop.copy()
    direction = np.ones(n, dtype=np.int32)

    for i in range(1, n):
        long_adj[i] = max(longStop[i], long_adj[i - 1])
        short_adj[i] = min(shortStop[i], short_adj[i - 1])

        if MAvg[i] > short_adj[i - 1]:
            direction[i] = 1
        elif MAvg[i] < long_adj[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return long_adj, short_adj, direction

def calculate_ott(data: pd.DataFrame, length: int = 5, percent: float = 1.5):
    src = data['Close'].values
    valpha = 2 / (length + 1)

    diff = np.diff(src, prepend=src[0])
    vud = np.where(diff > 0, diff, 0.0)
    vdd = np.where(diff < 0, -diff, 0.0)

    # Rolling sum over 9 periods
    vUD = pd.Series(vud).rolling(9, min_periods=1).sum().values
    vDD = pd.Series(vdd).rolling(9, min_periods=1).sum().values

    denom = vUD + vDD
    vCMO = np.where(denom != 0, (vUD - vDD) / denom, 0.0)

    VAR = _calc_var_loop(src, valpha, vCMO)

    fark = VAR * percent * 0.01
    longStop = VAR - fark
    shortStop = VAR + fark

    long_adj, short_adj, direction = _calc_ott_loop(VAR, longStop, shortStop)

    MT = np.where(direction == 1, long_adj, short_adj)
    OTT = np.where(VAR > MT,
                   MT * (200 + percent) / 200,
                   MT * (200 - percent) / 200)

    idx = data.index
    return pd.Series(VAR, index=idx), pd.Series(OTT, index=idx)

# ---------------------- SIGNAL DETECTION ----------------------

def detect_signals(MAvg: pd.Series, OTT: pd.Series):
    OTT_shifted = OTT.shift(2)
    MAvg_prev = MAvg.shift(1)
    OTT_shifted_prev = OTT_shifted.shift(1)

    buy = (MAvg > OTT_shifted) & (MAvg_prev <= OTT_shifted_prev)
    sell = (MAvg < OTT_shifted) & (MAvg_prev >= OTT_shifted_prev)
    return buy.fillna(False), sell.fillna(False)

# ---------------------- WATCHLIST LOADING ----------------------

def load_watchlist(filepath: str = "watchlist.csv") -> list:
    try:
        symbols = pd.read_csv(filepath)["symbol"].dropna().tolist()
        logger.info(f"Loaded {len(symbols)} symbols from watchlist")
        return symbols
    except Exception as e:
        logger.error(f"Failed to load watchlist: {e}")
        return []

# ---------------------- EMAIL ----------------------

def send_combined_email(alerts: list, email_settings: dict) -> bool:
    if not alerts:
        return False
    try:
        buy_alerts = [a for a in alerts if a["signal"] == "BUY"]
        sell_alerts = [a for a in alerts if a["signal"] == "SELL"]

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        body = f"OTT Strategy Summary\n\nTime: {now_str}\nTotal Signals: {len(alerts)}\n\n"

        if buy_alerts:
            body += "BUY Signals:\n" + "".join(f"• {a['symbol']} @ ₹{a['price']:.2f}\n" for a in buy_alerts) + "\n"
        if sell_alerts:
            body += "SELL Signals:\n" + "".join(f"• {a['symbol']} @ ₹{a['price']:.2f}\n" for a in sell_alerts)

        msg = MIMEMultipart()
        msg['From'] = email_settings['email']
        msg['To'] = email_settings['recipient']
        msg['Subject'] = f"[OTT] {len(alerts)} Signals | {datetime.now().strftime('%H:%M')}"
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=10) as server:
            server.starttls()
            server.login(email_settings['email'], email_settings['password'])
            server.sendmail(email_settings['email'], email_settings['recipient'], msg.as_string())

        logger.info("Combined email sent successfully")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False

# ---------------------- TELEGRAM ----------------------

def send_telegram_alert(alerts: list, telegram_settings: dict) -> bool:
    """Send a combined Telegram message for all signals via the Bot API."""
    if not alerts:
        return False

    bot_token = telegram_settings.get("bot_token", "")
    chat_id = telegram_settings.get("chat_id", "")

    if not bot_token or not chat_id:
        logger.warning("Telegram credentials not set. Skipping Telegram alert.")
        return False

    try:
        buy_alerts  = [a for a in alerts if a["signal"] == "BUY"]
        sell_alerts = [a for a in alerts if a["signal"] == "SELL"]

        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        lines = [
            f"📊 *OTT Strategy Alert*",
            f"🕐 `{now_str}`",
            f"Total Signals: *{len(alerts)}*",
        ]

        if buy_alerts:
            lines.append("\n🟢 *BUY Signals:*")
            lines += [f"  • `{a['symbol']}` @ ₹{a['price']:.2f}" for a in buy_alerts]

        if sell_alerts:
            lines.append("\n🔴 *SELL Signals:*")
            lines += [f"  • `{a['symbol']}` @ ₹{a['price']:.2f}" for a in sell_alerts]

        text = "\n".join(lines)

        payload = json.dumps({
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown",
        }).encode("utf-8")

        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
            if not result.get("ok"):
                logger.error(f"Telegram API error: {result}")
                return False

        logger.info("Telegram alert sent successfully")
        return True

    except Exception as e:
        logger.error(f"Telegram alert failed: {e}")
        return False

# ---------------------- SCAN ONE SYMBOL ----------------------

def scan_symbol(symbol: str, ott_period: int, ott_percent: float) -> Optional[dict]:
    data = get_stock_data(symbol)
    if data is None or len(data) < 3:
        return None

    MAvg, OTT = calculate_ott(data, ott_period, ott_percent)
    buy, sell = detect_signals(MAvg, OTT)

    recent_buy = buy.iloc[-1] and not buy.iloc[-2]
    recent_sell = sell.iloc[-1] and not sell.iloc[-2]

    if not recent_buy and not recent_sell:
        return None

    signal = "BUY" if recent_buy else "SELL"
    price = data['Close'].iloc[-1]
    logger.info(f"{signal} - {symbol} @ ₹{price:.2f}")
    return {"symbol": symbol, "signal": signal, "price": price}

# ---------------------- MAIN LOOP ----------------------

def main():
    watchlist = load_watchlist()
    if not watchlist:
        logger.error("Watchlist is empty. Exiting.")
        return

    email_settings = {
        "email": os.environ.get("OTT_EMAIL", ""),
        "password": os.environ.get("OTT_EMAIL_PASSWORD", ""),
        "recipient": os.environ.get("OTT_RECIPIENT", ""),
    }

    if not all(email_settings.values()):
        logger.warning(
            "Email credentials not set. Set OTT_EMAIL, OTT_EMAIL_PASSWORD, "
            "and OTT_RECIPIENT environment variables."
        )

    telegram_settings = {
        "bot_token": os.environ.get("OTT_TG_BOT_TOKEN", ""),
        "chat_id": os.environ.get("OTT_TG_CHAT_ID", ""),
    }

    if not all(telegram_settings.values()):
        logger.warning(
            "Telegram credentials not set. Set OTT_TG_BOT_TOKEN and "
            "OTT_TG_CHAT_ID environment variables to enable Telegram alerts."
        )

    ott_period = 5
    ott_percent = 1.5
    scan_interval = 900       # 15 minutes
    max_workers = min(8, len(watchlist))

    logger.info("Starting OTT Alert Server")

    while True:
        now = datetime.now(IST)

        if not is_market_open(now):
            logger.info("Market closed. Sleeping...")
            time.sleep(scan_interval)
            continue

        logger.info("Starting new scan...")
        combined_alerts = []
        seen_symbols = set()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(scan_symbol, sym, ott_period, ott_percent): sym
                for sym in watchlist
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result()
                    if result and sym not in seen_symbols:
                        combined_alerts.append(result)
                        seen_symbols.add(sym)
                except Exception as e:
                    logger.error(f"Unexpected error for {sym}: {e}")

        if combined_alerts:
            send_combined_email(combined_alerts, email_settings)
            send_telegram_alert(combined_alerts, telegram_settings)
        else:
            logger.info("No signals this scan.")

        logger.info(f"Sleeping {scan_interval} seconds...\n")
        time.sleep(scan_interval)

if __name__ == "__main__":
    main()
