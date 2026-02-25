"""
OTT (Optimized Trend Tracker) Scanner — NSE Intraday/Swing Setups
Usage:
    python ott_scanner.py                    # test mode: first 30 symbols
    python ott_scanner.py --full             # scan entire watchlist
    python ott_scanner.py --no-market-check  # skip market-hours guard
    python ott_scanner.py --interval 5m      # override candle interval
    python ott_scanner.py --length 7 --percent 2.0  # OTT params
"""

import argparse
import logging
import signal
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from numba import njit

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # Data
    watchlist_path: str = "watchlist.csv"
    period: str = "10d"
    interval: str = "15m"
    min_bars: int = 60
    fetch_retries: int = 3

    # OTT
    ott_length: int = 5
    ott_percent: float = 1.5

    # Signal filters
    min_atr_pct: float = 0.006      # 0.6% minimum ATR/price
    min_volume_ratio: float = 0.80  # relative to 20-bar avg
    atr_stop_multiplier: float = 2.0
    confirm_bars: int = 2           # crossover must hold for N bars

    # Execution
    max_workers: int = 5
    test_limit: int = 30            # symbols in test mode
    full_scan: bool = False
    skip_market_check: bool = False

    # Logging
    log_file: str = "ott_alerts.log"
    log_level: int = logging.INFO


# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────

def setup_logger(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("OTT")
    logger.setLevel(cfg.log_level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")

    fh = logging.FileHandler(cfg.log_file)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ──────────────────────────────────────────────────────────────────────────────
# MARKET HOURS
# ──────────────────────────────────────────────────────────────────────────────

IST = pytz.timezone("Asia/Kolkata")

def is_market_open(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    open_t  = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    close_t = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return open_t <= now <= close_t


# ──────────────────────────────────────────────────────────────────────────────
# WATCHLIST
# ──────────────────────────────────────────────────────────────────────────────

def load_watchlist(path: str, logger: logging.Logger) -> List[str]:
    try:
        df = pd.read_csv(path)
        symbols = df["symbol"].dropna().str.strip().unique().tolist()
        logger.info(f"Loaded {len(symbols)} symbols from {path}")
        return symbols
    except Exception as e:
        logger.error(f"Failed to load watchlist: {e}")
        return []


# ──────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ──────────────────────────────────────────────────────────────────────────────

def _format_symbol(sym: str) -> str:
    if sym.startswith("^"):
        return sym
    return sym if sym.endswith(".NS") else f"{sym}.NS"


def get_stock_data(
    symbol: str,
    cfg: Config,
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    formatted = _format_symbol(symbol)
    for attempt in range(cfg.fetch_retries + 1):
        try:
            ticker = yf.Ticker(formatted)
            data = ticker.history(period=cfg.period, interval=cfg.interval)
            if data.empty or len(data) < cfg.min_bars:
                logger.debug(f"{formatted}: insufficient data ({len(data)} bars)")
                return None
            return data.dropna()
        except Exception as e:
            logger.debug(f"{formatted} attempt {attempt + 1} failed: {e}")
            if attempt < cfg.fetch_retries:
                time.sleep(0.4 * (attempt + 1))
    logger.warning(f"All fetch attempts failed: {formatted}")
    return None


# ──────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ──────────────────────────────────────────────────────────────────────────────

@njit
def _var_loop(src: np.ndarray, valpha: float, vCMO: np.ndarray) -> np.ndarray:
    VAR = np.empty(len(src))
    VAR[0] = src[0]
    for i in range(1, len(src)):
        alpha = valpha * abs(vCMO[i])
        VAR[i] = alpha * src[i] + (1.0 - alpha) * VAR[i - 1]
    return VAR


@njit
def _ott_loop(
    MAvg: np.ndarray,
    longStop: np.ndarray,
    shortStop: np.ndarray,
):
    n = len(MAvg)
    long_adj  = longStop.copy()
    short_adj = shortStop.copy()
    direction = np.ones(n, dtype=np.int32)

    for i in range(1, n):
        long_adj[i]  = max(longStop[i],  long_adj[i - 1])
        short_adj[i] = min(shortStop[i], short_adj[i - 1])

        if   MAvg[i] > short_adj[i - 1]:  direction[i] =  1
        elif MAvg[i] < long_adj[i - 1]:   direction[i] = -1
        else:                              direction[i] =  direction[i - 1]

    return long_adj, short_adj, direction


def calculate_ott(
    data: pd.DataFrame,
    length: int = 5,
    percent: float = 1.5,
):
    src    = data["Close"].values.astype(np.float64)
    valpha = 2.0 / (length + 1)

    diff = np.diff(src, prepend=src[0])
    vud  = np.where(diff > 0,  diff, 0.0)
    vdd  = np.where(diff < 0, -diff, 0.0)

    vUD  = pd.Series(vud).ewm(span=9, adjust=False).mean().values
    vDD  = pd.Series(vdd).ewm(span=9, adjust=False).mean().values

    denom = vUD + vDD
    vCMO  = np.zeros_like(denom)
    mask  = denom != 0
    vCMO[mask] = (vUD[mask] - vDD[mask]) / denom[mask]

    VAR      = _var_loop(src, valpha, vCMO)
    fark     = VAR * percent * 0.01
    long_adj, short_adj, direction = _ott_loop(VAR, VAR - fark, VAR + fark)

    MT  = np.where(direction == 1, long_adj, short_adj)
    OTT = np.where(
        VAR > MT,
        MT * (200 + percent) / 200,
        MT * (200 - percent) / 200,
    )
    return VAR, OTT


def calculate_atr(data: pd.DataFrame, period: int = 14) -> np.ndarray:
    high, low, close = (
        data["High"].values,
        data["Low"].values,
        data["Close"].values,
    )
    prev_close     = np.roll(close, 1)
    prev_close[0]  = close[0]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    return pd.Series(tr).rolling(period, min_periods=1).mean().values


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_signal(VAR: np.ndarray, OTT: np.ndarray, confirm: int = 2) -> Optional[str]:
    """
    Crossover detected in the last `confirm` bars.
    All `confirm` bars must agree (avoids single-bar whipsaws).
    """
    if len(VAR) < confirm + 2:
        return None

    # Check crossover at bar -(confirm+1) → -(confirm)
    cross_idx = -(confirm + 1)
    above_before = VAR[cross_idx - 1] > OTT[cross_idx - 1]
    below_before = VAR[cross_idx - 1] < OTT[cross_idx - 1]

    # All recent bars since crossover must maintain direction
    recent_var = VAR[cross_idx:]
    recent_ott = OTT[cross_idx:]

    if not above_before and all(recent_var > recent_ott):
        return "BUY"
    if not below_before and all(recent_var < recent_ott):
        return "SELL"

    return None


# ──────────────────────────────────────────────────────────────────────────────
# SCAN
# ──────────────────────────────────────────────────────────────────────────────

def scan_symbol(
    symbol: str,
    cfg: Config,
    logger: logging.Logger,
) -> Optional[Dict[str, Any]]:
    try:
        data = get_stock_data(symbol, cfg, logger)
        if data is None:
            return None

        VAR, OTT = calculate_ott(data, cfg.ott_length, cfg.ott_percent)
        ATR       = calculate_atr(data)

        signal = detect_signal(VAR, OTT, cfg.confirm_bars)
        if not signal:
            return None

        close     = float(data["Close"].iloc[-1])
        atr_val   = float(ATR[-1])
        atr_pct   = atr_val / close

        # ATR filter
        if atr_pct < cfg.min_atr_pct:
            return None

        # Volume filter
        vol_now = float(data["Volume"].iloc[-1])
        vol_avg = float(data["Volume"].iloc[-20:].mean())
        if vol_avg == 0 or vol_now < cfg.min_volume_ratio * vol_avg:
            return None

        vol_ratio = vol_now / vol_avg
        stop = (
            close - cfg.atr_stop_multiplier * atr_val
            if signal == "BUY"
            else close + cfg.atr_stop_multiplier * atr_val
        )
        risk_pct = abs(close - stop) / close * 100

        return {
            "symbol":  symbol,
            "signal":  signal,
            "price":   round(close, 2),
            "atr%":    round(atr_pct * 100, 2),
            "vol_x":   round(vol_ratio, 2),
            "stop":    round(stop, 2),
            "risk%":   round(risk_pct, 2),
            "time":    datetime.now(IST).strftime("%H:%M"),
        }

    except Exception as e:
        logger.debug(f"{symbol} scan error: {e}")
        return None


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

_shutdown = False

def _handle_sigint(sig, frame):
    global _shutdown
    print("\nInterrupted — showing results so far...\n")
    _shutdown = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OTT Scanner for NSE")
    p.add_argument("--full",             action="store_true",  help="Scan entire watchlist")
    p.add_argument("--no-market-check",  action="store_true",  help="Skip market-hours guard")
    p.add_argument("--interval",         default="15m",        help="Candle interval (default: 15m)")
    p.add_argument("--period",           default="10d",        help="History period  (default: 10d)")
    p.add_argument("--length",           type=int,   default=5,   help="OTT length")
    p.add_argument("--percent",          type=float, default=1.5, help="OTT percent")
    p.add_argument("--workers",          type=int,   default=5,   help="Thread workers")
    p.add_argument("--limit",            type=int,   default=30,  help="Test-mode symbol limit")
    p.add_argument("--watchlist",        default="watchlist.csv")
    p.add_argument("--debug",            action="store_true")
    return p.parse_args()


def print_result(res: Dict[str, Any]):
    tag  = "▲ BUY " if res["signal"] == "BUY" else "▼ SELL"
    line = (
        f"  {tag}  {res['symbol']:<12} "
        f"@ {res['price']:>9.2f}  "
        f"| ATR {res['atr%']:>4.1f}%  "
        f"| Vol {res['vol_x']:>4.1f}x  "
        f"| Stop {res['stop']:>9.2f}  "
        f"| Risk {res['risk%']:>4.1f}%  "
        f"[{res['time']}]"
    )
    print(line)


def main():
    signal.signal(signal.SIGINT, _handle_sigint)

    args   = parse_args()
    cfg    = Config(
        watchlist_path     = args.watchlist,
        interval           = args.interval,
        period             = args.period,
        ott_length         = args.length,
        ott_percent        = args.percent,
        max_workers        = args.workers,
        test_limit         = args.limit,
        full_scan          = args.full,
        skip_market_check  = args.no_market_check,
        log_level          = logging.DEBUG if args.debug else logging.INFO,
    )
    logger = setup_logger(cfg)

    # Market-hours guard
    if not cfg.skip_market_check and not is_market_open():
        logger.warning("Market is closed. Use --no-market-check to scan anyway.")
        sys.exit(0)

    watchlist = load_watchlist(cfg.watchlist_path, logger)
    if not watchlist:
        logger.error("Empty watchlist — aborting.")
        sys.exit(1)

    symbols = watchlist if cfg.full_scan else watchlist[: cfg.test_limit]
    mode    = "FULL" if cfg.full_scan else f"TEST ({cfg.test_limit} symbols)"

    print(f"\n{'─'*65}")
    print(f"  OTT Scanner  |  {mode}  |  {cfg.interval} bars  |  {len(symbols)} symbols")
    print(f"{'─'*65}\n")

    results: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {executor.submit(scan_symbol, s, cfg, logger): s for s in symbols}
        done = 0
        for future in as_completed(futures):
            if _shutdown:
                executor.shutdown(wait=False, cancel_futures=True)
                break
            done += 1
            res = future.result()
            if res:
                results.append(res)
                print_result(res)
            # Progress tick every 10 symbols
            if done % 10 == 0:
                pct = done / len(symbols) * 100
                print(f"  ... {done}/{len(symbols)} scanned ({pct:.0f}%)")

    # ── Summary table ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    if results:
        df = (
            pd.DataFrame(results)
            .sort_values(["signal", "atr%"], ascending=[True, False])
            .reset_index(drop=True)
        )
        buys  = df[df["signal"] == "BUY"]
        sells = df[df["signal"] == "SELL"]

        for label, subset in [("BUY SETUPS", buys), ("SELL SETUPS", sells)]:
            if subset.empty:
                continue
            print(f"\n  {label}  ({len(subset)} found)")
            print(f"  {'Symbol':<12} {'Price':>9} {'ATR%':>6} {'Vol':>6} {'Stop':>9} {'Risk%':>6}")
            print(f"  {'-'*55}")
            for _, row in subset.iterrows():
                print(
                    f"  {row['symbol']:<12} "
                    f"{row['price']:>9.2f} "
                    f"{row['atr%']:>5.1f}% "
                    f"{row['vol_x']:>5.1f}x "
                    f"{row['stop']:>9.2f} "
                    f"{row['risk%']:>5.1f}%"
                )
        print()
    else:
        print("  No setups found. Try during market hours (9:15–15:30 IST)\n")


if __name__ == "__main__":
    main()