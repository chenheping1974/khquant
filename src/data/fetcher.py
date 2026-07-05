"""
数据采集模块

数据源:
  - A股日线K线      → Sina (未复权, 稳定)
  - 复权因子         → Eastmoney (每周更新一次, 低频)
  - A股实时行情      → Sina hq.sinajs.cn
  - 黄金白银         → Sina 全球期货
  - 宏观             → yfinance

后复权处理:
  Sina返回未复权数据 → 存储层保存未复权
  → 因子计算时通过复权因子转为后复权
  → 复权因子每周从 Eastmoney 同步一次 (低频, 不限流)
"""
import time
import json
import pandas as pd
import requests
from datetime import date, timedelta
from typing import Optional, List, Dict
from pathlib import Path
import logging

from config import DATA_RETENTION_DAYS, MACRO_RETENTION_DAYS
from src.data.storage import write_daily_bars, read_daily_bars, get_latest_date
from config import A_STOCK_DIR, GOLD_SILVER_DIR, MACRO_DIR, DATA_DIR

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://finance.sina.com.cn/",
}

REQUEST_DELAY = 0.15     # Sina请求间隔 (Sina极少限流)
MAX_RETRIES = 3
IDLE_THRESHOLD = 200
PROGRESS_INTERVAL = 200

# ── API URLs ─────────────────────────────────────────
SINA_KLN_URL = (
    "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
    "/CN_MarketData.getKLineData"
)
SINA_QUOTE_URL = "https://hq.sinajs.cn/list"
SINA_FUTURES_URL = (
    "https://stock2.finance.sina.com.cn/futures/api/json_v2.php"
    "/GlobalFuturesService.getGlobalFuturesDailyKLine"
)
EASTMONEY_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


# ═══════════════════════════════════════════════════════
#  A股日线 — Sina (未复权)
# ═══════════════════════════════════════════════════════

def _to_sina_symbol(code: str) -> str:
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_a_stock_daily(
    symbols: List[str],
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    incremental: bool = True,
) -> pd.DataFrame:
    """
    A股日线 — Sina (未复权)。

    增量模式: 已有股票拉最近K线, 新股票全量
    """
    if end_date is None:
        end_date = date.today()

    if incremental:
        latest = get_latest_date(A_STOCK_DIR)
        fetch_start = latest - timedelta(days=5) if latest else end_date - timedelta(days=DATA_RETENTION_DAYS)
    else:
        fetch_start = start_date or (end_date - timedelta(days=DATA_RETENTION_DAYS))

    # 已有代码
    existing_codes = set()
    if incremental and latest:
        try:
            from src.data.storage import get_unique_symbols
            existing_codes = set(get_unique_symbols(A_STOCK_DIR))
        except Exception:
            pass

    logger.info(f"A股(Sina): {len(symbols)}只, 已有{len(existing_codes)}, {fetch_start}→{end_date}")
    t0 = time.time()

    all_data = []
    consecutive_empty = 0

    for i, code in enumerate(symbols):
        code = str(code).zfill(6)
        datalen = 10 if (code in existing_codes and incremental) else 5000

        try:
            df = _fetch_sina_kline(_to_sina_symbol(code), datalen=datalen)
            if df is not None and not df.empty:
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
                if incremental and latest:
                    df = df[df["trade_date"] > latest]
                else:
                    df = df[(df["trade_date"] >= fetch_start) & (df["trade_date"] <= end_date)]
                if not df.empty:
                    all_data.append(df)
                    consecutive_empty = 0
                else:
                    consecutive_empty += 1
            else:
                consecutive_empty += 1
        except Exception:
            consecutive_empty += 1

        if consecutive_empty >= IDLE_THRESHOLD:
            logger.info(f"  连续{consecutive_empty}只无数据, 判定休市")
            break

        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  [{i+1}/{len(symbols)}] {len(all_data)}批, {elapsed:.1f}min")

        time.sleep(REQUEST_DELAY)

    if not all_data:
        logger.warning("未获取到A股数据 (可能休市)")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)
    logger.info(f"A股完成: {len(result)}行, {result['symbol'].nunique()}只, {(time.time()-t0)/60:.1f}min")
    return result


def _fetch_sina_kline(sina_sym: str, datalen: int = 5000) -> Optional[pd.DataFrame]:
    """新浪个股日线 (未复权)"""
    params = {"symbol": sina_sym, "scale": "240", "ma": "no", "datalen": str(datalen)}

    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(SINA_KLN_URL, params=params, headers=HEADERS, timeout=15)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                return None

            df = pd.DataFrame(data)
            df = df.rename(columns={
                "day": "trade_date", "open": "open", "high": "high",
                "low": "low", "close": "close", "volume": "volume",
            })
            df["symbol"] = sina_sym[2:]
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            return df[["trade_date", "symbol", "open", "high", "low", "close", "volume"]]
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(REQUEST_DELAY * (2 ** attempt))
    return None


# ═══════════════════════════════════════════════════════
#  复权因子 — Eastmoney (每周同步一次)
# ═══════════════════════════════════════════════════════

ADJUST_FACTOR_FILE = DATA_DIR / "adjust_factors.parquet"


def sync_adjust_factors(symbols: List[str]):
    """
    从 Eastmoney 同步后复权因子 (每周一次)。

    后复权因子 = 后复权收盘价 / 未复权收盘价

    然后用这个因子乘以 Sina 的未复权数据得到后复权价。
    """
    logger.info(f"同步复权因子: {len(symbols)}只...")
    factors = {}
    t0 = time.time()

    for i, code in enumerate(symbols):
        code = str(code).zfill(6)
        try:
            # 获取最新的未复权日线 (从Sina已缓存的数据)
            raw_df = _fetch_sina_kline(_to_sina_symbol(code), datalen=2)
            if raw_df is None or raw_df.empty:
                continue

            latest_raw_close = float(raw_df["close"].iloc[-1])
            latest_date = raw_df["trade_date"].iloc[-1]

            # 获取同日的后复权价格 (Eastmoney)
            adj_df = _fetch_em_kline_raw(code, start_date=latest_date, end_date=latest_date, fqt=2)
            if adj_df is None or adj_df.empty:
                # 回退: 用前复权日线中最新的收盘价
                adj_df = _fetch_em_kline_raw(code, start_date=latest_date, end_date=latest_date, fqt=2)
                if adj_df is None or adj_df.empty:
                    continue

            latest_adj_close = float(adj_df["close"].iloc[-1])
            factor = latest_adj_close / latest_raw_close if latest_raw_close > 0 else 1.0
            factors[code] = {"trade_date": latest_date, "factor": factor}

        except Exception:
            continue

        time.sleep(0.3)  # 低频请求, 不限流

        if (i + 1) % 500 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  [{i+1}/{len(symbols)}] {len(factors)}个, {elapsed:.1f}min")

    df = pd.DataFrame(factors).T.reset_index()
    df.columns = ["symbol", "trade_date", "factor"]
    df.to_parquet(ADJUST_FACTOR_FILE, index=False)
    logger.info(f"复权因子保存: {ADJUST_FACTOR_FILE}, {len(df)}只, {(time.time()-t0)/60:.1f}min")
    return df


def load_adjust_factors() -> pd.DataFrame:
    """加载复权因子"""
    if not ADJUST_FACTOR_FILE.exists():
        return pd.DataFrame(columns=["symbol", "trade_date", "factor"])
    return pd.read_parquet(ADJUST_FACTOR_FILE)


def apply_hfq(df: pd.DataFrame, adjust_factors: pd.DataFrame) -> pd.DataFrame:
    """
    对未复权数据应用后复权因子。

    df: 未复权日线 (trade_date, symbol, open, high, low, close, volume)
    adjust_factors: (symbol, trade_date, factor)
    """
    if adjust_factors.empty or df.empty:
        df["is_hfq"] = False
        return df

    # merge factor
    merged = df.merge(adjust_factors, on="symbol", how="left", suffixes=("", "_adj"))
    merged["factor"] = merged["factor"].fillna(1.0)

    for col in ["open", "high", "low", "close"]:
        if col in merged.columns:
            merged[col] = merged[col] * merged["factor"]

    merged["is_hfq"] = True
    drop_cols = [c for c in merged.columns if c.endswith("_adj") or c == "factor"]
    return merged.drop(columns=drop_cols)


def _fetch_em_kline_raw(code: str, start_date: date, end_date: date,
                        fqt: int = 2) -> Optional[pd.DataFrame]:
    """Eastmoney K线 (内部, 无重试)"""
    code = str(code).zfill(6)
    secid = f"1.{code}" if code.startswith("6") else f"0.{code}"
    params = {
        "secid": secid,
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": str(fqt),
        "beg": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "lmt": "100",
    }

    r = requests.get(EASTMONEY_URL, params=params, headers=HEADERS, timeout=15)
    data = r.json()
    klines = (data.get("data") or {}).get("klines")
    if not klines:
        return None

    rows = [line.split(",") for line in klines]
    df = pd.DataFrame(rows, columns=[
        "trade_date", "open", "close", "high", "low", "volume",
        "amount", "amplitude", "pct_change", "change", "turnover_rate",
    ])
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y-%m-%d").dt.date
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════
#  实时行情 — Sina
# ═══════════════════════════════════════════════════════

def fetch_sina_quotes(sina_symbols: List[str]) -> pd.DataFrame:
    """新浪实时行情"""
    if not sina_symbols:
        return pd.DataFrame()
    all_rows = []
    for i in range(0, len(sina_symbols), 800):
        batch = sina_symbols[i:i + 800]
        try:
            r = requests.get(f"{SINA_QUOTE_URL}={','.join(batch)}",
                             headers=HEADERS, timeout=30)
            r.encoding = "gbk"
            for line in r.text.strip().split("\n"):
                if "=" not in line:
                    continue
                parts = line.split("=")
                sym = parts[0].strip().replace("var hq_str_", "")
                fields = parts[1].strip().strip('";').split(",")
                if len(fields) < 10:
                    continue
                all_rows.append({
                    "symbol": sym,
                    "name": fields[0],
                    "open": _float(fields, 1),
                    "close": _float(fields, 3),
                    "high": _float(fields, 4),
                    "low": _float(fields, 5),
                    "volume": _float(fields, 8),
                    "amount": _float(fields, 9),
                })
        except Exception:
            pass
    return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()


def _float(fields, idx):
    try:
        return float(fields[idx]) if idx < len(fields) and fields[idx] else None
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════
#  黄金白银 — Sina 全球期货
# ═══════════════════════════════════════════════════════

def fetch_gold_silver_daily(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> pd.DataFrame:
    """黄金白银日线 — Sina 全球期货"""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=MACRO_RETENTION_DAYS)

    logger.info(f"金银(Sina): {start_date}→{end_date}")
    dfs = []

    for name, sym in [("XAUUSD", "XAU"), ("XAGUSD", "XAG")]:
        try:
            df = _fetch_global_futures(sym, datalen=5000)
            if df is not None and not df.empty:
                df["symbol"] = name
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
                df = df[(df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)]
                if not df.empty:
                    dfs.append(df)
                    logger.info(f"  {name}: {len(df)}条")
        except Exception as e:
            logger.warning(f"  {name}: {e}")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _fetch_global_futures(symbol: str, datalen: int = 5000) -> Optional[pd.DataFrame]:
    """新浪全球期货"""
    params = {"symbol": symbol, "datalen": str(datalen)}
    r = requests.get(SINA_FUTURES_URL, params=params, headers=HEADERS, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        return None

    # API 返回的 key 是 "date", 映射到 "trade_date"
    df = pd.DataFrame(data)
    if "date" in df.columns:
        df = df.rename(columns={"date": "trade_date"})
    # 只保留需要的列
    need_cols = ["trade_date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in need_cols if c in df.columns]]
    for c in ["open", "high", "low", "close", "volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ═══════════════════════════════════════════════════════
#  宏观 — yfinance
# ═══════════════════════════════════════════════════════

MACRO_TICKERS = {"DXY": "DX-Y.NYB", "US10Y": "^TNX", "GLD": "GLD", "SLV": "SLV"}


def fetch_macro_daily(start_date=None, end_date=None) -> pd.DataFrame:
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=MACRO_RETENTION_DAYS)

    logger.info(f"宏观(yfinance): {start_date}→{end_date}")
    import yfinance as yf
    dfs = []

    for name, ticker in MACRO_TICKERS.items():
        if name in ("GLD", "SLV"):
            continue  # 金银已由Sina覆盖
        for attempt in range(MAX_RETRIES):
            try:
                data = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                                   end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                                   progress=False, auto_adjust=True)
                if not data.empty:
                    df = data[["Close"]].reset_index()
                    df.columns = ["date", "close"]
                    df["trade_date"] = pd.to_datetime(df["date"]).dt.date
                    df["symbol"] = name
                    dfs.append(df[["trade_date", "symbol", "close"]])
                    break
            except Exception:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
        else:
            logger.warning(f"  {name}: 失败")

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  完整数据更新
# ═══════════════════════════════════════════════════════

def update_all_data(symbols: List[str]) -> Dict[str, int]:
    """每日增量更新"""
    stats, today = {}, date.today()

    logger.info("=" * 50)
    logger.info("[1/3] A股 (Sina 未复权)...")
    df = fetch_a_stock_daily(symbols, incremental=True, end_date=today)
    if not df.empty:
        write_daily_bars(df, A_STOCK_DIR, market="a_stock")
        stats["a_stock"] = len(df)
    else:
        stats["a_stock"] = 0

    logger.info("=" * 50)
    logger.info("[2/3] 金银 (Sina)...")
    df = fetch_gold_silver_daily(start_date=today - timedelta(days=30), end_date=today)
    if not df.empty:
        write_daily_bars(df, GOLD_SILVER_DIR, market="gold_silver")
        stats["gold_silver"] = len(df)
    else:
        stats["gold_silver"] = 0

    logger.info("=" * 50)
    logger.info("[3/3] 宏观 (yfinance)...")
    df = fetch_macro_daily(start_date=today - timedelta(days=60), end_date=today)
    if not df.empty:
        write_daily_bars(df, MACRO_DIR, market="macro")
        stats["macro"] = len(df)
    else:
        stats["macro"] = 0

    logger.info("=" * 50)
    logger.info(f"完成: {stats}")
    return stats


def load_all_data(symbols=None, lookback_days=252, end_date=None):
    if end_date is None:
        end_date = date.today()
    start = end_date - timedelta(days=lookback_days)
    return {
        "a_stock": read_daily_bars(A_STOCK_DIR, symbols=symbols,
                                   start_date=start, end_date=end_date, market="a_stock"),
        "gold_silver": read_daily_bars(GOLD_SILVER_DIR, start_date=start,
                                       end_date=end_date, market="gold_silver"),
        "macro": read_daily_bars(MACRO_DIR, start_date=start,
                                 end_date=end_date, market="macro"),
    }


def _list_parquet_symbols(base_dir) -> List[str]:
    import pyarrow.parquet as pq
    codes = set()
    for d in sorted(base_dir.glob("year=*/month=*/")):
        f = d / "data.parquet"
        if f.exists():
            for batch in pq.ParquetFile(f).iter_batches(columns=["symbol"]):
                codes.update(str(s).zfill(6) for s in batch.column("symbol").to_pylist())
    return sorted(codes)
