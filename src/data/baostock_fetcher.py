"""
baostock 历史估值数据

提供逐日 PE(TTM) + PB(MRQ) — 用于正确的时间点回测。

用法:
    df = fetch_historical_valuation(symbols=['600519','000001'],
                                     start_date='2023-01-01',
                                     end_date='2024-01-01')
"""
import logging
import time
from datetime import date
from typing import Optional, List

import numpy as np
import pandas as pd

# Monkey-patch: baostock 需要 pandas append (pandas 2.0 已移除)
if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat(
        [self, other], ignore_index=ignore_index
    )

import baostock as bs

logger = logging.getLogger(__name__)

# 缓存文件
_CACHE_DIR = None


def _ensure_login():
    """确保登录"""
    try:
        bs.login()
    except Exception:
        pass


def fetch_historical_valuation(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    逐只下载历史 PE/PB (日频)。

    Parameters
    ----------
    symbols : 股票代码列表 ['600519', '000001']
    start_date, end_date : '2023-01-01'

    Returns
    -------
    DataFrame: trade_date, symbol, close, pe_ttm, pb, total_shares
    """
    _ensure_login()
    logger.info(f"baostock: {len(symbols)}只, {start_date}~{end_date}")
    t0 = time.time()
    all_data = []

    for i, code in enumerate(symbols):
        code_str = str(code).zfill(6)
        prefix = "sh" if code_str.startswith("6") else "sz"
        full_code = f"{prefix}.{code_str}"

        try:
            rs = bs.query_history_k_data_plus(
                full_code,
                "date,code,close,peTTM,pbMRQ",
                start_date=start_date, end_date=end_date,
                frequency="d", adjustflag="2"  # 前复权
            )
            df = rs.get_data()
            if df is None or df.empty:
                continue

            df = df.rename(columns={
                "date": "trade_date",
                "code": "symbol",
                "close": "close",
                "peTTM": "pe_ttm",
                "pbMRQ": "pb",
            })
            df["symbol"] = code_str
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            for c in ["close", "pe_ttm", "pb"]:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            all_data.append(df[["trade_date", "symbol", "close", "pe_ttm", "pb"]])

        except Exception as e:
            logger.debug(f"  {code}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{i+1}/{len(symbols)}] {elapsed:.0f}s")

        time.sleep(0.05)  # 避免请求太快

    if not all_data:
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    # 清洗
    result.loc[result["pe_ttm"] <= 0, "pe_ttm"] = np.nan
    result.loc[result["pb"] <= 0, "pb"] = np.nan

    elapsed = time.time() - t0
    logger.info(f"  完成: {len(result)}行, {result['symbol'].nunique()}只, {elapsed:.0f}s")
    return result


def _ensure_logout():
    try:
        bs.logout()
    except Exception:
        pass
