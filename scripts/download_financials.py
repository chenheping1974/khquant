#!/usr/bin/env python3
"""下载 baostock 季度财报 (利润表+资产负债表+现金流量表)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import time
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda s, o, **kw: pd.concat([s, o], ignore_index=kw.get("ignore_index", False))

import baostock as bs
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_fin")

N_STOCKS = 200
START_YEAR = 2023
END_YEAR = 2026

# 取股票列表
end = date.today()
start = end - timedelta(days=365 * 3)
syms = sorted(read_daily_bars(A_STOCK_DIR, start_date=start, end_date=end, market="a_stock")["symbol"].unique())[:N_STOCKS]
logger.info(f"下载 {len(syms)} 只股票财报, {START_YEAR}-{END_YEAR}")

bs.login()
rows = []
t0 = time.time()

for i, code in enumerate(syms):
    code_str = str(code).zfill(6)
    prefix = "sh" if code_str.startswith("6") else "sz"
    fc = f"{prefix}.{code_str}"

    # 利润表
    for y in range(START_YEAR, END_YEAR + 1):
        for q in [1, 2, 3, 4]:
            try:
                rs = bs.query_profit_data(code=fc, year=y, quarter=q)
                df = rs.get_data()
                if df is not None and not df.empty:
                    rows.append({
                        "symbol": code_str,
                        "pubDate": pd.to_datetime(df["pubDate"].iloc[0]).date(),
                        "statDate": pd.to_datetime(df["statDate"].iloc[0]).date(),
                        "roe": float(df["roeAvg"].iloc[0]) if df["roeAvg"].iloc[0] != "" else np.nan,
                        "gpMargin": float(df["gpMargin"].iloc[0]) if df["gpMargin"].iloc[0] != "" else np.nan,
                        "npMargin": float(df["npMargin"].iloc[0]) if df["npMargin"].iloc[0] != "" else np.nan,
                        "netProfit": float(df["netProfit"].iloc[0]) if df["netProfit"].iloc[0] != "" else np.nan,
                    })
            except Exception:
                pass
            time.sleep(0.03)

    # 资产负债表
    for y in range(START_YEAR, END_YEAR + 1):
        for q in [1, 2, 3, 4]:
            try:
                rs = bs.query_balance_data(code=fc, year=y, quarter=q)
                df = rs.get_data()
                if df is not None and not df.empty:
                    sd = pd.to_datetime(df["statDate"].iloc[0]).date()
                    for r in rows:
                        if r["symbol"] == code_str and r["statDate"] == sd:
                            r["debt_ratio"] = float(df["liabilityToAsset"].iloc[0]) if df["liabilityToAsset"].iloc[0] != "" else np.nan
            except Exception:
                pass
            time.sleep(0.03)

    # 现金流量表
    for y in range(START_YEAR, END_YEAR + 1):
        for q in [1, 2, 3, 4]:
            try:
                rs = bs.query_cash_flow_data(code=fc, year=y, quarter=q)
                df = rs.get_data()
                if df is not None and not df.empty:
                    sd = pd.to_datetime(df["statDate"].iloc[0]).date()
                    for r in rows:
                        if r["symbol"] == code_str and r["statDate"] == sd:
                            r["CFOtoNP"] = float(df["CFOToNP"].iloc[0]) if df["CFOToNP"].iloc[0] != "" else np.nan
            except Exception:
                pass
            time.sleep(0.03)

    if (i + 1) % 50 == 0:
        elapsed = (time.time() - t0) / 60
        logger.info(f"  [{i+1}/{len(syms)}] {len(rows)}条, {elapsed:.1f}min")

# 保存
fin = pd.DataFrame(rows)
fin = fin.sort_values(["symbol", "statDate"])
# 盈利稳定性: 8季度ROE滚动标准差
fin["roe_stability"] = fin.groupby("symbol")["roe"].transform(
    lambda x: x.rolling(8, min_periods=4).std()
)
fin.to_parquet(".cache_fin_200.parquet", index=False)
bs.logout()

elapsed = (time.time() - t0) / 60
logger.info(f"完成: {len(fin)}条, {fin['symbol'].nunique()}只, {elapsed:.1f}min")
logger.info(f"文件: .cache_fin_200.parquet")
