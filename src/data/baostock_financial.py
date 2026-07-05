"""
baostock 历史季度财务数据

提供逐季 ROE, 毛利率, 负债率 — 用于 Quality 因子的时间点回测。

关键: pubDate 字段确保无前视偏差。
  回测在 2023-07-07 时, 只能用 pubDate <= 2023-07-07 的财报。

用法:
    df = fetch_quarterly_financials(symbols=['600519','000001'])
"""
import logging
import time
from typing import List, Optional

import numpy as np
import pandas as pd

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda self, other, ignore_index=False, **kw: pd.concat(
        [self, other], ignore_index=ignore_index
    )

import baostock as bs

logger = logging.getLogger(__name__)


def _ensure_login():
    try:
        bs.login()
    except Exception:
        pass


def fetch_quarterly_financials(
    symbols: List[str],
    start_year: int = 2020,
    end_year: int = 2026,
) -> pd.DataFrame:
    """
    逐只下载季度利润表 + 资产负债表。

    Returns DataFrame:
        symbol, pubDate, statDate,
        roe, gross_margin, net_profit, eps_ttm,  (利润表)
        debt_ratio,                                (资产负债表)
        accrual_pct                                 (应计/总资产, 算得)
    """
    _ensure_login()
    n = len(symbols)
    logger.info(f"baostock财报: {n}只, {start_year}-{end_year}")
    t0 = time.time()

    all_rows = []

    for i, code in enumerate(symbols):
        code_str = str(code).zfill(6)
        prefix = "sh" if code_str.startswith("6") else "sz"
        full_code = f"{prefix}.{code_str}"

        # ── 利润表 ──
        for y in range(start_year, end_year + 1):
            for q in [1, 2, 3, 4]:
                try:
                    rs = bs.query_profit_data(code=full_code, year=y, quarter=q)
                    df = rs.get_data()
                    if df is None or df.empty:
                        continue
                    row = {
                        "symbol": code_str,
                        "pubDate": pd.to_datetime(df["pubDate"].iloc[0]).date(),
                        "statDate": pd.to_datetime(df["statDate"].iloc[0]).date(),
                        "roe": float(df["roeAvg"].iloc[0]) if df["roeAvg"].iloc[0] else np.nan,
                        "gross_margin": float(df["gpMargin"].iloc[0]) if df["gpMargin"].iloc[0] else np.nan,
                        "net_profit": float(df["netProfit"].iloc[0]) if df["netProfit"].iloc[0] else np.nan,
                        "eps_ttm": float(df["epsTTM"].iloc[0]) if df["epsTTM"].iloc[0] else np.nan,
                    }
                    all_rows.append(row)
                except Exception:
                    pass
                time.sleep(0.03)

        # ── 资产负债表 ──
        for y in range(start_year, end_year + 1):
            for q in [1, 2, 3, 4]:
                try:
                    rs = bs.query_balance_data(code=full_code, year=y, quarter=q)
                    df = rs.get_data()
                    if df is None or df.empty:
                        continue
                    # 找到对应的利润表行, 加上负债率
                    stat_dt = pd.to_datetime(df["statDate"].iloc[0]).date()
                    for row in all_rows:
                        if row["symbol"] == code_str and row["statDate"] == stat_dt:
                            row["debt_ratio"] = float(df["liabilityToAsset"].iloc[0]) if df["liabilityToAsset"].iloc[0] else np.nan
                            break
                except Exception:
                    pass
                time.sleep(0.03)

        if (i + 1) % 50 == 0:
            elapsed = (time.time() - t0) / 60
            logger.info(f"  [{i+1}/{n}] {elapsed:.1f}min")

    result = pd.DataFrame(all_rows)
    if not result.empty:
        # 应计利润 = (1 - CFO/NP) 近似: 用资产周转率...这里简化
        # CFOtoNP ≈ 0.9 for most companies, accrual = (1-CFOtoNP)*netProfit/totalAssets
        # 简化: accrual_pct = 1% of net profit (placeholder)
        result["accrual_pct"] = np.nan  # TODO: 接入现金流量表
        result["roe"] = result["roe"].clip(-1, 1)

    elapsed = (time.time() - t0) / 60
    logger.info(f"  完成: {len(result)}行, {result['symbol'].nunique()}只, {elapsed:.1f}min")
    return result
