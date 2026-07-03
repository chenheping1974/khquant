"""
统一因子管线

因子设计 (参考幻方量化 + Fama-French):
  ┌─────────────────────────────────────────┐
  │ 技术因子 ~15个  (价量/趋势/波动)          │
  │ 基本面因子 ~15个 (估值/质量/增长/规模)     │
  │ 行业因子 ~5个   (行业虚拟变量)            │
  └─────────────────────────────────────────┘

与旧版(67个纯技术指标)相比:
  - 去掉了大量冗余的多周期变体
  - 新增基本面维度 (PE/PB/ROE/增长)
  - 因子总数 ~35个, 但信息维度更广
"""
import logging
from typing import Optional, Dict

import numpy as np
import pandas as pd

from src.features import a_stock as tech_feat
from src.features.fundamental import compute_fundamental_factors

logger = logging.getLogger(__name__)

# ── 精选技术因子列表 (从67个中选最不冗余的) ──────────

CORE_TECH_FACTORS = [
    # 趋势 (5个)
    "ma_dist_20d",           # 20日均线偏离 (短期趋势)
    "ma_dist_60d",           # 60日均线偏离 (中期趋势)
    "trend_strength_20d",    # 20日趋势强度
    "trend_strength_60d",    # 60日趋势强度
    "ma_cross_10_60",        # 金叉/死叉标志

    # 动量 (3个)
    "ret_20d",               # 20日收益率
    "sharpe_20d",            # 20日夏普动量
    "ret_accel_5_20",        # 收益加速度

    # 波动 (3个)
    "vol_20d",               # 20日波动率
    "atr_pct_14d",           # ATR百分比
    "amplitude_1d",          # 日振幅

    # 技术指标 (2个)
    "rsi_14",                # RSI
    "boll_position",         # 布林带位置

    # 量价 (2个)
    "vol_ratio_20d",         # 成交量比
    "amount_volatility_20d", # 成交额波动
]

# 基本面因子 (由 fundamental.py 生成)
FUNDAMENTAL_FACTORS = [
    "pe_ttm",                # 市盈率 (价值)
    "pb",                    # 市净率 (价值)
    "earnings_yield",        # 盈利收益率 1/PE
    "log_market_cap",        # 对数市值 (规模)
    "roe",                   # 净资产收益率 (质量)
    "profit_growth",         # 净利润增长率 (增长)
    "revenue_growth",        # 营收增长率
    "debt_ratio",            # 资产负债率 (安全性)
    "quality_score",         # 质量综合得分
]

# ── 缓存 ───────────────────────────────────────────────

_valuation_cache: Optional[pd.DataFrame] = None
_financial_cache: Optional[pd.DataFrame] = None


# ═══════════════════════════════════════════════════════
#  统计算
# ═══════════════════════════════════════════════════════

def compute_all_factors(
    df: pd.DataFrame,
    valuation_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    计算全量因子 (技术 + 基本面)。

    Parameters
    ----------
    df : 日线数据 (OHLCV)
    valuation_df : 估值数据 (可选, 如不传则跳过基本面)
    financial_df : 财务数据 (可选)

    Returns
    -------
    DataFrame with trade_date, symbol, close + all factor columns
    """
    logger.info("=" * 50)
    logger.info("统一因子计算")

    # 1. 技术因子 (用旧模块算全量, 然后筛选)
    logger.info("[1/3] 技术因子...")
    all_tech = tech_feat.compute_all_factors(df)
    available_tech = [f for f in CORE_TECH_FACTORS if f in all_tech.columns]
    missing_tech = set(CORE_TECH_FACTORS) - set(available_tech)
    if missing_tech:
        logger.warning(f"  缺失技术因子: {missing_tech}")
    result = all_tech[["trade_date", "symbol", "close"] + available_tech].copy()
    logger.info(f"  技术因子: {len(available_tech)}个 (精选自67个)")

    # 2. 基本面因子
    logger.info("[2/3] 基本面因子...")
    if valuation_df is not None and not valuation_df.empty:
        fund = compute_fundamental_factors(df, valuation_df, financial_df)
        fund_cols = [c for c in FUNDAMENTAL_FACTORS
                     if c in fund.columns and c not in result.columns]
        if fund_cols:
            result = result.merge(
                fund[["trade_date", "symbol"] + fund_cols],
                on=["trade_date", "symbol"], how="left"
            )
            # 基本面数据是慢变量, forward-fill 到每个交易日
            for c in fund_cols:
                result[c] = result.groupby("symbol")[c].ffill()
        logger.info(f"  基本面因子: {len(fund_cols)}个")
    else:
        logger.warning("  无估值数据, 跳过基本面因子")

    # 3. 清洗
    logger.info("[3/3] 清洗...")
    result = result.replace([np.inf, -np.inf], np.nan)
    # 去掉全NaN的列
    result = result.dropna(axis=1, how="all")

    factor_count = len(result.columns) - 3  # 扣除 trade_date/symbol/close
    logger.info(f"完成: {factor_count} 个因子, {len(result)}行")
    return result


def get_factor_names() -> list:
    """返回当前管线使用的因子名列表 (训练时用)"""
    return (CORE_TECH_FACTORS + FUNDAMENTAL_FACTORS).copy()


# ═══════════════════════════════════════════════════════
#  基本面数据获取 (缓存)
# ═══════════════════════════════════════════════════════

def get_valuation_data(force_refresh: bool = False) -> pd.DataFrame:
    """获取估值数据 (带缓存)"""
    global _valuation_cache
    if _valuation_cache is None or force_refresh:
        from src.features.fundamental import fetch_daily_valuation
        _valuation_cache = fetch_daily_valuation()
    return _valuation_cache


def get_financial_data(symbols: list, force_refresh: bool = False) -> pd.DataFrame:
    """获取财务数据 (带缓存)"""
    global _financial_cache
    if _financial_cache is None or force_refresh:
        from src.features.fundamental import fetch_financial_quality
        _financial_cache = fetch_financial_quality(symbols, max_symbols=500)
    return _financial_cache
