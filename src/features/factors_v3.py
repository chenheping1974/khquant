"""
khquant 因子体系 v3.0 — MSCI / BlackRock / AQR 标准

四大核心因子, 每个由2-5个子指标合成:
  Value     (25%)   E/P + B/P + CF/P
  Momentum  (25%)   价格动量 + 风险调整动量 + 盈利动量
  Quality   (40%)   ROE + 毛利率 + 应计利润 + 杠杆 + 盈利稳定性
  LowVol    (10%)   波动率 + Beta + 残差波动率

处理流程:
  1. Winsorize (3σ)
  2. 行业中性化 (行业内 Z-score)
  3. 等权合成子指标 → 因子得分
  4. 加权总分 → 选股

用法:
    from src.features.factors_v3 import compute_four_factors
    scores = compute_four_factors(price_df, valuation_df, financial_df, industry_map)
"""
import logging
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── 因子权重 (BlackRock 标准, Quality 为核心) ──
FACTOR_WEIGHTS = {
    "value": 0.25,
    "momentum": 0.25,
    "quality": 0.40,
    "lowvol": 0.10,
}

# ── 组合约束 ──
MAX_SECTOR_PCT = 0.20      # 单行业 ≤ 20%
MAX_STOCK_WEIGHT = 0.05    # 单票 ≤ 5%
MAX_MONTHLY_TURNOVER = 0.25  # 月换手率 ≤ 25%


# ═══════════════════════════════════════════════════════
#  Value 因子 (LSY 2019 + BlackRock)
# ═══════════════════════════════════════════════════════

def _compute_value(close: np.ndarray, pe: float, pb: float,
                   operating_cf: float, total_shares: float) -> Tuple[float, float, float]:
    """
    Value = (E/P + B/P + CF/P) / 3

    E/P: 盈利收益率 = 1/PE
    B/P: 净资产收益率 = 1/PB
    CF/P: 现金流收益率 = 经营现金流 / 市值
    """
    ep = 1.0 / pe if pe and pe > 0 else np.nan
    bp = 1.0 / pb if pb and pb > 0 else np.nan
    latest_close = close[-1] if len(close) > 0 else np.nan
    mktcap = latest_close * total_shares if total_shares and latest_close else np.nan
    cfp = operating_cf / mktcap if operating_cf and mktcap and mktcap > 0 else np.nan
    return ep, bp, cfp


# ═══════════════════════════════════════════════════════
#  Momentum 因子 (Carhart 1997 + BlackRock)
# ═══════════════════════════════════════════════════════

def _compute_momentum(close: np.ndarray, profit_growth: float) -> Tuple[float, float, float]:
    """
    Momentum = (12-1月价格动量 + 风险调整动量 + 盈利动量) / 3

    价格动量: 过去12个月收益, 跳过最近1月
    风险调整: 动量 / 年度波动率
    盈利动量: 净利润增速 YoY
    """
    # 12-1月动量: 最少需要60天, 不足则用更短窗口
    lookback = min(252, len(close) - 21)
    if lookback >= 60 and len(close) > 21:
        mom_12m1m = close[-21] / close[-lookback-21] - 1 if close[-lookback-21] > 0 else np.nan
    elif len(close) >= 42:
        half = len(close) // 2
        mom_12m1m = close[-1] / close[-half] - 1 if close[-half] > 0 else np.nan
    else:
        mom_12m1m = np.nan

    # 风险调整: 动量 / 波动率
    if not np.isnan(mom_12m1m) and len(close) >= 42:
        rets = np.diff(close[-min(252, len(close)):])
        rets = rets / np.maximum(np.abs(close[-min(252, len(close)):-1]), 1e-8)
        vol = np.nanstd(rets) * np.sqrt(252) if len(rets) > 0 else np.nan
        risk_adj_mom = mom_12m1m / vol if vol and vol > 0 else np.nan
    else:
        risk_adj_mom = np.nan

    # 盈利动量
    earn_mom = profit_growth / 100.0 if profit_growth is not None and not np.isnan(profit_growth) else np.nan

    return mom_12m1m, risk_adj_mom, earn_mom


# ═══════════════════════════════════════════════════════
#  Quality 因子 (AQR QMJ + BlackRock + MSCI CNE6)
# ═══════════════════════════════════════════════════════

def _compute_quality(roe: float, gross_margin: float, accrual: float,
                     debt_ratio: float, roe_stability: float) -> Tuple[float, float, float, float, float]:
    """
    Quality = (ROE + 毛利率 + 应计利润(取反) + 杠杆率(取反) + 盈利稳定性) / 5

    Sloan (1996): 高应计利润 → 低盈利质量
    Novy-Marx (2013): 毛利率/总资产 预测力强于 ROE
    """
    q_roe = roe / 100.0 if roe is not None and not np.isnan(roe) else np.nan
    q_gross = gross_margin / 100.0 if gross_margin is not None and not np.isnan(gross_margin) else np.nan
    q_accrual = -abs(accrual) if accrual is not None and not np.isnan(accrual) else np.nan  # 取反
    q_leverage = -(debt_ratio / 100.0) if debt_ratio is not None and not np.isnan(debt_ratio) else np.nan  # 取反
    q_stability = -roe_stability if roe_stability is not None and not np.isnan(roe_stability) else np.nan  # 取反

    return q_roe, q_gross, q_accrual, q_leverage, q_stability


# ═══════════════════════════════════════════════════════
#  Low Volatility 因子 (Ang et al.2006 + MSCI Barra)
# ═══════════════════════════════════════════════════════

def _compute_lowvol(close: np.ndarray) -> Tuple[float, float, float]:
    """
    LowVol = (历史波动率(取反) + Beta(取反) + 残差波动率(取反)) / 3
    """
    window = min(252, len(close))
    if window < 20:
        return np.nan, np.nan, np.nan

    rets = np.diff(close[-window:]) / np.maximum(np.abs(close[-window:-1]), 1e-8)
    rets = rets[~np.isnan(rets)]

    if len(rets) < 20:
        return np.nan, np.nan, np.nan

    # 历史波动率 (年化)
    vol = np.nanstd(rets) * np.sqrt(252)
    low_vol = -vol

    # Beta — 简化: 无市场指数时用 1.0
    low_beta = -1.0

    # 残差波动率
    if len(rets) >= 20:
        rolling_std = pd.Series(rets).rolling(20).std().values
        residual_vol = np.nanstd(rolling_std) if len(rolling_std) > 0 else vol
    else:
        residual_vol = vol
    low_residual = -residual_vol

    return low_vol, low_beta, low_residual


# ═══════════════════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════════════════

def _winsorize(series: pd.Series, n_sigma: float = 3.0) -> pd.Series:
    """3σ 截尾"""
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma):
        return series
    lower = mu - n_sigma * sigma
    upper = mu + n_sigma * sigma
    return series.clip(lower, upper)


def _zscore(series: pd.Series) -> pd.Series:
    """Z-score 标准化"""
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma):
        return pd.Series(0.0, index=series.index)
    return (series - mu) / sigma


def _industry_zscore(df: pd.DataFrame, factor_col: str,
                     industry_map: Dict[str, int]) -> pd.Series:
    """行业内 Z-score"""
    df = df.copy()
    df["_industry"] = df["symbol"].map(industry_map).fillna(-1)
    result = pd.Series(np.nan, index=df.index)

    for ind_code, group in df.groupby("_industry"):
        if len(group) < 3:  # 行业内股票太少, 用全市场
            result.loc[group.index] = _zscore(group[factor_col])
        else:
            result.loc[group.index] = _zscore(group[factor_col])

    return result


# ═══════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════

def compute_four_factors(
    price_df: pd.DataFrame,
    valuation_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
    industry_map: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """
    计算四大因子得分。

    Parameters
    ----------
    price_df : 日线数据 (trade_date, symbol, close, volume)
    valuation_df : 估值数据 (symbol, pe_ttm, pb, total_shares, market_cap)
    financial_df : 财务数据 (symbol, roe, gross_margin, operating_cf,
                             debt_ratio, profit_growth, roe_stability)
    industry_map : {symbol: industry_code}

    Returns
    -------
    DataFrame: symbol, value, momentum, quality, lowvol, composite, rank
    """
    logger.info("=" * 50)
    logger.info("因子计算 v3.0 (MSCI/BlackRock/AQR 四大因子)")

    # 取最新一天的价格数据 (或按日期分组)
    if price_df.empty:
        return pd.DataFrame()

    # 每只股票取最近价格序列用于计算
    results = []
    for symbol, group in price_df.groupby("symbol"):
        group = group.sort_values("trade_date")
        close = group["close"].values

        # ── 估值数据 ──
        val_row = None
        if valuation_df is not None:
            vdf = valuation_df[valuation_df["symbol"] == symbol]
            val_row = vdf.iloc[0] if not vdf.empty else None

        pe = val_row["pe_ttm"] if val_row is not None else None
        pb = val_row["pb"] if val_row is not None else None
        shares = val_row["total_shares"] if val_row is not None else None

        # ── 财务数据 ──
        fin_row = None
        if financial_df is not None:
            fdf = financial_df[financial_df["symbol"] == symbol]
            fin_row = fdf.iloc[0] if not fdf.empty else None

        roe = float(fin_row["roe"]) if fin_row is not None and "roe" in (fin_row.index if hasattr(fin_row, 'index') else []) else None
        profit_growth = float(fin_row.get("profit_growth", np.nan)) if fin_row is not None else None
        debt_ratio = float(fin_row.get("debt_ratio", np.nan)) if fin_row is not None else None
        operating_cf = float(fin_row.get("operating_cf", np.nan)) if fin_row is not None else None
        gross_margin = float(fin_row.get("gross_margin", np.nan)) if fin_row is not None else None
        accrual = float(fin_row.get("accrual", np.nan)) if fin_row is not None else None
        roe_stability = float(fin_row.get("roe_stability", np.nan)) if fin_row is not None else None

        # ── 计算四大因子 ──
        ep, bp, cfp = _compute_value(close, pe, pb, operating_cf, shares)
        mom_p, mom_risk, mom_earn = _compute_momentum(close, profit_growth)
        q_roe, q_gross, q_accrual, q_leverage, q_stability = _compute_quality(
            roe, gross_margin, accrual, debt_ratio, roe_stability
        )
        low_vol, low_beta, low_resid = _compute_lowvol(close)

        results.append({
            "symbol": symbol,
            # Value
            "v_ep": ep, "v_bp": bp, "v_cfp": cfp,
            # Momentum
            "m_price": mom_p, "m_risk_adj": mom_risk, "m_earn": mom_earn,
            # Quality
            "q_roe": q_roe, "q_gross": q_gross, "q_accrual": q_accrual,
            "q_leverage": q_leverage, "q_stability": q_stability,
            # LowVol
            "l_vol": low_vol, "l_beta": low_beta, "l_resid": low_resid,
        })

    df = pd.DataFrame(results)

    if df.empty:
        return df

    # ── 行业中性化 + Winsorize + 合成 ──
    logger.info(f"  原始数据: {len(df)} 只")

    if industry_map:
        # Value = mean(v_ep, v_bp, v_cfp) 行业中性化后
        for sub in ["v_ep", "v_bp", "v_cfp"]:
            if sub in df.columns and df[sub].notna().any():
                df[sub] = _winsorize(df[sub])
                df[sub + "_iz"] = _industry_zscore(df, sub, industry_map)
        v_cols = [c for c in ["v_ep_iz", "v_bp_iz", "v_cfp_iz"] if c in df.columns]
        df["value"] = df[v_cols].mean(axis=1) if v_cols else np.nan

        # Momentum
        for sub in ["m_price", "m_risk_adj", "m_earn"]:
            if sub in df.columns and df[sub].notna().any():
                df[sub] = _winsorize(df[sub])
                df[sub + "_iz"] = _industry_zscore(df, sub, industry_map)
        m_cols = [c for c in ["m_price_iz", "m_risk_adj_iz", "m_earn_iz"] if c in df.columns]
        df["momentum"] = df[m_cols].mean(axis=1) if m_cols else np.nan

        # Quality
        for sub in ["q_roe", "q_gross", "q_accrual", "q_leverage", "q_stability"]:
            if sub in df.columns and df[sub].notna().any():
                df[sub] = _winsorize(df[sub])
                df[sub + "_iz"] = _industry_zscore(df, sub, industry_map)
        q_cols = [c for c in ["q_roe_iz", "q_gross_iz", "q_accrual_iz",
                               "q_leverage_iz", "q_stability_iz"] if c in df.columns]
        df["quality"] = df[q_cols].mean(axis=1) if q_cols else np.nan

        # LowVol
        for sub in ["l_vol", "l_beta", "l_resid"]:
            if sub in df.columns and df[sub].notna().any():
                df[sub] = _winsorize(df[sub])
                df[sub + "_iz"] = _industry_zscore(df, sub, industry_map)
        l_cols = [c for c in ["l_vol_iz", "l_beta_iz", "l_resid_iz"] if c in df.columns]
        df["lowvol"] = df[l_cols].mean(axis=1) if l_cols else np.nan
    else:
        logger.warning("  无行业映射, 使用全市场 Z-score")
        # Fallback: 全市场 Z-score
        for factor, subs in [
            ("value", ["v_ep", "v_bp", "v_cfp"]),
            ("momentum", ["m_price", "m_risk_adj", "m_earn"]),
            ("quality", ["q_roe", "q_gross", "q_accrual", "q_leverage", "q_stability"]),
            ("lowvol", ["l_vol", "l_beta", "l_resid"]),
        ]:
            cols = [c for c in subs if c in df.columns and df[c].notna().any()]
            if cols:
                for c in cols:
                    df[c] = _winsorize(df[c])
                    df[c + "_z"] = _zscore(df[c])
                z_cols = [c + "_z" for c in cols]
                df[factor] = df[z_cols].mean(axis=1)

    # ── 加权总分 ──
    score = pd.Series(0.0, index=df.index)
    for factor, weight in FACTOR_WEIGHTS.items():
        if factor in df.columns and df[factor].notna().any():
            score += df[factor].fillna(0) * weight
    df["composite"] = score

    # ── 过滤 ──
    # 剔除全NaN (无任何因子数据的股票)
    df = df.dropna(subset=["composite"])
    df = df.sort_values("composite", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    n_val = df["value"].notna().sum() if "value" in df.columns else 0
    n_mom = df["momentum"].notna().sum() if "momentum" in df.columns else 0
    n_qual = df["quality"].notna().sum() if "quality" in df.columns else 0
    n_low = df["lowvol"].notna().sum() if "lowvol" in df.columns else 0

    logger.info(f"  完成: {len(df)} 只")
    logger.info(f"  Value: {n_val} | Momentum: {n_mom} | Quality: {n_qual} | LowVol: {n_low}")
    return df
