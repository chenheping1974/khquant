"""
金银宏观+技术因子库

因子分类:
  1. 宏观驱动层 — DXY, 实际利率, VIX, 汇率, ETF持仓
  2. 技术指标层 — 趋势/动量/波动/量价 (复用A股因子基础设施)
  3. 跨资产层 — 金铜比, 金银比, 金油比
  4. 持仓情绪层 — ETF流入流出, CFTC持仓变化
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# 宏观因子参数
MACRO_LOOKBACK = {
    "dxy_change": [1, 5, 10, 20],
    "real_yield_change": [1, 5, 10, 20],
    "vix_level": [1, 5, 10],
    "vix_change": [1, 5],
    "usdcny_change": [1, 5, 10],
    "etf_flow": [5, 20],
}


def compute_all_factors(
    gold_silver_df: pd.DataFrame,
    macro_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    融合金银价格数据 + 宏观数据 → 金银因子表。

    Parameters
    ----------
    gold_silver_df : DataFrame
        黄金白银日线, 包含: trade_date, symbol, close, (open/high/low/volume 可选)
    macro_df : DataFrame
        宏观指标, 包含: trade_date, symbol, close

    Returns
    -------
    DataFrame with trade_date + factor columns per symbol
    """
    if gold_silver_df.empty:
        logger.warning("黄金白银数据为空, 跳过因子计算")
        return pd.DataFrame()

    logger.info("计算金银因子...")

    # 1. 准备价格数据: 按symbol拆分
    price_data = {}
    for symbol, group in gold_silver_df.groupby("symbol"):
        group = group.sort_values("trade_date")
        price_data[symbol] = group.set_index("trade_date")["close"]

    # 2. 准备宏观数据: 按indicator拆分
    macro_data = {}
    if not macro_df.empty:
        for indicator, group in macro_df.groupby("symbol"):
            group = group.sort_values("trade_date")
            macro_data[indicator] = group.set_index("trade_date")["close"]

    # 3. 对每个品种计算因子
    all_factors = []
    for symbol, price_series in price_data.items():
        if price_series.empty:
            continue
        factors = _compute_single_asset(price_series, macro_data, symbol)
        if factors is not None and not factors.empty:
            factors["symbol"] = symbol
            all_factors.append(factors)

    if not all_factors:
        return pd.DataFrame()

    result = pd.concat(all_factors, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(f"金银因子计算完成: {len(result)} 条记录")
    return result


def _compute_single_asset(
    price: pd.Series,
    macro: Dict[str, pd.Series],
    symbol: str,
) -> pd.DataFrame:
    """对单个品种计算全量因子"""
    df = pd.DataFrame({"trade_date": price.index, "close": price.values})
    df = df.sort_values("trade_date")

    close = df["close"].values
    n = len(close)

    # ── 1. 宏观驱动层 ──
    # DXY变化
    if "DXY" in macro:
        dxy = _align_series(df["trade_date"], macro["DXY"])
        for p in MACRO_LOOKBACK["dxy_change"]:
            df[f"dxy_chg_{p}d"] = dxy / _shift(dxy, p) - 1
            _meta("macro", f"dxy_chg_{p}d", f"DXY {p}日变化")

    # 实际利率代理 (TIPS ETF变化, 取反: 利率↓→利好黄金)
    if "TIPS" in macro:
        tips = _align_series(df["trade_date"], macro["TIPS"])
        for p in MACRO_LOOKBACK["real_yield_change"]:
            df[f"real_yield_chg_{p}d"] = -(tips / _shift(tips, p) - 1)
            _meta("macro", f"real_yield_chg_{p}d", f"实际利率{p}日变化(取反)")

    # VIX 水平 & 变化
    if "VIX" in macro:
        vix = _align_series(df["trade_date"], macro["VIX"])
        df["vix_level"] = vix
        df["vix_percentile_60d"] = _rolling_percentile(vix, 60)
        for p in MACRO_LOOKBACK["vix_change"]:
            df[f"vix_chg_{p}d"] = vix / _shift(vix, p) - 1
        _meta("macro", "vix_level", "VIX绝对水平")
        _meta("macro", "vix_percentile_60d", "VIX 60日分位数")

    # 美元人民币汇率
    if "USDCNY" in macro:
        cny = _align_series(df["trade_date"], macro["USDCNY"])
        for p in MACRO_LOOKBACK["usdcny_change"]:
            df[f"cny_chg_{p}d"] = cny / _shift(cny, p) - 1
            _meta("macro", f"cny_chg_{p}d", f"人民币{p}日变化")

    # 美债10Y收益率变化
    if "US10Y" in macro:
        us10y = _align_series(df["trade_date"], macro["US10Y"])
        df["us10y_level"] = us10y
        df["us10y_chg_5d"] = us10y - _shift(us10y, 5)
        df["us10y_chg_20d"] = us10y - _shift(us10y, 20)
        _meta("macro", "us10y_level", "10Y美债收益率")
        _meta("macro", "us10y_chg_5d", "10Y美债5日变化")
        _meta("macro", "us10y_chg_20d", "10Y美债20日变化")

    # ETF持仓变化
    etf_ticker = "GLD" if "XAU" in symbol.upper() or "AU" in symbol.upper() else "SLV"
    if etf_ticker in macro:
        etf = _align_series(df["trade_date"], macro[etf_ticker])
        for p in MACRO_LOOKBACK["etf_flow"]:
            df[f"etf_flow_{p}d"] = etf / _shift(etf, p) - 1
            _meta("sentiment", f"etf_flow_{p}d", f"{etf_ticker} ETF {p}日持仓变化")

    # ── 2. 技术指标层 (复用A股逻辑, 简化版) ──
    df["ret_1d"] = close / _shift(close, 1) - 1

    for p in [5, 10, 20, 60]:
        df[f"ret_{p}d"] = close / _shift(close, p) - 1
        _meta("technical", f"ret_{p}d", f"{p}日收益率")

    # 波动率
    for p in [5, 10, 20]:
        df[f"vol_{p}d"] = _rolling_std(df["ret_1d"].values, p) * np.sqrt(252)
        _meta("technical", f"vol_{p}d", f"{p}日波动率(年化)")

    # ATR
    if "high" in df.columns and "low" in df.columns:
        tr = np.maximum(
            df["high"].values - df["low"].values,
            np.abs(df["high"].values - _shift(close, 1))
        )
        df["atr_14d"] = _rolling_mean(tr, 14)
        df["atr_pct_14d"] = df["atr_14d"] / (close + 1e-8)
        _meta("technical", "atr_14d", "14日ATR")
        _meta("technical", "atr_pct_14d", "ATR百分比")

    # RSI
    delta = np.diff(close, prepend=np.nan)
    up, dn = np.where(delta > 0, delta, 0), np.abs(np.where(delta < 0, delta, 0))
    avg_up = _rolling_mean(up, 14)
    avg_dn = _rolling_mean(dn, 14)
    df["rsi_14"] = 100 - 100 / (1 + avg_up / (avg_dn + 1e-8))
    _meta("technical", "rsi_14", "14日RSI")

    # MACD
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    dif = ema12 - ema26
    dea = _ema(dif, 9)
    df["macd_histogram"] = 2 * (dif - dea)
    df["macd_signal_cross"] = (dif > dea).astype(float)
    _meta("technical", "macd_histogram", "MACD柱")
    _meta("technical", "macd_signal_cross", "MACD金叉死叉")

    # 均线距离
    for p in [5, 10, 20]:
        ma = _rolling_mean(close, p)
        df[f"ma_dist_{p}d"] = close / (ma + 1e-8) - 1
        _meta("technical", f"ma_dist_{p}d", f"距MA{p}距离")

    # 布林带位置
    ma20 = _rolling_mean(close, 20)
    std20 = _rolling_std(close, 20)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    df["boll_position"] = (close - lower) / (upper - lower + 1e-8)
    _meta("technical", "boll_position", "布林带位置")

    # ── 3. 跨资产层 ──
    # (需要外部品种数据, Phase 1 只做金银比)
    # 金银比相关性会在训练阶段通过多品种因子自动学习

    # ── 4. 持仓情绪层 ──
    # ETF持仓趋势
    if etf_ticker in macro:
        etf = _align_series(df["trade_date"], macro[etf_ticker])
        df["etf_trend_20d"] = _rolling_mean(etf / _shift(etf, 1) - 1, 20)
        _meta("sentiment", "etf_trend_20d", "ETF 20日持仓趋势")

    return df


# ═══════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════

def _shift(arr: np.ndarray, n: int) -> np.ndarray:
    if n <= 0:
        return arr
    result = np.empty_like(arr)
    result[:n] = np.nan
    result[n:] = arr[:-n]
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).mean().values


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).std().values


def _rolling_percentile(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动百分位排名"""
    return pd.Series(arr).rolling(window, min_periods=window).apply(
        lambda x: (x <= x.iloc[-1]).mean(), raw=False
    ).values


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values


def _align_series(dates: pd.Series, series: pd.Series) -> np.ndarray:
    """将宏观序列对齐到价格日期"""
    aligned = series.reindex(dates, method="ffill")
    return aligned.values


# ── 因子元数据 ──

_META: dict = {}

def _meta(category, name, desc):
    _META[name] = {"category": category, "description": desc}


def get_factor_names() -> List[str]:
    return sorted(_META.keys())


def get_factor_categories() -> dict:
    cats = {}
    for name, meta in _META.items():
        cats.setdefault(meta["category"], []).append(name)
    return cats
