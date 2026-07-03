"""
A股因子库 (50+因子)

因子分类:
  1. 收益率类 (Returns)      — 多周期涨跌幅
  2. 动量类 (Momentum)        — 路径平滑动量
  3. 波动率类 (Volatility)    — 历史波动/振幅/ATR
  4. 量价类 (Volume-Price)    — 成交量比率/换手/量价背离
  5. 技术指标类 (Technical)   — RSI/MACD/布林/KDJ
  6. 均线类 (MA)              — 均线距离/交叉/排列
  7. 流动性类 (Liquidity)     — Amihud/换手波动
  8. 形态类 (Pattern)         — 缺口/新高新低/连涨连跌
"""
import pandas as pd
import numpy as np
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

# 通用参数
RETURN_PERIODS = [1, 5, 10, 20, 60]
VOL_PERIODS = [5, 10, 20, 60]
MA_PERIODS = [5, 10, 20, 60]
RSI_PERIOD = 14
MACD_PARAMS = (12, 26, 9)
BOLL_PERIOD = 20
ATR_PERIOD = 14


def compute_all_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    对原始日线数据计算全量因子。

    Parameters
    ----------
    df : DataFrame
        必须包含: trade_date, symbol, open, high, low, close, volume, amount, turnover_rate
        可选: pct_change

    Returns
    -------
    DataFrame with symbol, trade_date + all factor columns
    """
    if df.empty:
        return df

    df = df.sort_values(["symbol", "trade_date"]).reset_index(drop=True)

    # 按symbol分组计算 (向量化)
    logger.info(f"计算A股因子: {df['symbol'].nunique()} 只股票, {len(df)} 条记录")

    # 每个symbol单独计算后拼接
    results = []
    for symbol, group in df.groupby("symbol"):
        group = group.copy()
        group = _compute_returns(group)
        group = _compute_momentum(group)
        group = _compute_volatility(group)
        group = _compute_volume_price(group)
        group = _compute_technical(group)
        group = _compute_ma_factors(group)
        group = _compute_liquidity(group)
        group = _compute_pattern(group)
        results.append(group)

    result = pd.concat(results, ignore_index=True)

    # 过大的inf值处理
    result = result.replace([np.inf, -np.inf], np.nan)

    logger.info(f"因子计算完成: {len(_FACTOR_META)} 个因子")
    return result


# ── 因子元数据 (用于文档和特征选择) ──────────────────

_FACTOR_META: dict = {}  # factor_name → {category, description}


def _register(category, name, desc):
    _FACTOR_META[name] = {"category": category, "description": desc}


# ═══════════════════════════════════════════════════════
#  1. 收益率类
# ═══════════════════════════════════════════════════════

def _compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values

    # 如果原始数据有涨跌幅, 使用它; 否则用收盘价算
    if "pct_change" in df.columns:
        ret_1d = df["pct_change"].values / 100.0
    else:
        ret_1d = np.append([np.nan], np.diff(close) / close[:-1])

    df["ret_1d"] = ret_1d
    _register("returns", "ret_1d", "1日收益率")

    for period in RETURN_PERIODS:
        name = f"ret_{period}d"
        df[name] = close / _shift(close, period) - 1
        _register("returns", name, f"{period}日收益率")

    # 对数收益率 (用于统计建模)
    df["log_ret_1d"] = np.log(close / _shift(close, 1))
    _register("returns", "log_ret_1d", "1日对数收益率")

    return df


# ═══════════════════════════════════════════════════════
#  2. 动量类
# ═══════════════════════════════════════════════════════

def _compute_momentum(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values

    # 路径平滑动量 (夏普比率式)
    for period in [10, 20, 60]:
        ret = close / _shift(close, period) - 1
        vol = _rolling_std(df["ret_1d"].values, period) * np.sqrt(period)
        name = f"mom_{period}d"
        df[name] = ret / (vol + 1e-8)
        _register("momentum", name, f"{period}日夏普动量")

    # RSI-momentum (RMI)
    delta = np.diff(close, prepend=np.nan)
    for period in [10, 20]:
        up = _rolling_sum(np.where(delta > 0, delta, 0), period)
        dn = np.abs(_rolling_sum(np.where(delta < 0, delta, 0), period))
        name = f"rmi_{period}d"
        df[name] = up / (up + dn + 1e-8)
        _register("momentum", name, f"{period}日RSI动量")

    # 收益加速度
    df["ret_accel_5_20"] = df.get("ret_5d", _nan_arr(len(close))) - df.get("ret_20d", _nan_arr(len(close)))
    _register("momentum", "ret_accel_5_20", "5日-20日收益差(加速度)")

    # 路径相关 (趋势强度)
    for period in [20, 60]:
        n = min(period, len(close))
        rets = df["ret_1d"].values
        name = f"trend_strength_{period}d"
        df[name] = _rolling_mean(rets, n) / (_rolling_std(rets, n) + 1e-8) * np.sqrt(n)
        _register("momentum", name, f"{period}日趋势强度")

    return df


# ═══════════════════════════════════════════════════════
#  3. 波动率类
# ═══════════════════════════════════════════════════════

def _compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    rets = df["ret_1d"].values

    # 历史波动率 (年化)
    for period in VOL_PERIODS:
        name = f"vol_{period}d"
        df[name] = _rolling_std(rets, period) * np.sqrt(252)
        _register("volatility", name, f"{period}日历史波动率(年化)")

    # ATR (平均真实波幅)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    prev_close = _shift(close, 1)

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        )
    )

    for period in [14, 20]:
        name = f"atr_{period}d"
        df[name] = _rolling_mean(tr, period)
        _register("volatility", name, f"{period}日ATR")

    # ATR% (归一化)
    df["atr_pct_14d"] = df.get("atr_14d", _nan_arr(len(close))) / (close + 1e-8)
    _register("volatility", "atr_pct_14d", "14日ATR百分比")

    # 振幅 (日内)
    df["amplitude_1d"] = (high - low) / (_shift(close, 1) + 1e-8)
    _register("volatility", "amplitude_1d", "1日振幅")

    # 波动率变化
    df["vol_ratio_5_20"] = df.get("vol_5d", _nan_arr(len(close))) / (df.get("vol_20d", _nan_arr(len(close))) + 1e-8)
    _register("volatility", "vol_ratio_5_20", "5日/20日波动率比")

    return df


# ═══════════════════════════════════════════════════════
#  4. 量价类
# ═══════════════════════════════════════════════════════

def _compute_volume_price(df: pd.DataFrame) -> pd.DataFrame:
    volume = df["volume"].values if "volume" in df.columns else _nan_arr(len(df))
    amount = df["amount"].values if "amount" in df.columns else _nan_arr(len(df))
    turnover = df["turnover_rate"].values if "turnover_rate" in df.columns else _nan_arr(len(df))
    close = df["close"].values
    rets = df["ret_1d"].values

    # 成交量比率 (相对N日均量)
    for period in [5, 20, 60]:
        name = f"vol_ratio_{period}d"
        df[name] = volume / (_rolling_mean(volume, period) + 1e-8)
        _register("volume", name, f"{period}日成交量比")

    # 换手率相关
    for period in [5, 20]:
        name = f"turnover_{period}d"
        df[name] = _rolling_mean(turnover, period)
        _register("volume", name, f"{period}日平均换手率")

    # 换手率变化
    df["turnover_change_5d"] = turnover / (_shift(_rolling_mean(turnover, 5), 5) + 1e-8) - 1
    _register("volume", "turnover_change_5d", "换手率5日变化")

    # 量价背离 (量增价跌 = 负信号)
    for period in [5, 10]:
        vol_trend = _rolling_mean(volume, period) / _shift(_rolling_mean(volume, period), period) - 1
        price_trend = df.get(f"ret_{period}d", _nan_arr(len(close)))
        name = f"vp_divergence_{period}d"
        df[name] = vol_trend - price_trend
        _register("volume", name, f"{period}日量价背离")

    # 成交额稳定性
    df["amount_volatility_20d"] = _rolling_std(amount, 20) / (_rolling_mean(amount, 20) + 1e-8)
    _register("volume", "amount_volatility_20d", "20日成交额波动率")

    return df


# ═══════════════════════════════════════════════════════
#  5. 技术指标类
# ═══════════════════════════════════════════════════════

def _compute_technical(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    n = len(close)

    # ── RSI ──
    delta = np.diff(close, prepend=np.nan)
    up, dn = np.where(delta > 0, delta, 0), np.abs(np.where(delta < 0, delta, 0))
    avg_up = _rolling_mean(up, RSI_PERIOD)
    avg_dn = _rolling_mean(dn, RSI_PERIOD)
    df["rsi_14"] = 100 - 100 / (1 + avg_up / (avg_dn + 1e-8))
    _register("technical", "rsi_14", "14日RSI")

    # RSI 超买超卖
    df["rsi_overbought"] = (df["rsi_14"] > 70).astype(float)
    df["rsi_oversold"] = (df["rsi_14"] < 30).astype(float)
    _register("technical", "rsi_overbought", "RSI超买标志")
    _register("technical", "rsi_oversold", "RSI超卖标志")

    # ── MACD ──
    macd_line, signal_line, histogram = _macd(close)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_histogram"] = histogram
    df["macd_signal_cross"] = (macd_line > signal_line).astype(float) - (macd_line < signal_line).astype(float)
    _register("technical", "macd", "MACD线")
    _register("technical", "macd_signal", "MACD信号线")
    _register("technical", "macd_histogram", "MACD柱")
    _register("technical", "macd_signal_cross", "MACD金叉/死叉")

    # ── 布林带 ──
    ma = _rolling_mean(close, BOLL_PERIOD)
    std = _rolling_std(close, BOLL_PERIOD)
    upper = ma + 2 * std
    lower = ma - 2 * std
    df["boll_width"] = (upper - lower) / (ma + 1e-8)
    df["boll_position"] = (close - lower) / (upper - lower + 1e-8)
    _register("technical", "boll_width", "布林带宽度")
    _register("technical", "boll_position", "布林带位置(0-1)")

    # ── KDJ (简化: 只看K) ──
    for period in [9, 14]:
        lowest = _rolling_min(low, period)
        highest = _rolling_max(high, period)
        rsv = (close - lowest) / (highest - lowest + 1e-8) * 100
        name = f"kdj_k_{period}d"
        df[name] = _ema(rsv, 3)
        _register("technical", name, f"{period}日KDJ-K值")

    # ── CCI (商品通道指数) ──
    tp = (high + low + close) / 3
    ma_tp = _rolling_mean(tp, 20)
    md = _rolling_mean(np.abs(tp - ma_tp), 20)
    df["cci_20"] = (tp - ma_tp) / (0.015 * md + 1e-8)
    _register("technical", "cci_20", "20日CCI")

    return df


# ═══════════════════════════════════════════════════════
#  6. 均线类
# ═══════════════════════════════════════════════════════

def _compute_ma_factors(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values

    # 均线距离
    for period in MA_PERIODS:
        ma = _rolling_mean(close, period)
        name = f"ma_dist_{period}d"
        df[name] = close / (ma + 1e-8) - 1
        _register("ma", name, f"距MA{period}距离")

    # 均线交叉信号
    ma5 = _rolling_mean(close, MA_PERIODS[0])
    ma10 = _rolling_mean(close, MA_PERIODS[1])
    ma20 = _rolling_mean(close, MA_PERIODS[2])
    ma60 = _rolling_mean(close, MA_PERIODS[3])

    df["ma_cross_5_20"] = (ma5 > ma20).astype(float)
    df["ma_cross_10_60"] = (ma10 > ma60).astype(float)
    _register("ma", "ma_cross_5_20", "MA5>MA20标志")
    _register("ma", "ma_cross_10_60", "MA10>MA60标志")

    # 均线排列 (多头=全向上, 空头=全向下)
    ma_list = [ma5, ma10, ma20, ma60]
    arrangements = []
    for i in range(len(ma_list) - 1):
        arrangements.append((ma_list[i] > ma_list[i + 1]).astype(int))
    df["ma_bullish_alignment"] = np.sum(arrangements, axis=0) / len(arrangements)
    _register("ma", "ma_bullish_alignment", "均线多头排列度(0-1)")

    # 均线发散/收敛
    df["ma_dispersion"] = np.std([ma5, ma10, ma20, ma60], axis=0) / (close + 1e-8)
    _register("ma", "ma_dispersion", "均线离散度")

    return df


# ═══════════════════════════════════════════════════════
#  7. 流动性类
# ═══════════════════════════════════════════════════════

def _compute_liquidity(df: pd.DataFrame) -> pd.DataFrame:
    rets = df["ret_1d"].values if "ret_1d" in df.columns else _nan_arr(len(df))
    amount = df["amount"].values if "amount" in df.columns else _nan_arr(len(df))
    volume = df["volume"].values if "volume" in df.columns else _nan_arr(len(df))

    # Amihud 非流动性指标
    for period in [5, 20]:
        illiq = np.abs(rets) / (amount + 1e-8) * 1e8
        name = f"amihud_{period}d"
        df[name] = _rolling_mean(illiq, period)
        _register("liquidity", name, f"{period}日Amihud非流动性")

    # 换手率波动
    turnover = df["turnover_rate"].values if "turnover_rate" in df.columns else _nan_arr(len(df))
    df["turnover_vol_20d"] = _rolling_std(turnover, 20)
    _register("liquidity", "turnover_vol_20d", "20日换手率波动")

    return df


# ═══════════════════════════════════════════════════════
#  8. 形态类
# ═══════════════════════════════════════════════════════

def _compute_pattern(df: pd.DataFrame) -> pd.DataFrame:
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values if "open" in df.columns else _nan_arr(len(df))

    n = len(close)

    # N日新高/新低
    for period in [20, 60, 120]:
        df[f"high_{period}d"] = (close >= _rolling_max(_shift(close, 1), period)).astype(float)
        df[f"low_{period}d"] = (close <= _rolling_min(_shift(close, 1), period)).astype(float)
        _register("pattern", f"high_{period}d", f"{period}日新高标志")
        _register("pattern", f"low_{period}d", f"{period}日新低标志")

    # 连涨/连跌天数
    df["consecutive_up"] = _consecutive(close > _shift(close, 1), direction="up")
    df["consecutive_down"] = _consecutive(close < _shift(close, 1), direction="down")
    _register("pattern", "consecutive_up", "连涨天数")
    _register("pattern", "consecutive_down", "连跌天数")

    # 缺口 (今日最低 > 昨日最高 或 今日最高 < 昨日最低)
    prev_high, prev_low = _shift(high, 1), _shift(low, 1)
    df["gap_up"] = (low > prev_high).astype(float)
    df["gap_down"] = (high < prev_low).astype(float)
    _register("pattern", "gap_up", "向上跳空")
    _register("pattern", "gap_down", "向下跳空")

    # 上下影线比例
    body = np.abs(close - open_)
    upper_shadow = high - np.maximum(close, open_)
    lower_shadow = np.minimum(close, open_) - low
    df["upper_shadow_ratio"] = upper_shadow / (body + 1e-8)
    df["lower_shadow_ratio"] = lower_shadow / (body + 1e-8)
    _register("pattern", "upper_shadow_ratio", "上影线比例")
    _register("pattern", "lower_shadow_ratio", "下影线比例")

    return df


# ═══════════════════════════════════════════════════════
#  向量化工具函数
# ═══════════════════════════════════════════════════════

def _shift(arr: np.ndarray, n: int) -> np.ndarray:
    """向量化shift"""
    if n <= 0:
        return arr
    result = np.empty(len(arr), dtype=np.float64)
    result[:n] = np.nan
    result[n:] = arr[:-n]
    return result


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动均值"""
    if len(arr) < window:
        return np.full(len(arr), np.nan, dtype=np.float64)
    # pandas rolling 对边界处理不好, 用 convolve
    kernel = np.ones(window) / window
    result = np.convolve(np.nan_to_num(arr, 0), kernel, mode="same")
    # 边界用NaN
    result = result.astype(np.float64)
    result[:window-1] = np.nan
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差"""
    if len(arr) < window:
        return np.full(len(arr), np.nan, dtype=np.float64)
    return pd.Series(arr).rolling(window, min_periods=window).std().values


def _rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动求和"""
    if len(arr) < window:
        return np.full(len(arr), np.nan, dtype=np.float64)
    return pd.Series(arr).rolling(window, min_periods=window).sum().values


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return np.full(len(arr), np.nan, dtype=np.float64)
    return pd.Series(arr).rolling(window, min_periods=window).max().values


def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < window:
        return np.full(len(arr), np.nan, dtype=np.float64)
    return pd.Series(arr).rolling(window, min_periods=window).min().values


def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均"""
    return pd.Series(arr).ewm(span=period, adjust=False).mean().values


def _macd(close: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (DIF, DEA, 柱)"""
    ema_fast = _ema(close, MACD_PARAMS[0])
    ema_slow = _ema(close, MACD_PARAMS[1])
    dif = ema_fast - ema_slow
    dea = _ema(dif, MACD_PARAMS[2])
    bar = 2 * (dif - dea)
    return dif, dea, bar


def _consecutive(condition: np.ndarray, direction: str = "up") -> np.ndarray:
    """计算连续满足条件的次数"""
    result = np.zeros(len(condition))
    count = 0
    for i in range(len(condition)):
        if condition[i]:
            count += 1
        else:
            count = 0
        result[i] = count
    return result


def _nan_arr(n: int) -> np.ndarray:
    return np.full(n, np.nan)


def get_factor_names() -> List[str]:
    """返回所有因子名称"""
    return sorted(_FACTOR_META.keys())


def get_factor_categories() -> dict:
    """返回 {category: [factor_names]}"""
    cats = {}
    for name, meta in _FACTOR_META.items():
        cats.setdefault(meta["category"], []).append(name)
    return cats


def get_factor_importance() -> pd.DataFrame:
    """返回因子元数据表"""
    return pd.DataFrame([
        {"name": k, "category": v["category"], "description": v["description"]}
        for k, v in _FACTOR_META.items()
    ])
