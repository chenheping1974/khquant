"""
khquant 因子体系 v2.0

设计原则:
  1. 每个因子必须有学术论文支撑或幻方同款方法论
  2. 宁缺毋滥 — 总共18个, 不求数量
  3. 大道至简 — 每个因子独立、可解释、不冗余

框架 (4/3/2/1):
  Phase 1 — 价量因子  (5个, 40%)  Beta/反转/动量/低波/流动性
  Phase 2 — 基本面因子 (6个, 30%)  市值/价值/盈利/投资/情绪/质量
  Phase 3 — 另类因子   (4个, 20%)  北向/两融/龙虎榜/大宗
  Phase 4 — 结构因子   (3个, 10%)  行业/指数成分/宏观

论文来源:
  - Liu, Stambaugh & Yuan (2019, JFE): 中国版因子模型
  - Gharghori & Nguyen (2025, Pacific-Basin FJ): A股因子验证
  - Li, Liu, Liu, Wei (2024, Mgmt Science): 469异象A股复刻
  - 幻方量化公开资料 (2024-2025): 方法论参考

用法:
    from src.features.factors import compute_all_factors, get_factor_names
    df = compute_all_factors(price_df, valuation_df, financial_df)
"""
import logging
from typing import Optional, List, Dict, Tuple
from datetime import date

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  Phase 1: 价量因子 (5个)
#  来源: CAPM / Jegadeesh-Titman / Carhart / Ang et al. / Amihud
# ═══════════════════════════════════════════════════════

def compute_price_volume_factors(close: np.ndarray, volume: np.ndarray,
                                  amount: np.ndarray, market_close: Optional[np.ndarray] = None,
                                  n: int = 252) -> Dict[str, np.ndarray]:
    """
    Phase 1: 5个价量因子, 全部从日线OHLCV计算。

    Parameters
    ----------
    close, volume, amount : 价格/成交量/成交额序列
    market_close : 市场指数收盘价 (用于Beta计算)
    n : 回溯天数

    Returns dict of factor_name → array
    """
    factors = {}

    # ── 1. Beta (Sharpe 1964, CAPM) ──
    # Cov(r_i, r_m) / Var(r_m), 252日滚动
    ret = _returns(close)
    if market_close is not None:
        mkt_ret = _returns(market_close)
        factors["beta"] = _rolling_beta(ret, mkt_ret, window=n)
    else:
        # 无市场数据时, 用全样本平均beta=1
        factors["beta"] = np.ones_like(ret)

    # ── 2. 短期反转 (Jegadeesh 1990, Lehmann 1990) ──
    # 过去1个月(21天)收益, 预期负相关
    factors["short_reversal"] = _rolling_return(close, 21)

    # ── 3. 动量 12-1月 (Jegadeesh & Titman 1993, Carhart 1997) ──
    # 过去12个月收益, 跳过最近1个月
    # ret_12_1 = close[-21] / close[-252] - 1
    factors["momentum_12m1m"] = _momentum_12m1m(close)

    # ── 4. 低波动 (Ang, Hodrick, Xing, Zhang 2006, JOF) ──
    # 252日历史波动率 (年化)
    factors["low_volatility"] = -_rolling_std(ret, n) * np.sqrt(252)  # 负号: 低波=高因子值

    # ── 5. Amihud非流动性 (Amihud 2002, JFM) ──
    # 日度 |return| / dollar_volume, 20日平均
    # 非流动性越高的股票, 预期收益越高 (流动性溢价)
    dollar_vol = np.maximum(amount, 1)  # 避免除零
    illiq = np.abs(ret) / (dollar_vol / 1e8)  # 标准化
    factors["amihud"] = _rolling_mean(illiq, 20)

    return factors


# ═══════════════════════════════════════════════════════
#  Phase 2: 基本面因子 (6个)
#  来源: LSY 2019 / FF5 2015 / 幻方方法论
# ═══════════════════════════════════════════════════════

def compute_fundamental_factors(valuation_df: pd.DataFrame,
                                 financial_df: Optional[pd.DataFrame] = None
                                 ) -> pd.DataFrame:
    """
    Phase 2: 6个基本面因子。

    需要 akshare 数据:
      - stock_zh_a_spot_em(): PE, PB, market_cap
      - stock_financial_analysis_indicator(): ROE, 利润增速, 资产负债率

    Returns DataFrame with symbol + factor columns
    """
    result = valuation_df[["symbol"]].copy() if "symbol" in valuation_df.columns else pd.DataFrame()

    # ── 1. 规模因子 Size (LSY 2019, JFE) ──
    # log(市值), 训练时剔除最小30%
    if "market_cap" in valuation_df.columns:
        mc = valuation_df["market_cap"].copy()
        mc = pd.to_numeric(mc, errors="coerce")
        result["size"] = np.log(mc.clip(lower=1e8))  # 最低1亿
        # 标记最小30% (LSY方法)
        cutoff = mc.quantile(0.30)
        result["is_smallest_30pct"] = (mc < cutoff).astype(int)
    else:
        result["size"] = np.nan

    # ── 2. 价值因子 Value = E/P (LSY 2019, JFE) ──
    # 注意: LSY验证了A股用E/P而非B/P!
    # E/P = 1/PE = 盈利收益率
    if "pe_ttm" in valuation_df.columns:
        pe = pd.to_numeric(valuation_df["pe_ttm"], errors="coerce")
        # 负PE无意义, 设NaN
        pe = pe.clip(lower=1.0)  # 最低PE=1
        result["value_ep"] = 1.0 / pe  # E/P
    else:
        result["value_ep"] = np.nan

    # ── 3. 盈利因子 Profitability = ROE (FF5 2015 + LSY验证) ──
    if financial_df is not None and "roe" in financial_df.columns:
        result["roe"] = pd.to_numeric(financial_df["roe"], errors="coerce") / 100
    else:
        result["roe"] = np.nan

    # ── 4. 投资因子 Investment (FF5 2015, A股反向!) ──
    # 资产增长率 YoY, A股呈负溢价 (Li & Chen 2022)
    if financial_df is not None and "asset_growth" in financial_df.columns:
        result["investment"] = pd.to_numeric(financial_df["asset_growth"], errors="coerce")
    else:
        result["investment"] = np.nan

    # ── 5. 情绪代理 Sentiment (LSY PMO因子代理) ──
    # LSY用换手率作为误定价/情绪代理
    # 我们用 turnover_rate 作为替代
    if "turnover_rate" in valuation_df.columns:
        result["sentiment_proxy"] = pd.to_numeric(valuation_df["turnover_rate"], errors="coerce")
    else:
        result["sentiment_proxy"] = np.nan

    # ── 6. 质量因子 Quality (Asness et al. 2019, RFS + 幻方方法论) ──
    # 综合: ROE高 + 负债率低 + 盈利稳定
    score_cols = []
    if "roe" in result.columns:
        roe_z = _zscore(result["roe"])
        score_cols.append(roe_z)
    if financial_df is not None and "debt_ratio" in financial_df.columns:
        dr = pd.to_numeric(financial_df["debt_ratio"], errors="coerce")
        debt_z = -_zscore(dr)  # 负号: 低负债=高得分
        score_cols.append(debt_z)
    if score_cols:
        result["quality"] = pd.concat(score_cols, axis=1).mean(axis=1)
    else:
        result["quality"] = np.nan

    return result


# ═══════════════════════════════════════════════════════
#  Phase 3: 另类数据因子 (4个)
#  来源: 幻方方法论 + A股实证
#  数据: akshare (北向资金/两融/龙虎榜/大宗交易)
# ═══════════════════════════════════════════════════════

def compute_alternative_factors(price_df: pd.DataFrame,
                                  north_flow_df: Optional[pd.DataFrame] = None,
                                  margin_df: Optional[pd.DataFrame] = None,
                                  ) -> pd.DataFrame:
    """
    Phase 3: 4个另类数据因子。

    数据源 (akshare):
      - stock_hsgt_hist_em(): 北向资金
      - stock_margin_detail_szse/sse(): 融资融券
      - stock_dzjy_mrmx(): 大宗交易

    当前: 数据暂不可用, 返回NaN列, 不影响训练
    """
    result = price_df[["trade_date", "symbol"]].copy()

    # ── 1. 北向资金净流入变化 ──
    result["north_flow"] = np.nan

    # ── 2. 融资余额变化 ──
    result["margin_change"] = np.nan

    # ── 3. 龙虎榜异常 ──
    result["dragon_tiger"] = np.nan

    # ── 4. 大宗交易折溢价 ──
    result["block_trade_premium"] = np.nan

    # TODO: 填充实际数据 (akshare恢复后)
    # 当前返回NaN, 训练时会自动跳过

    return result


# ═══════════════════════════════════════════════════════
#  Phase 4: 结构因子 (3个)
#  来源: 幻方"其他"类别
# ═══════════════════════════════════════════════════════

def compute_structure_factors(price_df: pd.DataFrame,
                               industry_df: Optional[pd.DataFrame] = None,
                               ) -> pd.DataFrame:
    """
    Phase 4: 1个结构因子。

    行业虚拟变量 — 量化共识, 控制行业固定效应, 非Alpha因子。

    数据源:
      - akshare stock_board_industry_name_em(): 申万行业分类
    """
    result = price_df[["trade_date", "symbol"]].copy()

    # ── 行业代码 (行业中性化控制变量) ──
    result["industry_code"] = np.nan

    # TODO: 填充实际行业分类数据 (akshare恢复后)

    return result


# ═══════════════════════════════════════════════════════
#  统计算
# ═══════════════════════════════════════════════════════

def compute_all_factors(
    price_df: pd.DataFrame,
    valuation_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
    market_index_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    计算全部18个因子 (4 Phases)。

    Returns DataFrame with trade_date, symbol, close + all factor columns
    """
    logger.info("=" * 50)
    logger.info("khquant 因子计算 v2.0 (4/3/2/1 框架)")

    df = price_df.sort_values(["symbol", "trade_date"]).copy()
    results = []

    for symbol, group in df.groupby("symbol"):
        group = group.sort_values("trade_date")
        close = group["close"].values
        volume = group["volume"].values if "volume" in group.columns else np.ones(len(close))
        amount = group.get("amount", pd.Series(close * volume)).values

        # Phase 1: 价量 (5因子, 从K线计算)
        pv = compute_price_volume_factors(close, volume, amount)
        for k, v in pv.items():
            group[k] = v

        results.append(group)

    result = pd.concat(results, ignore_index=True)

    # Phase 2: 基本面 (6因子, merge估值+财务)
    if valuation_df is not None and not valuation_df.empty:
        fund = compute_fundamental_factors(valuation_df, financial_df)
        fund_cols = [c for c in fund.columns
                     if c not in ("symbol",) and not c.startswith("is_")]
        if fund_cols:
            result = result.merge(
                fund[["symbol"] + fund_cols], on="symbol", how="left"
            )
        n_fund = len([c for c in fund_cols if c in result.columns])
        logger.info(f"  Phase 2 基本面: {n_fund}个")
    else:
        logger.warning("  Phase 2 基本面: 无数据, 跳过")

    # Phase 3: 另类 (4因子)
    alt = compute_alternative_factors(result)
    alt_cols = [c for c in alt.columns if c not in ("trade_date", "symbol")]
    if alt_cols:
        result = result.merge(alt, on=["trade_date", "symbol"], how="left")
    logger.info(f"  Phase 3 另类: {sum(1 for c in alt_cols if result[c].notna().any())}个可用")

    # Phase 4: 结构 (3因子)
    struct = compute_structure_factors(result)
    struct_cols = [c for c in struct.columns if c not in ("trade_date", "symbol")]
    if struct_cols:
        result = result.merge(struct, on=["trade_date", "symbol"], how="left")
    logger.info(f"  Phase 4 结构: {sum(1 for c in struct_cols if result[c].notna().any())}个可用")

    # 清洗
    result = result.replace([np.inf, -np.inf], np.nan)

    # 统计
    factor_cols = _get_all_factor_columns(result)
    n_available = sum(1 for c in factor_cols if result[c].notna().any())
    logger.info(f"完成: {n_available}/{len(factor_cols)}个因子可用, {len(result)}行")

    return result


def get_factor_names() -> List[str]:
    """返回所有16个因子名称 (不含标记列)"""
    return [
        # Phase 1: 价量 (5)
        "beta", "short_reversal", "momentum_12m1m",
        "low_volatility", "amihud",
        # Phase 2: 基本面 (6)
        "size", "value_ep", "roe",
        "investment", "sentiment_proxy", "quality",
        # Phase 3: 另类 (4)
        "north_flow", "margin_change",
        "dragon_tiger", "block_trade_premium",
        # Phase 4: 结构 (1)
        "industry_code",
    ]


def _get_all_factor_columns(df: pd.DataFrame) -> List[str]:
    """从DataFrame提取实际存在的因子列"""
    all_names = get_factor_names()
    return [c for c in all_names if c in df.columns]


# ═══════════════════════════════════════════════════════
#  向量化工具
# ═══════════════════════════════════════════════════════

def _returns(close: np.ndarray) -> np.ndarray:
    r = np.diff(close) / np.maximum(close[:-1], 1e-8)
    return np.append([0], r)

def _rolling_return(c: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(c), np.nan, dtype=np.float64)
    out[n:] = c[n:] / np.maximum(c[:-n], 1e-8) - 1
    return out

def _rolling_mean(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) < n:
        return np.full(len(a), np.nan)
    kernel = np.ones(n) / n
    out = np.convolve(np.nan_to_num(a, 0), kernel, mode="same")
    out[:n-1] = np.nan
    return out

def _rolling_std(a: np.ndarray, n: int) -> np.ndarray:
    if len(a) < n:
        return np.full(len(a), np.nan)
    return pd.Series(a).rolling(n, min_periods=n).std().values

def _rolling_beta(r_i: np.ndarray, r_m: np.ndarray, window: int) -> np.ndarray:
    """滚动Beta = Cov(r_i, r_m) / Var(r_m)"""
    out = np.full(len(r_i), np.nan, dtype=np.float64)
    for t in range(window, len(r_i)):
        ri = r_i[t-window:t]
        rm = r_m[t-window:t]
        mask = ~(np.isnan(ri) | np.isnan(rm))
        if mask.sum() < window // 2:
            continue
        cov = np.cov(ri[mask], rm[mask])[0, 1]
        var = np.var(rm[mask])
        out[t] = cov / var if var > 1e-8 else 1.0
    return out

def _momentum_12m1m(close: np.ndarray) -> np.ndarray:
    """12-1月动量: ret between t-21 and t-252"""
    out = np.full(len(close), np.nan, dtype=np.float64)
    for t in range(252, len(close)):
        if close[t-21] > 0:
            out[t] = close[t-21] / close[t-252] - 1
    return out

def _zscore(s: pd.Series) -> pd.Series:
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / std
