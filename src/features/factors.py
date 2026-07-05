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
                                 financial_df: Optional[pd.DataFrame] = None,
                                 price_df: Optional[pd.DataFrame] = None,
                                 ) -> pd.DataFrame:
    """
    Phase 2: 6个基本面因子。

    数据源:
      - 腾讯财经: PE, PB, market_cap (日频)
      - akshare财报: ROE, 利润增速, 资产负债率, 资产增长率 (季度→前向填充到日频)

    Returns DataFrame with symbol [+ trade_date] + factor columns
    """
    has_dates = "trade_date" in valuation_df.columns
    key_cols = ["symbol"] if not has_dates else ["trade_date", "symbol"]
    result = valuation_df[key_cols].copy()

    # ── 1. 规模因子 Size (LSY 2019) ──
    if "market_cap" in valuation_df.columns:
        mc = pd.to_numeric(valuation_df["market_cap"], errors="coerce")
        result["size"] = np.log(mc.clip(lower=1e8))
        cutoff = mc.quantile(0.30)
        result["is_smallest_30pct"] = (mc < cutoff).astype(int)
    else:
        result["size"] = np.nan

    # ── 2. 价值因子 Value = E/P (LSY 2019) ──
    if "pe_ttm" in valuation_df.columns:
        pe = pd.to_numeric(valuation_df["pe_ttm"], errors="coerce")
        pe = pe.clip(lower=1.0)
        result["value_ep"] = 1.0 / pe
    else:
        result["value_ep"] = np.nan

    # ── 3-5. 财务质量因子 (从 akshare 财务数据) ──
    if financial_df is not None and not financial_df.empty:
        fin_cols = [c for c in ["roe", "profit_growth", "debt_ratio", "asset_growth"]
                    if c in financial_df.columns]
        if fin_cols:
            if has_dates:
                result = result.merge(
                    financial_df[["trade_date", "symbol"] + fin_cols],
                    on=["trade_date", "symbol"], how="left"
                )
            else:
                result = result.merge(
                    financial_df[["symbol"] + fin_cols].drop_duplicates(subset=["symbol"], keep="last"),
                    on="symbol", how="left"
                )

    # 标准化因子名
    if "roe" not in result.columns:
        result["roe"] = np.nan
    if "asset_growth" not in result.columns:
        result["investment"] = np.nan
    else:
        result["investment"] = pd.to_numeric(result["asset_growth"], errors="coerce")
        result.drop(columns=["asset_growth"], inplace=True, errors="ignore")
    if "debt_ratio" not in result.columns:
        result["debt_ratio"] = np.nan

    # ── 5. 情绪代理 = turnover_rate (LSY PMO) ──
    # 优先用腾讯的换手率, 否则稍后在 compute_all_factors 中从volume/shares计算
    if "turnover_rate" in valuation_df.columns:
        result["sentiment_proxy"] = pd.to_numeric(valuation_df["turnover_rate"], errors="coerce")
    # 不设 NaN — 让 compute_all_factors 中的 volume/shares 逻辑接管

    # ── 6. 质量因子 Quality (Asness et al. 2019) ──
    if "roe" in result.columns and result["roe"].notna().any():
        roe_z = _zscore(result["roe"].fillna(0))
        debt_z = -_zscore(result["debt_ratio"].fillna(50)) if "debt_ratio" in result.columns else 0
        result["quality"] = (roe_z + debt_z) / 2
    else:
        result["quality"] = np.nan

    return result


# ═══════════════════════════════════════════════════════
#  Phase 3: 另类数据因子 (4个)
#  来源: 幻方方法论 + A股实证
#  数据: akshare (北向资金/两融/龙虎榜/大宗交易)
# ═══════════════════════════════════════════════════════

def compute_alternative_factors(price_df: pd.DataFrame,
                                  north_flow_raw: Optional[pd.DataFrame] = None,
                                  margin_raw: Optional[pd.DataFrame] = None,
                                  dragon_tiger_raw: Optional[pd.DataFrame] = None,
                                  ) -> pd.DataFrame:
    """
    Phase 3: 4个另类数据因子。

    数据源 (all verified ✅):
      - akshare stock_hsgt_hist_em(): 北向资金
      - akshare stock_margin_underlying_info_szse(): 融资融券
      - akshare stock_lhb_detail_em(): 龙虎榜
    """
    result = price_df[["trade_date", "symbol"]].copy()
    result["north_flow"] = np.nan
    result["margin_change"] = np.nan
    result["dragon_tiger"] = np.nan
    result["block_trade_premium"] = np.nan

    # ── 1. 北向资金净流入 (广发/国金 研报, 2024) ──
    # 全市场级别宏观信号: 北向流入→利好所有A股
    if north_flow_raw is not None and not north_flow_raw.empty:
        nf = north_flow_raw[["trade_date", "north_flow_5d"]].copy()
        nf["trade_date"] = nf["trade_date"].astype(str)
        result["trade_date_str"] = result["trade_date"].astype(str)
        result = result.merge(nf, left_on="trade_date_str", right_on="trade_date",
                              how="left", suffixes=("", "_nf"))
        result["north_flow"] = result["north_flow_5d"].fillna(0)
        result.drop(columns=["trade_date_str", "trade_date_nf",
                      "north_flow_5d"], inplace=True, errors="ignore")

    # ── 2. 融资融券标的 (广发/聚源 研报, 2024) ──
    # 融资买入占融资余额比 RankIC -5.65% (中证500)
    if margin_raw is not None and not margin_raw.empty:
        margin_symbols = set(
            margin_raw["证券代码"].astype(str).str.zfill(6).tolist()
        )
        result["margin_change"] = result["symbol"].apply(
            lambda s: 1.0 if str(s).zfill(6) in margin_symbols else 0.0
        )
        # 标记非两融标的为NaN (避免用0污染训练)
        result.loc[~result["symbol"].isin(margin_symbols), "margin_change"] = np.nan

    # ── 3. 龙虎榜净买额 (开源证券 研报, 2022) ──
    if dragon_tiger_raw is not None and not dragon_tiger_raw.empty:
        dt = dragon_tiger_raw.copy()
        if "代码" in dt.columns and "上榜日" in dt.columns:
            dt["symbol"] = dt["代码"].astype(str).str.zfill(6)
            dt["trade_date"] = pd.to_datetime(dt["上榜日"], errors="coerce").dt.date
            # 净买额 (万元) → 标准化信号
            if "龙虎榜净买额" in dt.columns:
                dt["net_buy"] = pd.to_numeric(dt["龙虎榜净买额"], errors="coerce")
                dt = dt[["trade_date", "symbol", "net_buy"]].dropna()
                result = result.merge(dt, on=["trade_date", "symbol"], how="left")
                result["dragon_tiger"] = result["net_buy"].fillna(0)
                result.drop(columns=["net_buy"], inplace=True, errors="ignore")

    # ── 4. 大宗交易折溢价 (光大证券 研报, 2023) ──
    # stock_dzjy_hygtj: 个股大宗交易统计, 折溢率有预测力
    try:
        import akshare as ak
        dzjy = ak.stock_dzjy_hygtj()
        if dzjy is not None and not dzjy.empty:
            dzjy["symbol"] = dzjy["证券代码"].astype(str).str.zfill(6)
            dzjy["bt_premium"] = pd.to_numeric(dzjy["折溢率"], errors="coerce")
            dzjy = dzjy[["symbol", "bt_premium"]].dropna()
            result = result.merge(dzjy, on="symbol", how="left")
            result["block_trade_premium"] = result["bt_premium"].fillna(0)
            result.drop(columns=["bt_premium"], inplace=True, errors="ignore")
    except Exception:
        result["block_trade_premium"] = np.nan

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

    行业分类 — Eastmoney F10 API (免费, 稳定)

    数据源:
      - emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey
        返回 jbzl.sshy (申万行业) + jbzl.sszjhhy (证监会行业)
    """
    result = price_df[["trade_date", "symbol"]].copy()

    # 获取行业映射 (带缓存)
    industry_map = _get_industry_map(result["symbol"].unique().tolist())

    if industry_map:
        result["industry_code"] = result["symbol"].map(industry_map).fillna(-1).astype(int)
    else:
        result["industry_code"] = np.nan

    return result


# ═══════════════════════════════════════════════════════
#  行业分类 — Eastmoney F10 API
# ═══════════════════════════════════════════════════════

_industry_map_cache = None
_INDUSTRY_CACHE_FILE = None  # lazy init

def _get_industry_map(symbols: list) -> dict:
    """获取股票→行业编码映射 (申万行业), 自动缓存到JSON文件"""
    global _industry_map_cache, _INDUSTRY_CACHE_FILE
    import json, requests, time as _time
    from pathlib import Path

    # lazy init cache file path
    if _INDUSTRY_CACHE_FILE is None:
        _INDUSTRY_CACHE_FILE = Path(__file__).parent.parent.parent / ".industry_cache.json"

    # 加载缓存
    if _industry_map_cache is None:
        if _INDUSTRY_CACHE_FILE.exists():
            try:
                raw = json.loads(_INDUSTRY_CACHE_FILE.read_text())
                # json keys are strings → convert to dict
                _industry_map_cache = {str(k): int(v) for k, v in raw.items()}
                logger.info(f"  行业缓存: {len(_industry_map_cache)}只")
            except Exception:
                _industry_map_cache = {}
        else:
            _industry_map_cache = {}

    # 找出缺失的股票
    missing = [s for s in symbols if str(s).zfill(6) not in _industry_map_cache]
    if not missing:
        n = len([s for s in symbols if str(s).zfill(6) in _industry_map_cache])
        logger.info(f"  行业: {n}只 (全部缓存命中)")
        return {str(s).zfill(6): _industry_map_cache[str(s).zfill(6)]
                for s in symbols if str(s).zfill(6) in _industry_map_cache}

    # 只查缺失的 (限制每批最多查500只, 避免太慢)
    to_fetch = missing[:500]
    logger.info(f"  行业: {len(to_fetch)}只待查 (缓存{len(_industry_map_cache)}只)")

    headers = {'User-Agent': 'Mozilla/5.0'}
    industry_names = {}
    # 从已有缓存推断编码
    all_codes_set = set(_industry_map_cache.values())
    next_code = max(all_codes_set) + 1 if all_codes_set else 0

    for i, code in enumerate(to_fetch):
        code_str = str(code).zfill(6)
        prefix = "SH" if code_str.startswith("6") else "SZ"
        url = (f"https://emweb.securities.eastmoney.com/PC_HSF10/"
               f"CompanySurvey/CompanySurveyAjax?code={prefix}{code_str}")
        try:
            r = requests.get(url, headers=headers, timeout=5)
            if r.status_code == 200:
                data = r.json()
                hy = data.get("jbzl", {}).get("sshy", "")
                if hy:
                    if hy not in industry_names:
                        industry_names[hy] = next_code
                        next_code += 1
                    _industry_map_cache[code_str] = industry_names[hy]
        except Exception:
            pass

        if (i + 1) % 100 == 0:
            logger.info(f"    [{i+1}/{len(to_fetch)}]")
            # 每100只保存一次缓存
            _INDUSTRY_CACHE_FILE.write_text(
                json.dumps(_industry_map_cache, ensure_ascii=False)
            )
        _time.sleep(0.15)

    # 最终保存
    _INDUSTRY_CACHE_FILE.write_text(
        json.dumps(_industry_map_cache, ensure_ascii=False)
    )
    logger.info(f"  行业: {len(_industry_map_cache)}只 ({len(industry_names)}个行业)")

    return {str(s).zfill(6): _industry_map_cache[str(s).zfill(6)]
            for s in symbols if str(s).zfill(6) in _industry_map_cache}


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

    # 1.5. sentiment_proxy = 换手率 = volume / total_shares
    if "volume" in result.columns and valuation_df is not None and "total_shares" in valuation_df.columns:
        shares_map = valuation_df.set_index("symbol")["total_shares"].to_dict()
        result["sentiment_proxy"] = result.apply(
            lambda r: r["volume"] / max(shares_map.get(r["symbol"], 1e10), 1)
            if r["symbol"] in shares_map else np.nan, axis=1
        )
    else:
        result["sentiment_proxy"] = np.nan

    # Phase 2: 基本面 (6因子)
    if valuation_df is not None and not valuation_df.empty:
        fund = compute_fundamental_factors(valuation_df, financial_df, result)
        merge_on = ["symbol"] if "trade_date" not in fund.columns else ["trade_date", "symbol"]
        # 排除 result 中已有的列 (避免 merge 产生 _x/_y 后缀)
        existing = set(result.columns)
        fund_cols = [c for c in fund.columns
                     if c not in merge_on and not c.startswith("is_") and c not in existing]
        if fund_cols:
            result = result.merge(
                fund[merge_on + fund_cols], on=merge_on, how="left"
            )
        n_fund_real = len([c for c in fund_cols
                          if c in result.columns and result[c].notna().any()])
        logger.info(f"  Phase 2 基本面: {n_fund_real}个 (共{len(fund_cols)}个候选)")
    else:
        logger.warning("  Phase 2 基本面: 无数据, 跳过")

    # Phase 3: 另类 (4因子)
    try:
        north_flow_raw = _get_north_flow_data()
    except Exception:
        north_flow_raw = None
    try:
        margin_raw = _get_margin_data()
    except Exception:
        margin_raw = None
    try:
        dragon_tiger_raw = _get_dragon_tiger_data()
    except Exception:
        dragon_tiger_raw = None

    alt = compute_alternative_factors(result, north_flow_raw, margin_raw, dragon_tiger_raw)
    alt_cols = [c for c in alt.columns if c not in ("trade_date", "symbol")]
    if alt_cols:
        result = result.merge(alt, on=["trade_date", "symbol"], how="left")
    n_alt = sum(1 for c in alt_cols if c in result.columns and result[c].notna().any())
    logger.info(f"  Phase 3 另类: {n_alt}个可用")

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


# ═══════════════════════════════════════════════════════
#  Phase 3 数据获取 (带缓存)
# ═══════════════════════════════════════════════════════

_north_flow_cache = None
_margin_cache = None
_dragon_tiger_cache = None


def _get_north_flow_data() -> Optional[pd.DataFrame]:
    """北向资金日度净流向 (全市场汇总, 作为宏观信号)"""
    global _north_flow_cache
    if _north_flow_cache is not None:
        return _north_flow_cache
    try:
        import akshare as ak
        df = ak.stock_hsgt_hist_em()
        if df is not None and not df.empty:
            df = df.rename(columns={
                "日期": "trade_date",
                "当日成交净买额": "net_buy",
                "买入成交额": "buy_amount",
                "卖出成交额": "sell_amount",
            })
            df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.date
            df["net_buy"] = pd.to_numeric(df["net_buy"], errors="coerce")
            df = df.dropna(subset=["trade_date", "net_buy"])
            df = df.sort_values("trade_date")
            df["north_flow_5d"] = df["net_buy"].rolling(5, min_periods=1).mean()
            _north_flow_cache = df
            logger.info(f"  北向资金(全市场): {len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"  北向资金失败: {e}")
    return None


def _get_margin_data() -> Optional[pd.DataFrame]:
    """融资融券标的信息"""
    global _margin_cache
    if _margin_cache is not None:
        return _margin_cache
    try:
        import akshare as ak
        df = ak.stock_margin_underlying_info_szse()
        if df is not None and not df.empty:
            _margin_cache = df
            logger.info(f"  两融标的: {len(df)}只")
            return df
    except Exception as e:
        logger.warning(f"  两融数据失败: {e}")
    return None


def _get_dragon_tiger_data() -> Optional[pd.DataFrame]:
    """龙虎榜上榜数据 (近30天)"""
    global _dragon_tiger_cache
    if _dragon_tiger_cache is not None:
        return _dragon_tiger_cache
    try:
        import akshare as ak
        from datetime import date, timedelta
        end = date.today().strftime("%Y%m%d")
        start = (date.today() - timedelta(days=90)).strftime("%Y%m%d")
        df = ak.stock_lhb_detail_em(start_date=start, end_date=end)
        if df is not None and not df.empty:
            _dragon_tiger_cache = df
            logger.info(f"  龙虎榜: {len(df)}条")
            return df
    except Exception as e:
        logger.warning(f"  龙虎榜失败: {e}")
    return None
