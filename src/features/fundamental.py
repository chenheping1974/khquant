"""
基本面因子模块

因子设计参考:
  - 幻方量化: 基本面因子占30% (PE/PB/ROE/利润增速/现金流质量)
  - Fama-French 5-factor: 价值(Value) + 盈利(Profitability) + 投资(Investment)
  - A股实战: 低PE/低PB/高ROE 长期有效

数据源:
  - akshare (东方财富) — PE/PB/市值 (每日)
  - akshare (财务指标) — ROE/利润增速/负债率 (季度, 前向填充到每日)
"""
import logging
from datetime import date, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

import akshare as ak

logger = logging.getLogger(__name__)

# 因子注册表 (供特征选择/文档)
FUNDAMENTAL_FACTORS: Dict[str, str] = {}

REQUEST_DELAY = 0.5  # akshare 请求间隔(秒)


# ═══════════════════════════════════════════════════════
#  1. 估值因子 (每日更新)
# ═══════════════════════════════════════════════════════

def fetch_daily_valuation() -> pd.DataFrame:
    """
    获取全市场A股的每日估值指标。

    Returns
    -------
    DataFrame: 代码, 名称, pe_ttm, pb, market_cap, ...
    """
    logger.info("获取全市场估值数据...")
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.rename(columns={
            "代码": "symbol",
            "名称": "name",
            "市盈率-动态": "pe_ttm",
            "市净率": "pb",
            "总市值": "market_cap",
        })
        # 只保留需要的列
        cols = ["symbol", "pe_ttm", "pb", "market_cap"]
        df = df[[c for c in cols if c in df.columns]].copy()

        # 清洗
        for c in ["pe_ttm", "pb", "market_cap"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
                # 负PE是没有意义的
                if c == "pe_ttm":
                    df.loc[df[c] < 0, c] = np.nan
                # 极端值裁剪
                if c in ("pe_ttm", "pb"):
                    df[c] = df[c].clip(lower=0).replace(0, np.nan)

        # 注册因子
        FUNDAMENTAL_FACTORS["pe_ttm"] = "滚动市盈率(越低越便宜)"
        FUNDAMENTAL_FACTORS["pb"] = "市净率(越低越接近净资产)"
        FUNDAMENTAL_FACTORS["log_market_cap"] = "对数市值(规模因子)"

        # 添加衍生因子
        df["log_market_cap"] = np.log(df["market_cap"].fillna(1e10))
        # PE倒数 = 盈利收益率 (越高越好, 类似E/P)
        df["earnings_yield"] = 1.0 / df["pe_ttm"].replace(0, np.nan)
        FUNDAMENTAL_FACTORS["earnings_yield"] = "盈利收益率(1/PE,越高越便宜)"

        logger.info(f"  估值数据: {len(df)} 只")
        return df

    except Exception as e:
        logger.warning(f"获取估值数据失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  2. 财务质量因子 (季度, 前向填充)
# ═══════════════════════════════════════════════════════

def fetch_financial_quality(symbols: list, max_symbols: int = 200) -> pd.DataFrame:
    """
    获取财务质量指标 (ROE/利润增速/负债率)。

    注意: 财务数据按季度更新, 需要 forward-fill 到每日。
    由于 akshare 逐只查询较慢, 默认只拉前 max_symbols 只。
    """
    logger.info(f"获取财务质量数据 (最多{max_symbols}只)...")
    import time

    all_data = []
    symbols = symbols[:max_symbols]

    for i, code in enumerate(symbols):
        try:
            df = ak.stock_financial_analysis_indicator(symbol=code)
            if df is None or df.empty:
                continue

            # 标准化列名
            col_map = {
                "净资产收益率": "roe",
                "净利润增长率": "profit_growth",
                "营业总收入增长率": "revenue_growth",
                "资产负债率": "debt_ratio",
            }
            found_cols = {}
            for cn, en in col_map.items():
                for c in df.columns:
                    if cn in c:
                        found_cols[en] = c
                        break

            if not found_cols:
                continue

            df = df.rename(columns=found_cols)
            df["symbol"] = code
            # 日期列: 取第一列为日期
            date_col = df.columns[0]
            df["report_date"] = pd.to_datetime(date_col, errors="coerce")
            df = df.dropna(subset=["report_date"])

            keep = ["symbol", "report_date"] + list(found_cols.keys())
            all_data.append(df[[c for c in keep if c in df.columns]])

        except Exception:
            pass

        if (i + 1) % 50 == 0:
            logger.info(f"  [{i+1}/{len(symbols)}]")
        time.sleep(REQUEST_DELAY)

    if not all_data:
        logger.warning("无财务数据")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    # 注册因子
    for en, desc in [
        ("roe", "净资产收益率(越高盈利越好)"),
        ("profit_growth", "净利润增长率(越高增长越快)"),
        ("revenue_growth", "营收增长率"),
        ("debt_ratio", "资产负债率(越低越安全)"),
        ("quality_score", "质量综合得分(ROE+增长)"),
    ]:
        FUNDAMENTAL_FACTORS[en] = desc

    logger.info(f"  财务数据: {len(result)}行, {result['symbol'].nunique()}只")
    return result


# ═══════════════════════════════════════════════════════
#  3. 行业因子
# ═══════════════════════════════════════════════════════

def fetch_industry_mapping() -> pd.DataFrame:
    """
    获取申万行业分类 → 做行业虚拟变量。

    行业因子逻辑: 同行业股票有共同风险敞口,
    控制行业后模型能学到真正的选股能力。
    """
    logger.info("获取行业分类...")
    try:
        # 东方财富行业板块成分股
        df = ak.stock_board_industry_name_em()
        logger.info(f"  行业: {len(df)} 个")
        FUNDAMENTAL_FACTORS["industry_code"] = "申万行业编码"
        return df
    except Exception as e:
        logger.warning(f"获取行业分类失败: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  4. 因子合并
# ═══════════════════════════════════════════════════════

def compute_fundamental_factors(
    price_df: pd.DataFrame,
    valuation_df: Optional[pd.DataFrame] = None,
    financial_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    将估值 + 财务数据合并到每日价格数据中。

    合并策略:
      - 估值(PE/PB): 按 symbol 直接 merge (每日更新)
      - 财务(ROE等): forward-fill 最新季报数据到每日
    """
    result = price_df[["trade_date", "symbol", "close"]].copy()

    # 合并估值因子
    if valuation_df is not None and not valuation_df.empty:
        val_cols = ["symbol", "pe_ttm", "pb", "log_market_cap", "earnings_yield"]
        val_cols = [c for c in val_cols if c in valuation_df.columns]
        result = result.merge(
            valuation_df[val_cols], on="symbol", how="left"
        )

    # 合并财务质量因子 (forward-fill 到每日)
    if financial_df is not None and not financial_df.empty:
        fin_cols = ["symbol", "report_date", "roe", "profit_growth",
                     "revenue_growth", "debt_ratio"]
        fin_cols = [c for c in fin_cols if c in financial_df.columns]
        fin = financial_df[fin_cols].copy()

        if "report_date" in fin.columns and "roe" in fin.columns:
            # 对每只股票, 按报告期排序后 forward-fill
            fin = fin.sort_values(["symbol", "report_date"])
            for sym in result["symbol"].unique():
                sym_fin = fin[fin["symbol"] == sym]
                if sym_fin.empty:
                    continue
                # 获取最新一期数据
                latest = sym_fin.drop_duplicates(subset=["symbol"], keep="last")
                for c in ["roe", "profit_growth", "revenue_growth", "debt_ratio"]:
                    if c in latest.columns:
                        result.loc[result["symbol"] == sym, c] = latest[c].values[0]

        # 质量综合得分: ROE + 增长 (简单等权)
        score_cols = []
        for c in ["roe", "profit_growth", "revenue_growth"]:
            if c in result.columns:
                # 标准化到 0-1 范围
                vals = result[c].fillna(0)
                if vals.std() > 0:
                    result[f"{c}_z"] = (vals - vals.mean()) / vals.std()
                    score_cols.append(f"{c}_z")
        if score_cols:
            result["quality_score"] = result[score_cols].mean(axis=1)

    logger.info(f"基本面因子合并完成: {len(result)}行")
    return result


def get_fundamental_factor_names() -> list:
    """返回已注册的基本面因子名"""
    return list(FUNDAMENTAL_FACTORS.keys())
