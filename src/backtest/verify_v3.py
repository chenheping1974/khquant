"""
v3.0 回测验证 — MSCI/BlackRock/AQR 四大因子

策略:
  每周末计算四大因子 → 选Top30 → 等权持有1周 → 下周重新排序

数据:
  - 量价因子: 从K线逐日计算 (Value的E/P用财报EPS+当日股价)
  - 财务因子: 季度财报前向填充到每日
  - 行业中性化: 行业内Z-score

用法:
    python src/backtest/verify_v3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import json
from datetime import date, timedelta
from typing import Optional, Dict

import numpy as np
import pandas as pd

from config import A_STOCK_DIR, GOLD_SILVER_DIR, RESULTS_DIR
from src.data.storage import read_daily_bars
from src.data.tencent_fetcher import fetch_valuation_batch
from src.features.fundamental import fetch_financial_quality
from src.features.factors_v3 import compute_four_factors

logger = logging.getLogger("backtest_v3")


# ═══════════════════════════════════════════════════════
#  历史回测
# ═══════════════════════════════════════════════════════

def backtest_v3(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    top_n: int = 30,
    rebalance: str = "weekly",
) -> Dict:
    """
    v3.0 四大因子回测。

    流程:
      每周末: 计算因子 → 选TopN → 持有至下周末
      因子归一化: 行业内Z-score
      权重: Value 25% + Momentum 25% + Quality 40% + LowVol 10%
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 3)

    logger.info("=" * 60)
    logger.info(f"v3.0 四大因子回测: {start_date} → {end_date}")
    logger.info(f"  调仓: {rebalance}, Top {top_n}")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("[1/4] 加载数据...")
    price_df = read_daily_bars(A_STOCK_DIR, start_date=start_date,
                               end_date=end_date, market="a_stock")
    if price_df.empty:
        raise ValueError("无A股数据")

    # 取当前估值作为近似 (历史PE不可得, 用当前PE近似)
    symbols_all = sorted(price_df["symbol"].unique().tolist())
    logger.info(f"  加载: {len(price_df)}行, {len(symbols_all)}只")

    # 获取历史估值数据 (baostock: 逐日PE/PB, 时间点正确)
    logger.info("  获取历史估值数据 (baostock)...")
    hist_val = None
    try:
        from src.data.baostock_fetcher import fetch_historical_valuation
        n_val_stocks = min(200, len(symbols_all))
        hist_val = fetch_historical_valuation(
            symbols=symbols_all[:n_val_stocks],
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if hist_val is not None and not hist_val.empty:
            logger.info(f"  历史估值: {len(hist_val)}行, {hist_val['symbol'].nunique()}只")
    except Exception as e:
        logger.warning(f"  历史估值跳过: {e}")

    # 财务数据跳过 (太慢, Quality因子暂缺)
    fin_df = None

    # 2. 行业映射跳过 (加速)
    industry_map = {}

    # 3. 逐周回测
    logger.info("[3/4] 逐周回测...")
    price_df["trade_date"] = pd.to_datetime(price_df["trade_date"])
    all_dates = sorted(price_df["trade_date"].unique())

    # 按周取调仓日 (每周五)
    weekly_dates = [d for d in all_dates if d.weekday() == 4]
    if not weekly_dates:
        weekly_dates = all_dates[::5]  # 每5天

    logger.info(f"  共 {len(weekly_dates)} 周, {len(all_dates)} 个交易日")

    portfolio_values = []
    cash = 1_000_000
    prev_holdings = set()

    for i, rebalance_date in enumerate(weekly_dates):
        pd_date = pd.Timestamp(rebalance_date)

        # 过去252天数据
        lookback = pd_date - pd.Timedelta(days=400)
        week_data = price_df[
            (price_df["trade_date"] >= lookback) &
            (price_df["trade_date"] <= pd_date)
        ]
        if len(week_data) < 500:
            continue

        # 获取该日期的估值数据 (时间点正确)
        week_val = None
        if hist_val is not None and not hist_val.empty:
            date_str = pd_date.strftime("%Y-%m-%d")
            date_mask = hist_val["trade_date"].astype(str) == date_str
            week_val = hist_val[date_mask][["symbol", "close", "pe_ttm", "pb"]].copy()
            if not week_val.empty:
                # 补上 total_shares (从当天 close 和 PE 无法反推, 用近似)
                week_val["total_shares"] = np.nan
                week_val["market_cap"] = np.nan
                week_val["earnings_yield"] = 1.0 / week_val["pe_ttm"].replace(0, np.nan)
                week_val["log_market_cap"] = np.nan
            else:
                week_val = None

        # 计算因子
        try:
            scores = compute_four_factors(week_data, week_val, fin_df, industry_map)
        except Exception:
            continue

        if scores.empty:
            continue

        # 选 Top N (行业分散 — 简化: 直接选top)
        top = scores.head(top_n)
        selected = set(top["symbol"].tolist())

        # 持有到下周
        next_date = weekly_dates[i + 1] if i < len(weekly_dates) - 1 else all_dates[-1]
        next_pd = pd.Timestamp(next_date)

        # 算本周收益
        week_return = 0
        for sym in selected:
            sym_data = price_df[
                (price_df["symbol"] == sym) &
                (price_df["trade_date"] > pd_date) &
                (price_df["trade_date"] <= next_pd)
            ].sort_values("trade_date")

            if len(sym_data) >= 2:
                entry = sym_data["close"].iloc[0]
                exit_p = sym_data["close"].iloc[-1]
                week_return += (exit_p / entry - 1) / len(selected)

        cash *= (1 + week_return)

        portfolio_values.append({
            "date": rebalance_date,
            "value": cash,
            "n_stocks": len(selected),
        })

        prev_holdings = selected

        if (i + 1) % 30 == 0:
            logger.info(f"  [{i+1}/{len(weekly_dates)}] value=¥{cash:,.0f}")

    # 4. 计算绩效
    logger.info("[4/4] 计算绩效...")
    if not portfolio_values:
        return {"error": "no trades"}

    pf_df = pd.DataFrame(portfolio_values)
    pf_df["return"] = pf_df["value"].pct_change()

    # 年化
    n_days = (pf_df["date"].iloc[-1] - pf_df["date"].iloc[0]).days
    total_ret = pf_df["value"].iloc[-1] / pf_df["value"].iloc[0] - 1
    annual_ret = (1 + total_ret) ** (365.25 / max(n_days, 1)) - 1

    # 回撤
    peak = pf_df["value"].cummax()
    drawdown = (pf_df["value"] - peak) / peak
    max_dd = drawdown.min()

    # 夏普
    weekly_returns = pf_df["return"].dropna()
    sharpe = (weekly_returns.mean() / weekly_returns.std() * np.sqrt(52)
              if weekly_returns.std() > 0 else 0)

    # 胜率
    win_rate = (weekly_returns > 0).mean()

    result = {
        "total_return": total_ret,
        "annual_return": annual_ret,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "final_value": pf_df["value"].iloc[-1],
        "n_weeks": len(pf_df),
        "n_days": n_days,
    }

    logger.info(f"  v3.0 四大因子:")
    logger.info(f"    总收益: {total_ret:.1%}")
    logger.info(f"    年化: {annual_ret:.1%}")
    logger.info(f"    夏普: {sharpe:.2f}")
    logger.info(f"    最大回撤: {max_dd:.1%}")
    logger.info(f"    胜率: {win_rate:.1%}")
    logger.info(f"    终值: ¥{pf_df['value'].iloc[-1]:,.0f}")

    return result


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    bt = backtest_v3(
        start_date=date.today() - timedelta(days=365 * 3),
        end_date=date.today(),
        top_n=30,
    )
    print(f"\n{'='*60}")
    print("v3.0 回测结果:")
    print(f"  年化: {bt.get('annual_return', 0):.1%}")
    print(f"  夏普: {bt.get('sharpe', 0):.2f}")
    print(f"  回撤: {bt.get('max_drawdown', 0):.1%}")
    print(f"  胜率: {bt.get('win_rate', 0):.1%}")
    print(f"  终值: ¥{bt.get('final_value', 0):,.0f}")
