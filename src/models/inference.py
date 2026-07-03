"""
每日推理模块

输入: 今日因子 + 训练好的模型
输出: 选股排序 + 金银方向信号

用法:
    from src.models.inference import run_daily_inference
    signals = run_daily_inference()
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from config import (
    MODEL_DIR, RESULTS_DIR, LABEL_HORIZONS,
    A_STOCK_DIR, GOLD_SILVER_DIR, MACRO_DIR,
)
from src.data.storage import read_daily_bars, get_latest_date
from src.data.fetcher import load_adjust_factors, apply_hfq
from src.features import a_stock as a_stock_feat
from src.features import macro as macro_feat

logger = logging.getLogger(__name__)


def run_daily_inference(
    inference_date: Optional[date] = None,
    lookback_days: int = 120,
) -> Dict[str, pd.DataFrame]:
    """
    每日推理 — 一站式生成选股+金银信号。

    Returns
    -------
    {
        "a_stock_signals": DataFrame (symbol, score, rank, signal),
        "gold_silver_signals": DataFrame (symbol, direction, prob),
        "top50": DataFrame,
        "industry_breakdown": DataFrame,
    }
    """
    if inference_date is None:
        inference_date = date.today()

    logger.info(f"每日推理: {inference_date}")

    results = {}

    # ── A股推理 ──
    try:
        results["a_stock_signals"] = _infer_a_stock(inference_date, lookback_days)
    except Exception as e:
        logger.error(f"A股推理失败: {e}")
        results["a_stock_signals"] = pd.DataFrame()

    # ── 金银推理 ──
    try:
        results["gold_silver_signals"] = _infer_gold_silver(inference_date, lookback_days)
    except Exception as e:
        logger.error(f"金银推理失败: {e}")
        results["gold_silver_signals"] = pd.DataFrame()

    # ── 保存结果 ──
    _save_results(results, inference_date)

    return results


def _infer_a_stock(inference_date: date, lookback_days: int) -> pd.DataFrame:
    """A股排序推理"""
    model_path = MODEL_DIR / "a_stock_ranker.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}\n请先运行: python src/models/train.py")

    logger.info("A股推理: 加载模型...")
    model = lgb.Booster(model_file=str(model_path))

    # 加载数据
    start = inference_date - timedelta(days=lookback_days)
    raw_df = read_daily_bars(A_STOCK_DIR, start_date=start,
                             end_date=inference_date, market="a_stock")
    if raw_df.empty:
        raise ValueError("无A股数据")

    # 后复权
    adjust_factors = load_adjust_factors()
    if not adjust_factors.empty:
        raw_df = apply_hfq(raw_df, adjust_factors)

    # 因子
    factor_df = a_stock_feat.compute_all_factors(raw_df)
    factor_names = a_stock_feat.get_factor_names()

    # 只取最新一天
    latest_date = pd.to_datetime(factor_df["trade_date"]).max().date()
    today_factors = factor_df[
        pd.to_datetime(factor_df["trade_date"]).dt.date == latest_date
    ].copy()

    if today_factors.empty:
        raise ValueError(f"无{latest_date}的因子数据")

    logger.info(f"  推理日: {latest_date}, {len(today_factors)}只股票")

    # 预测
    X = today_factors[factor_names].fillna(0).astype(np.float32)
    scores = model.predict(X)

    today_factors["score"] = scores
    today_factors["rank"] = today_factors["score"].rank(ascending=False).astype(int)

    # 信号分级
    top_pct = today_factors["rank"] / len(today_factors)
    conditions = [
        top_pct <= 0.05,           # 前5% → Strong Buy
        top_pct <= 0.15,           # 前15% → Buy
        top_pct >= 0.80,           # 后20% → Sell
    ]
    choices = ["STRONG_BUY", "BUY", "SELL"]
    today_factors["signal"] = np.select(conditions, choices, default="HOLD")

    result = today_factors[["symbol", "score", "rank", "signal"]].sort_values("rank")
    logger.info(f"  Top5: {result.head(5)['symbol'].tolist()}")
    return result


def _infer_gold_silver(inference_date: date, lookback_days: int) -> pd.DataFrame:
    """金银方向推理"""
    model_path = MODEL_DIR / "gold_silver_classifier.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    logger.info("金银推理: 加载模型...")
    model = lgb.Booster(model_file=str(model_path))

    # 数据
    start = inference_date - timedelta(days=lookback_days)
    gs_df = read_daily_bars(GOLD_SILVER_DIR, start_date=start,
                            end_date=inference_date, market="gold_silver")
    macro_df = read_daily_bars(MACRO_DIR, start_date=start,
                               end_date=inference_date, market="macro")

    if gs_df.empty:
        raise ValueError("无金银数据")

    # 因子
    factor_df = macro_feat.compute_all_factors(gs_df, macro_df)
    factor_names = macro_feat.get_factor_names()

    # 最新一天
    latest_date = pd.to_datetime(factor_df["trade_date"]).max().date()
    today_factors = factor_df[
        pd.to_datetime(factor_df["trade_date"]).dt.date == latest_date
    ]

    results = []
    for symbol, group in today_factors.groupby("symbol"):
        X = group[factor_names].fillna(0).astype(np.float32).iloc[0:1]
        proba = model.predict(X)[0]   # [跌概率, 震荡概率, 涨概率]

        direction_map = {0: ("SELL", "做空"), 1: ("BUY", "做多"), 2: ("HOLD", "观望")}
        pred_class = int(np.argmax(proba))
        action, action_cn = direction_map[pred_class]

        results.append({
            "symbol": symbol,
            "direction": action,
            "direction_cn": action_cn,
            "prob_down": round(float(proba[0]), 3),
            "prob_up": round(float(proba[1]), 3),
            "prob_hold": round(float(proba[2]), 3),
            "confidence": round(float(proba[pred_class]) * 100, 1),
            "close": float(group["close"].iloc[-1]) if "close" in group.columns else None,
        })

    result = pd.DataFrame(results)
    logger.info(f"  金银信号: {result[['symbol','direction_cn','confidence']].to_dict('records')}")
    return result


def _save_results(results: Dict, inference_date: date):
    """保存推理结果"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # A股信号
    a_df = results.get("a_stock_signals", pd.DataFrame())
    if not a_df.empty:
        a_df.to_parquet(RESULTS_DIR / f"a_stock_signals_{inference_date}.parquet", index=False)

        # Top50 markdown
        top50 = a_df.head(50)
        md = f"## 📊 A股选股信号 ({inference_date})\n\n"
        md += "| 排名 | 代码 | 得分 | 信号 |\n|------|------|------|------|\n"
        for _, r in top50.iterrows():
            md += f"| {int(r['rank'])} | {r['symbol']} | {r['score']:.4f} | {r['signal']} |\n"
        (RESULTS_DIR / f"a_stock_signals_{inference_date}.md").write_text(md, encoding="utf-8")

    # 金银信号
    gs_df = results.get("gold_silver_signals", pd.DataFrame())
    if not gs_df.empty:
        gs_df.to_parquet(RESULTS_DIR / f"gold_silver_signals_{inference_date}.parquet", index=False)

        md = f"## 🥇 金银信号 ({inference_date})\n\n"
        md += "| 品种 | 方向 | 置信度 | 做多概率 | 做空概率 |\n|------|------|------|------|------|\n"
        for _, r in gs_df.iterrows():
            md += f"| {r['symbol']} | {r['direction_cn']} | {r['confidence']}% | {r['prob_up']} | {r['prob_down']} |\n"
        (RESULTS_DIR / f"gold_silver_signals_{inference_date}.md").write_text(md, encoding="utf-8")

    logger.info(f"结果保存: {RESULTS_DIR}")


def get_latest_signals() -> Dict[str, pd.DataFrame]:
    """获取最近一次推理结果 (HuggingFace展示用)"""
    a_files = sorted(RESULTS_DIR.glob("a_stock_signals_*.parquet"))
    gs_files = sorted(RESULTS_DIR.glob("gold_silver_signals_*.parquet"))

    return {
        "a_stock": pd.read_parquet(a_files[-1]) if a_files else pd.DataFrame(),
        "gold_silver": pd.read_parquet(gs_files[-1]) if gs_files else pd.DataFrame(),
    }
