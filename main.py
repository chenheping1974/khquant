#!/usr/bin/env python3
"""
khquant — A股选股 + 金银择时 量化系统

模式:
  python main.py --mode daily       # 每日自动化流水线 (GitHub Actions)
  python main.py --mode train       # 本地训练模型 (每周一次)
  python main.py --mode backtest    # 本地回测验证
  python main.py --mode full        # 训练→回测→(通过则)推送模型
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    ROOT, A_STOCK_DIR, GOLD_SILVER_DIR, MACRO_DIR, MODEL_DIR, RESULTS_DIR,
)
from src.data.fetcher import fetch_a_stock_daily, fetch_gold_silver_daily, fetch_macro_daily
from src.data.storage import write_daily_bars, read_daily_bars, get_latest_date
from src.data.symbols import get_today_tradable_pool
from src.features import a_stock as a_stock_features
from src.features import macro as macro_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(ROOT / "pipeline.log", encoding="utf-8")],
)
logger = logging.getLogger("khquant")


# ═══════════════════════════════════════════════════════
#  模式1: 每日自动化流水线 (GitHub Actions)
# ═══════════════════════════════════════════════════════

def daily_pipeline(test_mode: bool = False, n_symbols: int = 0):
    """每天 16:00 自动运行: 拉数据 → 因子 → 推理 → 信号"""
    today = date.today()
    logger.info("=" * 60)
    logger.info(f"每日流水线 — {today} — {'测试' if test_mode else '正式'}")
    logger.info("=" * 60)

    # ── 获取股票池 ──
    symbols = _get_stock_symbols(test_mode, n_symbols)

    # ── Step 1: 数据采集 ──
    logger.info("[1/4] 数据采集...")
    if symbols:
        a_df = fetch_a_stock_daily(symbols, incremental=True, end_date=today)
        if not a_df.empty:
            write_daily_bars(a_df, A_STOCK_DIR, market="a_stock")
            logger.info(f"  A股: {len(a_df)}行 ✓")

    gs_df = fetch_gold_silver_daily(start_date=today - timedelta(days=30), end_date=today)
    if not gs_df.empty:
        write_daily_bars(gs_df, GOLD_SILVER_DIR, market="gold_silver")

    macro_df = fetch_macro_daily(start_date=today - timedelta(days=60), end_date=today)
    if not macro_df.empty:
        write_daily_bars(macro_df, MACRO_DIR, market="macro")

    # ── Step 2: 因子计算 (仅算最新, 供推理用) ──
    logger.info("[2/4] 因子计算...")
    lookback = 30 if test_mode else 120
    start = today - timedelta(days=lookback)

    a_raw = read_daily_bars(A_STOCK_DIR, start_date=start, end_date=today, market="a_stock")
    if not a_raw.empty:
        a_factors = a_stock_features.compute_all_factors(a_raw)
        latest_date = pd.to_datetime(a_factors["trade_date"]).max().date()
        a_factors[a_factors["trade_date"].astype(str) == str(latest_date)].to_parquet(
            A_STOCK_DIR / "latest_factors.parquet"
        )
        logger.info(f"  A股因子: 最新{latest_date}")

    gs_raw = read_daily_bars(GOLD_SILVER_DIR, start_date=start, end_date=today, market="gold_silver")
    m_raw = read_daily_bars(MACRO_DIR, start_date=start, end_date=today, market="macro")
    if not gs_raw.empty:
        macro_feat_df = macro_features.compute_all_factors(gs_raw, m_raw)
        macro_feat_df.to_parquet(GOLD_SILVER_DIR / "latest_factors.parquet")
        logger.info(f"  金银因子: {len(macro_feat_df)}行")

    # ── Step 3: 推理 ──
    logger.info("[3/4] LightGBM 推理...")
    from src.models.inference import run_daily_inference
    signals = run_daily_inference(inference_date=today, lookback_days=lookback)

    # ── Step 4: 信号输出 ──
    logger.info("[4/4] 信号输出...")
    _print_daily_summary(signals)

    logger.info("✅ 每日流水线完成")
    return signals


# ═══════════════════════════════════════════════════════
#  模式2: 本地训练
# ═══════════════════════════════════════════════════════

def train_pipeline(quick: bool = False, n_symbols: int = 0):
    """本地训练模型"""
    from src.models.train import train_a_stock_ranker, train_gold_silver_classifier

    symbols = None
    if n_symbols > 0:
        symbols = _get_test_symbols(n_symbols)

    logger.info("=" * 60)
    logger.info("模型训练 (本地)")
    logger.info("=" * 60)

    # A股
    model_a, importance_a, stats_a = train_a_stock_ranker(
        symbols=symbols, quick=quick
    )
    logger.info(f"A股模型: {stats_a['n_estimators']} trees, "
                f"Top3特征: {stats_a['top10_features'][:3]}")

    # 金银
    model_gs, importance_gs, stats_gs = train_gold_silver_classifier(quick=quick)
    logger.info(f"金银模型: {stats_gs['n_estimators']} trees")

    return {"a_stock": (model_a, stats_a), "gold_silver": (model_gs, stats_gs)}


# ═══════════════════════════════════════════════════════
#  模式3: 回测验证
# ═══════════════════════════════════════════════════════

def backtest_pipeline(compare: bool = False):
    """回测验证"""
    from src.backtest.verify import (
        backtest_a_stock_ranking, backtest_gold_silver, compare_params,
    )

    if compare:
        logger.info("参数网格搜索...")
        results = compare_params()
        print("\n最佳参数组合:\n", results.sort_values("sharpe", ascending=False).head(10))
        return results

    logger.info("=" * 60)
    logger.info("策略回测")
    logger.info("=" * 60)

    # A股
    a_result = backtest_a_stock_ranking(top_n=20, hold_days=5)
    a_s = a_result["strategy"]
    a_b = a_result["benchmark"]
    print(f"\n{'='*60}")
    print(f"A股选股回测结果:")
    print(f"  策略: 年化{a_s['annual_return']:.1%}, 夏普{a_s['sharpe']:.2f}, "
          f"回撤{a_s['max_drawdown']:.1%}")
    print(f"  基准: 年化{a_b['annual_return']:.1%}, 夏普{a_b['sharpe']:.2f}")
    print(f"  超额: {a_result['excess_return']:.1%}")
    print(f"  终值: ¥{a_s['end_value']:,.0f} (初始¥{a_s['start_value']:,.0f})")

    # 金银
    gs_result = backtest_gold_silver()
    for sym, r in gs_result.items():
        s = r["strategy"]
        print(f"\n{sym}: 年化{s['annual_return']:.1%}, 夏普{s['sharpe']:.2f}, "
              f"超额{r['excess']:.1%}")

    return a_result, gs_result


# ═══════════════════════════════════════════════════════
#  模式4: 完整循环 (训练→回测→决策)
# ═══════════════════════════════════════════════════════

def full_cycle(quick: bool = False):
    """完整循环: 训练→回测→对比→决策"""
    # 1. 训练新模型
    logger.info("Phase 1/3: 训练...")
    models = train_pipeline(quick=quick)

    # 2. 回测
    logger.info("Phase 2/3: 回测...")
    bt = backtest_pipeline()

    # 3. 决策
    logger.info("Phase 3/3: 决策...")
    sharpe = bt[0]["strategy"]["sharpe"] if isinstance(bt, tuple) else bt["strategy"]["sharpe"]

    if sharpe >= 0.8:
        logger.info(f"✅ 夏普{sharpe:.2f} ≥ 0.8 → 模型可上线")
        logger.info("   git add models/ && git commit -m '模型更新' && git push")
        logger.info("   下次Actions自动使用新模型!")
    else:
        logger.info(f"❌ 夏普{sharpe:.2f} < 0.8 → 需调整因子/标签/参数")


# ═══════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════

def _get_stock_symbols(test_mode: bool, n_symbols: int):
    """获取股票池"""
    if test_mode or n_symbols > 0:
        try:
            pool, _ = get_today_tradable_pool()
            syms = pool["symbol"].tolist()[:n_symbols or 10]
            logger.info(f"测试池: {len(syms)}只")
            return syms
        except Exception:
            return _get_test_symbols(n_symbols or 10)
    # 正式模式: 从已有数据中获取全量
    try:
        from src.data.storage import get_unique_symbols
        return get_unique_symbols(A_STOCK_DIR)
    except Exception:
        return _get_test_symbols(500)


def _get_test_symbols(n: int):
    defaults = ["000001", "600036", "600519", "000858", "002415",
                "300750", "601318", "000333", "600900", "002594",
                "000002", "600276", "300124", "002475", "600809",
                "000725", "603259", "600030", "000063", "300274"]
    return defaults[:n]


def _print_daily_summary(signals):
    """打印每日信号摘要"""
    a_df = signals.get("a_stock_signals", pd.DataFrame())
    gs_df = signals.get("gold_silver_signals", pd.DataFrame())

    if not a_df.empty:
        top5 = a_df.head(5)
        print(f"\n📊 A股 Top5:")
        for _, r in top5.iterrows():
            print(f"  {int(r['rank']):2d}. {r['symbol']} 得分:{r['score']:.4f}  {r['signal']}")

    if not gs_df.empty:
        print(f"\n🥇 金银信号:")
        for _, r in gs_df.iterrows():
            print(f"  {r['symbol']}: {r['direction_cn']} (置信度{r['confidence']:.0f}%)")


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="khquant")
    parser.add_argument("--mode", choices=["daily", "train", "backtest", "full"],
                        default="daily")
    parser.add_argument("--symbols", type=int, default=0,
                        help="限制股票数 (0=全部)")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--quick", action="store_true", help="快速训练 (少树)")
    parser.add_argument("--compare", action="store_true", help="参数网格搜索")
    args = parser.parse_args()

    logger.info(f"🚀 khquant — 模式: {args.mode}")

    try:
        if args.mode == "daily":
            daily_pipeline(test_mode=args.test, n_symbols=args.symbols)
        elif args.mode == "train":
            train_pipeline(quick=args.quick, n_symbols=args.symbols)
        elif args.mode == "backtest":
            backtest_pipeline(compare=args.compare)
        elif args.mode == "full":
            full_cycle(quick=args.quick)
    except Exception as e:
        logger.exception(f"❌ 失败: {e}")
        sys.exit(1)
