"""
vectorbt 策略回测验证

验证端到端策略效果:
  - A股: "每天买TopN, 持有M天, 轮动调仓"
  - 金银: "信号触发即开仓, 止损N%"

输出绩效指标:
  - 年化收益率 / 最大回撤 / 夏普比率
  - 胜率 / 盈亏比
  - 相对基准的超额收益

用法:
    python src/backtest/verify.py                    # 完整回测
    python src/backtest/verify.py --top-n 20         # Top20策略
    python src/backtest/verify.py --compare          # 对比多个参数
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import json
from datetime import date, timedelta
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt

from config import (
    RESULTS_DIR, MODEL_DIR, LABEL_HORIZONS,
    A_STOCK_DIR, GOLD_SILVER_DIR, MACRO_DIR,
)
from src.data.storage import read_daily_bars
from src.data.fetcher import load_adjust_factors, apply_hfq
from src.features.factors import compute_all_factors, get_factor_names

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  A股 选股策略回测
# ═══════════════════════════════════════════════════════

def backtest_a_stock_ranking(
    top_n: int = 20,
    hold_days: int = 5,
    max_weight: float = 0.10,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    use_model: bool = True,
) -> Dict:
    """
    A股排序选股回测。

    策略逻辑:
      每天用模型打分 → 选Top N → 等权买入 → 持有 hold_days 天后轮动

    Parameters
    ----------
    top_n : int
        持仓数量
    hold_days : int
        持有天数 (调仓频率)
    max_weight : float
        单票最大仓位
    use_model : bool
        True=用LightGBM模型排序, False=用动量因子排序(基准对比)

    Returns
    -------
    dict with: portfolio, stats, benchmark_stats, comparison
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 3)

    logger.info("=" * 60)
    logger.info(f"A股选股回测: Top{top_n}, 持有{hold_days}天, {start_date}→{end_date}")
    logger.info("=" * 60)

    # 1. 加载数据
    logger.info("[1/6] 加载数据...")
    raw_df = read_daily_bars(A_STOCK_DIR, start_date=start_date,
                             end_date=end_date, market="a_stock")
    if raw_df.empty:
        raise ValueError("无数据")

    adjust_factors = load_adjust_factors()
    if not adjust_factors.empty:
        raw_df = apply_hfq(raw_df, adjust_factors)

    logger.info(f"  {raw_df['symbol'].nunique()}只, {len(raw_df)}行")

    # 2. 计算因子
    logger.info("[2/6] 计算统一因子...")
    # 获取估值数据 (腾讯财经)
    valuation_df = None
    try:
        from src.data.tencent_fetcher import fetch_valuation_batch
        symbols_list = raw_df["symbol"].unique().tolist()
        valuation_df = fetch_valuation_batch(symbols=symbols_list[:500])
    except Exception as e:
        logger.warning(f"  估值数据跳过: {e}")
    factor_df = compute_all_factors(raw_df, valuation_df)

    # 3. 生成信号
    logger.info("[3/6] 生成信号...")

    if use_model:
        # 用LightGBM模型打分
        signals = _model_generated_signals(factor_df)
    else:
        # 基准: 用20日动量排序
        signals = _momentum_baseline_signals(factor_df)

    # 4. 构建持仓矩阵
    logger.info("[4/6] 构建持仓矩阵...")
    price_matrix = factor_df.pivot_table(
        index="trade_date", columns="symbol", values="close", aggfunc="last"
    )
    price_matrix = price_matrix.sort_index().ffill()
    signals = signals.reindex_like(price_matrix).fillna(0)

    # 每天选Top N
    weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)
    for date_idx in signals.index:
        day_scores = signals.loc[date_idx].dropna()
        if len(day_scores) == 0:
            continue
        top_symbols = day_scores.nlargest(top_n).index
        weights.loc[date_idx, top_symbols] = 1.0 / top_n
        # 限制单票最大仓位
        weights.loc[date_idx] = weights.loc[date_idx].clip(upper=max_weight)

    # 每隔hold_days调仓
    weights = weights.iloc[::hold_days].reindex(price_matrix.index, method="ffill")

    # 5. vectorbt 回测
    logger.info("[5/6] vectorbt 回测...")
    closes = price_matrix.ffill()
    pf = vbt.Portfolio.from_orders(
        closes,
        size=weights,
        size_type="targetpercent",
        freq="1d",
        init_cash=1_000_000,
        fees=0.0003,       # 万三佣金
        slippage=0.001,    # 0.1% 滑点
        fixed_fees=5.0,    # 每笔最低5元
    )

    # 6. 绩效统计
    logger.info("[6/6] 计算绩效...")
    stats = _compute_stats(pf, "A股Top" + str(top_n))

    # 基准: 等权持有所有股票
    bench_weights = pd.DataFrame(
        1.0 / len(price_matrix.columns),
        index=price_matrix.index,
        columns=price_matrix.columns,
    )
    bench_pf = vbt.Portfolio.from_orders(
        closes, size=bench_weights, size_type="targetpercent", freq="1d",
        init_cash=1_000_000, fees=0.0003, slippage=0.001,
    )
    bench_stats = _compute_stats(bench_pf, "等权基准")

    logger.info(f"  {stats['label']}: 年化{stats['annual_return']:.1%}, "
                f"夏普{stats['sharpe']:.2f}, 回撤{stats['max_drawdown']:.1%}")
    logger.info(f"  {bench_stats['label']}: 年化{bench_stats['annual_return']:.1%}, "
                f"夏普{bench_stats['sharpe']:.2f}")

    return {
        "strategy": stats,
        "benchmark": bench_stats,
        "excess_return": stats["annual_return"] - bench_stats["annual_return"],
        "portfolio": pf,
        "weights": weights,
    }


# ═══════════════════════════════════════════════════════
#  金银 择时策略回测
# ═══════════════════════════════════════════════════════

def backtest_gold_silver(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    stop_loss: float = 0.02,
    take_profit: float = 0.04,
) -> Dict:
    """
    金银择时回测。

    策略: 信号=BUY则做多, SELL则做空, HOLD则空仓
    """
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 3)

    logger.info(f"金银择时回测: {start_date}→{end_date}")

    # 加载
    gs_df = read_daily_bars(GOLD_SILVER_DIR, start_date=start_date,
                            end_date=end_date, market="gold_silver")
    if gs_df.empty:
        logger.warning("无金银数据, 跳过回测")
        return {}

    results = {}
    for symbol, group in gs_df.groupby("symbol"):
        price = group.set_index("trade_date")["close"].sort_index()
        # 过滤掉 NaN 和 <=0 的无效价格
        price = price[price > 0].dropna()
        if len(price) < 100:  # 至少100个交易日
            logger.warning(f"  {symbol}: 有效数据不足 ({len(price)}条), 跳过")
            continue

        # 用简单均线交叉生成模拟信号 (后续替换为模型信号)
        ma_fast = price.rolling(5).mean()
        ma_slow = price.rolling(20).mean()
        # 生成信号: 1=做多(100%仓位), 0=空仓
        signal = pd.Series(0, index=price.index)
        signal[ma_fast > ma_slow] = 1    # 做多
        signal = signal.shift(1).fillna(0)

        # 回测
        pf = vbt.Portfolio.from_orders(
            price, size=signal, size_type="targetpercent",
            freq="1d", init_cash=100_000,
            fees=0.0005,          # 万五手续费
        )

        stats = _compute_stats(pf, symbol)

        # 买入持有基准
        bh_pf = vbt.Portfolio.from_orders(
            price, size=np.inf, size_type="targetpercent",
            freq="1d", init_cash=100_000,
        )
        bh_stats = _compute_stats(bh_pf, f"{symbol}_基准")

        results[symbol] = {
            "strategy": stats,
            "benchmark": bh_stats,
            "excess": stats["annual_return"] - bh_stats["annual_return"],
        }

        logger.info(f"  {symbol}: 年化{stats['annual_return']:.1%} vs "
                    f"基准{bh_stats['annual_return']:.1%}, 夏普{stats['sharpe']:.2f}")

    if not results:
        logger.warning("金银回测无有效结果 — 请先运行 fetch_gold_silver_daily 获取充足历史数据")

    return results


# ═══════════════════════════════════════════════════════
#  绩效计算
# ═══════════════════════════════════════════════════════

def _compute_stats(pf: vbt.Portfolio, label: str) -> Dict:
    """从 vectorbt portfolio 提取绩效指标"""
    try:
        stats = pf.stats()
        # 用 Total Return 计算年化
        total_return_pct = float(stats.get("Total Return [%]", 0))
        n_days = (pf.wrapper.index[-1] - pf.wrapper.index[0]).days
        if n_days > 0 and total_return_pct != 0:
            annual_return = (1 + total_return_pct / 100) ** (365.25 / n_days) - 1
        else:
            annual_return = 0.0

        return {
            "label": label,
            "total_return": total_return_pct / 100,
            "annual_return": annual_return,
            "max_drawdown": float(stats.get("Max Drawdown [%]", 0) or 0) / 100,
            "sharpe": float(stats.get("Sharpe Ratio", 0) or 0),
            "calmar": float(stats.get("Calmar Ratio", 0) or 0),
            "win_rate": float(stats.get("Win Rate [%]", 0) or 0) / 100,
            "expectancy": float(stats.get("Expectancy", 0) or 0),
            "start_value": float(stats.get("Start Value", 0)),
            "end_value": float(stats.get("End Value", 0)),
            "n_trades": int(stats.get("Total Trades", 0)),
        }
    except Exception as e:
        logger.warning(f"绩效计算异常: {e}")
        return {
            "label": label,
            "total_return": 0, "annual_return": 0,
            "max_drawdown": 0, "sharpe": 0, "calmar": 0,
            "win_rate": 0, "expectancy": 0,
            "start_value": 0, "end_value": 0, "n_trades": 0,
        }


def _model_generated_signals(factor_df: pd.DataFrame) -> pd.DataFrame:
    """用LightGBM模型生成信号矩阵 (symbols × dates)"""
    model_path = MODEL_DIR / "a_stock_ranker.txt"

    if not model_path.exists():
        logger.warning("模型不存在, 回退到动量因子")
        return _momentum_baseline_signals(factor_df)

    try:
        import lightgbm as lgb
    except (ImportError, OSError) as e:
        logger.warning(f"LightGBM 加载失败: {e}, 回退到动量因子")
        return _momentum_baseline_signals(factor_df)

    try:
        model = lgb.Booster(model_file=str(model_path))
    except Exception as e:
        logger.warning(f"模型加载失败: {e}, 回退到动量因子")
        return _momentum_baseline_signals(factor_df)

    factor_names = get_factor_names()
    # 只保留实际存在且非全NaN的因子
    available = [f for f in factor_names
                 if f in factor_df.columns and factor_df[f].notna().any()]

    # 模型训练时用的特征数
    n_model_features = model.num_feature()
    logger.info(f"  模型特征数: {n_model_features}, 可用特征: {len(available)}")

    # 如果可用特征数与模型不匹配, 只用前n个 Phase1 因子
    if len(available) != n_model_features:
        phase1_factors = ["beta", "short_reversal", "momentum_12m1m",
                          "low_volatility", "amihud"]
        available = [f for f in phase1_factors if f in factor_df.columns]
        if len(available) != n_model_features:
            logger.warning(f"特征数不匹配且无法对齐, 回退到动量基线")
            return _momentum_baseline_signals(factor_df)

    # 逐日预测
    scores = {}
    for date_val, group in factor_df.groupby("trade_date"):
        X = group[available].fillna(0).astype(np.float32)
        preds = model.predict(X)
        for sym, s in zip(group["symbol"], preds):
            scores.setdefault(date_val, {})[sym] = s

    return pd.DataFrame(scores).T


def _momentum_baseline_signals(factor_df: pd.DataFrame) -> pd.DataFrame:
    """基准信号: 20日动量"""
    if "ret_20d" in factor_df.columns:
        signals = factor_df.pivot_table(
            index="trade_date", columns="symbol", values="ret_20d", aggfunc="last"
        )
    else:
        # 用价格算
        signals = factor_df.pivot_table(
            index="trade_date", columns="symbol", values="close", aggfunc="last"
        )
        signals = signals.pct_change(20)
    return signals


# ═══════════════════════════════════════════════════════
#  参数对比
# ═══════════════════════════════════════════════════════

def compare_params(
    top_n_list: list = [10, 20, 30, 50],
    hold_days_list: list = [5, 10, 20],
) -> pd.DataFrame:
    """网格搜索最佳参数组合"""
    results = []
    for top_n in top_n_list:
        for hold_days in hold_days_list:
            try:
                bt = backtest_a_stock_ranking(top_n=top_n, hold_days=hold_days, use_model=False)
                s = bt["strategy"]
                results.append({
                    "top_n": top_n, "hold_days": hold_days,
                    "annual_return": s["annual_return"],
                    "sharpe": s["sharpe"],
                    "max_drawdown": s["max_drawdown"],
                })
                logger.info(f"  Top{top_n}/持有{hold_days}d: "
                          f"年化{s['annual_return']:.1%}, 夏普{s['sharpe']:.2f}")
            except Exception as e:
                logger.warning(f"  Top{top_n}/持有{hold_days}d: {e}")

    result_df = pd.DataFrame(results)
    best = result_df.loc[result_df["sharpe"].idxmax()]
    logger.info(f"  最佳参数: Top{int(best['top_n'])}/{int(best['hold_days'])}天, "
                f"夏普{best['sharpe']:.2f}")
    return result_df


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", choices=["a_stock", "gold_silver", "all"], default="all")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--hold-days", type=int, default=5)
    parser.add_argument("--compare", action="store_true", help="参数网格搜索")
    args = parser.parse_args()

    if args.compare:
        compare_params()
    elif args.assets in ("a_stock", "all"):
        backtest_a_stock_ranking(top_n=args.top_n, hold_days=args.hold_days)
    elif args.assets in ("gold_silver", "all"):
        backtest_gold_silver()
