"""
LightGBM 模型训练

A股: Lambdarank排序模型 → 预测股票相对收益率排序
金银: 分类模型 → 预测方向 (涨/跌/震荡)

用法:
    python src/models/train.py                          # 完整训练
    python src/models/train.py --assets a_stock         # 仅A股
    python src/models/train.py --assets gold_silver     # 仅金银
    python src/models/train.py --quick                  # 快速测试 (小样本)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
import json
from datetime import date, timedelta
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

from config import (
    MODEL_DIR, MODEL_PARAMS, LABEL_HORIZONS, LABEL_WEIGHTS,
    A_STOCK_DIR, GOLD_SILVER_DIR, MACRO_DIR,
)
from src.data.storage import read_daily_bars
from src.data.fetcher import load_adjust_factors, apply_hfq
from src.features import a_stock as a_stock_feat
from src.features import macro as macro_feat
from src.models.labels import build_labels, check_label_quality

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════
#  A股 Ranking 模型训练
# ═══════════════════════════════════════════════════════

def train_a_stock_ranker(
    symbols: list = None,
    train_end_date: Optional[date] = None,
    n_years_train: int = 3,
    n_years_valid: int = 1,
    quick: bool = False,
) -> Tuple[lgb.Booster, pd.DataFrame, Dict]:
    """
    训练A股排序模型。

    流程:
      加载数据 → 复权 → 因子计算 → 标签构建
      → 时间序列切分 (train/valid)
      → LightGBM Ranker训练
      → 保存模型 + 特征重要性

    Returns
    -------
    model, feature_importance, training_stats
    """
    if train_end_date is None:
        train_end_date = date.today()
    train_start = train_end_date - timedelta(days=365 * (n_years_train + n_years_valid))

    logger.info("=" * 60)
    logger.info("A股 LightGBM Ranker 训练")
    logger.info(f"  数据范围: {train_start} → {train_end_date}")
    logger.info(f"  训练集: {n_years_train}年, 验证集: {n_years_valid}年")

    # 1. 加载A股数据
    logger.info("[1/6] 加载数据...")
    raw_df = read_daily_bars(
        A_STOCK_DIR,
        symbols=symbols[:500] if quick and symbols else symbols,
        start_date=train_start, end_date=train_end_date,
        market="a_stock",
    )
    if raw_df.empty:
        raise ValueError("无A股数据! 先运行 python main.py --step fetch")

    logger.info(f"  原始数据: {len(raw_df)}行, {raw_df['symbol'].nunique()}只")

    # 2. 后复权
    logger.info("[2/6] 应用后复权...")
    adjust_factors = load_adjust_factors()
    if not adjust_factors.empty:
        raw_df = apply_hfq(raw_df, adjust_factors)
        logger.info(f"  已应用复权因子: {len(adjust_factors)}只")
    else:
        logger.warning("  无复权因子! 使用未复权数据 (训练结果可能不准确)")

    # 3. 计算因子
    logger.info("[3/6] 计算A股因子...")
    factor_df = a_stock_feat.compute_all_factors(raw_df)
    factor_names = a_stock_feat.get_factor_names()
    logger.info(f"  因子: {len(factor_names)}个")

    # 4. 构建标签
    logger.info("[4/6] 构建标签...")
    price_df = factor_df[["trade_date", "symbol", "close"]].copy()
    labels_df, label_meta = build_labels(
        price_df, horizons=LABEL_HORIZONS, label_type="return"
    )
    quality = check_label_quality(labels_df)
    logger.info(f"  标签质量: mean={quality.get('mean', 0):.4f}, "
                f"std={quality.get('std', 0):.4f}, "
                f"positive={quality.get('pct_positive', 0):.1%}")

    # 5. 合并因子+标签 → 训练数据
    logger.info("[5/6] 准备训练集...")
    merged = factor_df.merge(labels_df, on=["trade_date", "symbol"], how="inner")

    # 时间序列切分 (不能随机打乱!)
    dates = sorted(merged["trade_date"].unique())
    valid_cutoff = dates[-int(len(dates) * n_years_valid / (n_years_train + n_years_valid))]
    train_mask = merged["trade_date"] < valid_cutoff
    valid_mask = merged["trade_date"] >= valid_cutoff

    def _prepare_lgb(df):
        """准备 LightGBM 输入"""
        X = df[factor_names].fillna(0).astype(np.float32)
        y = df["label_composite"].fillna(0).astype(np.float32)
        # lambdarank 需要整数标签 → 分桶到0-4
        y_int = pd.qcut(
            y, q=5, labels=False, duplicates="drop"
        ).astype(int)
        # 按日期分组 (排序学习需要)
        group = df.groupby("trade_date")["symbol"].count().values
        return X, y_int, group

    X_train, y_train, group_train = _prepare_lgb(merged[train_mask])
    X_valid, y_valid, group_valid = _prepare_lgb(merged[valid_mask])

    logger.info(f"  训练: {len(X_train)}样本, {len(group_train)}天, "
                f"日均{len(X_train)//max(len(group_train),1)}只")
    logger.info(f"  验证: {len(X_valid)}样本, {len(group_valid)}天")

    # 6. 训练
    logger.info("[6/6] LightGBM 训练...")
    train_data = lgb.Dataset(X_train, label=y_train, group=group_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, group=group_valid, reference=train_data)

    params = MODEL_PARAMS.copy()
    if quick:
        params["num_iterations"] = 50
        params["num_leaves"] = 15

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, valid_data],
        valid_names=["train", "valid"],
        num_boost_round=params.get("num_iterations", 500),
        callbacks=[
            lgb.early_stopping(30),
            lgb.log_evaluation(20),
        ],
    )

    # 特征重要性
    importance = pd.DataFrame({
        "feature": factor_names,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    importance["pct"] = importance["importance"] / importance["importance"].sum()

    # 训练统计
    stats = {
        "model_type": "lambdarank",
        "features": factor_names,
        "n_features": len(factor_names),
        "n_estimators": model.current_iteration(),
        "label_meta": label_meta,
        "label_quality": quality,
        "train_samples": len(X_train),
        "valid_samples": len(X_valid),
        "top10_features": importance.head(10)["feature"].tolist(),
        "train_date": date.today().isoformat(),
    }

    # 保存
    model_path = MODEL_DIR / "a_stock_ranker.txt"
    model.save_model(str(model_path))
    importance.to_parquet(MODEL_DIR / "a_stock_importance.parquet")
    (MODEL_DIR / "a_stock_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    logger.info(f"  模型: {model_path}")
    logger.info(f"  Best iteration: {model.current_iteration()}")
    logger.info(f"  Valid NDCG: {model.best_score.get('valid', {}).get('ndcg@1', 'N/A')}")
    logger.info(f"  Top5特征: {importance.head(5)['feature'].tolist()}")

    return model, importance, stats


# ═══════════════════════════════════════════════════════
#  金银 分类/回归模型训练
# ═══════════════════════════════════════════════════════

def train_gold_silver_classifier(
    train_end_date: Optional[date] = None,
    n_years_train: int = 8,
    n_years_valid: int = 2,
    quick: bool = False,
) -> Tuple[lgb.Booster, pd.DataFrame, Dict]:
    """
    训练金银方向预测模型。

    标签: 未来5日方向 (涨/跌/震荡 → 三分类)
    因子: 宏观(40%) + 技术(60%)
    """
    if train_end_date is None:
        train_end_date = date.today()
    train_start = train_end_date - timedelta(days=365 * (n_years_train + n_years_valid))

    logger.info("=" * 60)
    logger.info("金银 LightGBM Classifier 训练")

    # 1. 加载数据
    logger.info("[1/5] 加载数据...")
    gs_df = read_daily_bars(GOLD_SILVER_DIR, start_date=train_start,
                            end_date=train_end_date, market="gold_silver")
    macro_df = read_daily_bars(MACRO_DIR, start_date=train_start,
                               end_date=train_end_date, market="macro")
    if gs_df.empty:
        raise ValueError("无金银数据!")

    # 2. 因子计算
    logger.info("[2/5] 计算金银因子...")
    factor_df = macro_feat.compute_all_factors(gs_df, macro_df)
    factor_names = macro_feat.get_factor_names()
    logger.info(f"  因子: {len(factor_names)}个")

    # 3. 标签 (方向分类)
    logger.info("[3/5] 构建标签...")
    for sym, group in factor_df.groupby("symbol"):
        close = group.sort_values("trade_date")["close"].values
        # 未来5日方向
        future = np.roll(close, -5)
        future[-5:] = np.nan
        ret = (future - close) / close
        factor_df.loc[group.index, "label_direction"] = np.where(
            ret > 0.01, 1, np.where(ret < -0.01, 0, 2)  # 1=涨, 0=跌, 2=震荡
        )[0] if hasattr(ret, '__iter__') else 2

    # 去掉NA
    factor_df = factor_df.dropna(subset=["label_direction"] + factor_names, how="any")
    factor_df["label_direction"] = factor_df["label_direction"].astype(int)

    # 4. 切分
    logger.info("[4/5] 准备训练集...")
    dates = sorted(factor_df["trade_date"].unique())
    if len(dates) < 30:
        logger.warning("  金银数据不足 (需≥30个交易日), 跳过")
        return None, pd.DataFrame(), {"error": "insufficient_data"}
    valid_size = max(1, int(len(dates) * n_years_valid / (n_years_train + n_years_valid)))
    valid_cutoff = dates[-valid_size]
    train_mask = factor_df["trade_date"] < valid_cutoff
    valid_mask = ~train_mask

    X_train = factor_df.loc[train_mask, factor_names].fillna(0).astype(np.float32)
    y_train = factor_df.loc[train_mask, "label_direction"].astype(int)
    X_valid = factor_df.loc[valid_mask, factor_names].fillna(0).astype(np.float32)
    y_valid = factor_df.loc[valid_mask, "label_direction"].astype(int)

    logger.info(f"  训练: {len(X_train)}条, 验证: {len(X_valid)}条")

    # 5. 训练
    logger.info("[5/5] LightGBM 训练...")
    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.03,
        "feature_fraction": 0.8,
        "verbose": -1,
        "seed": 42,
    }
    if quick:
        params["num_iterations"] = 30

    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid)

    model = lgb.train(
        params, train_data,
        valid_sets=[valid_data],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(20)],
        num_boost_round=params.get("num_iterations", 300),
    )

    # 特征重要性
    importance = pd.DataFrame({
        "feature": factor_names,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)

    # 保存
    model.save_model(str(MODEL_DIR / "gold_silver_classifier.txt"))
    importance.to_parquet(MODEL_DIR / "gold_silver_importance.parquet")

    stats = {
        "model_type": "multiclass",
        "n_features": len(factor_names),
        "n_estimators": model.current_iteration(),
        "train_samples": len(X_train),
        "valid_logloss": model.best_score.get("valid_0", {}).get("multi_logloss", None),
    }

    logger.info(f"  完成: {model.current_iteration()} trees")
    return model, importance, stats


# ═══════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", choices=["a_stock", "gold_silver", "all"],
                        default="all")
    parser.add_argument("--quick", action="store_true", help="快速测试")
    parser.add_argument("--symbols", type=int, default=0,
                        help="限制股票数 (0=全部)")
    args = parser.parse_args()

    if args.assets in ("a_stock", "all"):
        syms = None
        if args.symbols > 0:
            from src.data.symbols import get_today_tradable_pool
            try:
                pool, _ = get_today_tradable_pool()
                syms = pool["symbol"].tolist()[:args.symbols]
            except Exception:
                syms = ["000001", "600036", "600519", "000858", "002415",
                        "300750", "601318", "000333", "600900", "002594"][:args.symbols]
        train_a_stock_ranker(symbols=syms, quick=args.quick)

    if args.assets in ("gold_silver", "all"):
        train_gold_silver_classifier(quick=args.quick)
