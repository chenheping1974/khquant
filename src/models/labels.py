"""
标签构建模块

支持多种标签类型:
  - return: 未来N日收益率 (Ranking模型默认)
  - excess: 超额收益 (相对沪深300)
  - binary: 涨跌二分类
  - sharpe: 夏普标签 (收益/波动)

用法:
    from src.models.labels import build_labels
    y, meta = build_labels(df, horizons=[5,10,20], label_type="return")
"""
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


def build_labels(
    price_df: pd.DataFrame,
    horizons: List[int] = None,
    label_type: str = "return",
    benchmark_df: Optional[pd.DataFrame] = None,
    min_future_bars: int = 3,
) -> Tuple[pd.DataFrame, Dict]:
    """
    从价格数据构建训练标签。

    Parameters
    ----------
    price_df : DataFrame
        columns: trade_date, symbol, close
        必须是后复权价格！
    horizons : list[int]
        未来N日, 默认 [5, 10, 20]
    label_type : str
        "return" | "excess" | "binary" | "sharpe"
    benchmark_df : DataFrame or None
        基准数据 (excess标签时需要), columns: trade_date, close
    min_future_bars : int
        最少需要的未来K线数, 不够的标记为NaN

    Returns
    -------
    labels_df : DataFrame
        columns: trade_date, symbol, label_5d, label_10d, label_20d, label_composite
    meta : dict
        {horizon: {mean, std, coverage}}
    """
    if horizons is None:
        horizons = [5, 10, 20]

    logger.info(f"构建标签: type={label_type}, horizons={horizons}")

    # Pivot: symbol × trade_date 的收盘价矩阵
    close_matrix = price_df.pivot_table(
        index="trade_date", columns="symbol", values="close", aggfunc="last"
    )
    close_matrix = close_matrix.sort_index()

    all_labels = {}
    meta = {}

    for horizon in horizons:
        col_name = f"label_{horizon}d"

        # 未来第N天收盘价
        future_close = close_matrix.shift(-horizon)

        if label_type == "return":
            # 未来N日收益率
            labels = future_close / close_matrix - 1

        elif label_type == "binary":
            # 涨跌分类
            labels = ((future_close / close_matrix - 1) > 0).astype(float)
            labels = labels.replace(0, -1)  # LightGBM二分类: 1=涨, -1=跌

        elif label_type == "excess":
            # 超额收益 (需要基准)
            if benchmark_df is not None:
                bench_close = benchmark_df.set_index("trade_date")["close"].sort_index()
                bench_return = bench_close.shift(-horizon) / bench_close - 1
                stock_return = future_close / close_matrix - 1
                # 广播: 每只股票减基准
                common_dates = stock_return.index.intersection(bench_return.index)
                labels = stock_return.loc[common_dates].subtract(
                    bench_return.loc[common_dates], axis=0
                )
            else:
                logger.warning("无基准数据, 回退到return标签")
                labels = future_close / close_matrix - 1

        elif label_type == "sharpe":
            # 夏普标签: 未来N日收益 / 同期波动率
            future_returns = (future_close / close_matrix - 1)
            # 用过去20天波动率作为分母
            past_returns = close_matrix.pct_change()
            past_vol = past_returns.rolling(20).std() * np.sqrt(252)
            labels = future_returns / (past_vol + 1e-8)

        else:
            raise ValueError(f"未知标签类型: {label_type}")

        # 清洗: 最后N天没有未来数据
        labels = labels.iloc[:-horizon] if horizon > 0 else labels

        # 极端值剪裁
        labels = labels.clip(-0.5, 0.5)  # 限制±50%

        all_labels[col_name] = labels
        meta[horizon] = {
            "mean": float(labels.stack().mean()) if not labels.empty else 0,
            "std": float(labels.stack().std()) if not labels.empty else 0,
            "coverage": int(labels.notna().sum().sum()),
        }

    # 合并: trade_date × symbol 格式
    result_dfs = []
    for col_name, label_matrix in all_labels.items():
        melted = label_matrix.stack().reset_index()
        melted.columns = ["trade_date", "symbol", col_name]
        result_dfs.append(melted)

    labels_df = result_dfs[0]
    for df in result_dfs[1:]:
        labels_df = labels_df.merge(df, on=["trade_date", "symbol"], how="outer")

    # 综合标签: 加权平均
    weights = [0.2, 0.5, 0.3][:len(horizons)]  # 短:中:长
    label_cols = [f"label_{h}d" for h in horizons]
    labels_df["label_composite"] = 0.0
    weight_sum = 0
    for col, w in zip(label_cols, weights):
        if col in labels_df.columns:
            labels_df["label_composite"] += labels_df[col].fillna(0) * w
            weight_sum += w
    if weight_sum > 0:
        labels_df["label_composite"] /= weight_sum

    logger.info(f"标签完成: {len(labels_df)}行, 覆盖{labels_df['symbol'].nunique()}只")
    return labels_df, meta


def get_label_stats(labels_df: pd.DataFrame) -> pd.DataFrame:
    """标签统计: 每只股票的标签分布"""
    label_cols = [c for c in labels_df.columns if c.startswith("label_")]
    if not label_cols:
        return pd.DataFrame()

    stats = labels_df.groupby("symbol")[label_cols].agg(["mean", "std", "count"])
    stats.columns = ["_".join(c) for c in stats.columns]
    return stats.sort_values(f"{label_cols[0]}_mean", ascending=False)


def check_label_quality(labels_df: pd.DataFrame, horizon: int = 10) -> Dict:
    """
    标签质量检查。

    好的标签应该:
    - 均值接近0 (没有系统性偏差)
    - 有一定方差 (区分度)
    - 自相关低 (信息不重复)
    """
    col = f"label_{horizon}d"
    if col not in labels_df.columns:
        return {}

    values = labels_df[col].dropna()
    # 按日期分组
    daily_mean = labels_df.groupby("trade_date")[col].mean()

    return {
        "horizon": horizon,
        "count": len(values),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "sharpe": float(values.mean() / values.std() * np.sqrt(252)) if values.std() > 0 else 0,
        "daily_autocorr": float(daily_mean.autocorr()) if len(daily_mean) > 1 else 0,
        "pct_positive": float((values > 0).mean()),
    }
