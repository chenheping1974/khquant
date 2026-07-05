"""
Parquet 读写层
- 增量写入，按年月分区
- 高效列扫描
- 支持数据校验
- 支持本地 / Cloudflare R2 双后端
"""
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List, Dict
import logging

from config import STORAGE_BACKEND

logger = logging.getLogger(__name__)

# ── GitHub Raw 后端: HTTP 读取, 不需 clone 数据仓库 ──────
if STORAGE_BACKEND == "github":
    from src.data.github_storage import (
        read_daily_bars as _github_read,
        get_latest_date as _github_latest,
        get_unique_symbols as _github_symbols,
    )

    def read_daily_bars(base_dir, symbols=None, start_date=None,
                        end_date=None, columns=None, market="a_stock"):
        return _github_read(market=market, symbols=symbols,
                            start_date=start_date, end_date=end_date,
                            columns=columns)

    def write_daily_bars(df, base_dir, market="a_stock"):
        # GitHub 模式只读不写 (写由 Actions 本地模式完成)
        logger.warning("[github] 只读模式, 跳过写入")
        return

    def get_latest_date(base_dir):
        return _github_latest(base_dir.name)

    def get_unique_symbols(base_dir):
        return _github_symbols(base_dir.name)

# ── R2 后端: 透明替换所有函数 ───────────────────────────
elif STORAGE_BACKEND == "r2":
    from src.data.r2_storage import (
        read_daily_bars as _r2_read_daily_bars,
        write_daily_bars as _r2_write_daily_bars,
        get_latest_date as _r2_get_latest_date,
        get_unique_symbols as _r2_get_unique_symbols,
    )
    # 注意: R2 版本的 market 参数是第一个位置参数 (而非 Path),
    # 调用方使用 config 中的目录路径, 我们需要适配
    _MARKET_MAP = {
        "a_stock": "a_stock",
        "gold_silver": "gold_silver",
        "macro": "macro",
    }

    def _resolve_market(base_dir: Path, market: str) -> str:
        return market

    def read_daily_bars(base_dir: Path, symbols=None, start_date=None,
                        end_date=None, columns=None, market="a_stock"):
        return _r2_read_daily_bars(
            market=market, symbols=symbols,
            start_date=start_date, end_date=end_date, columns=columns,
        )

    def write_daily_bars(df, base_dir: Path, market="a_stock"):
        return _r2_write_daily_bars(df, market=market)

    def get_latest_date(base_dir: Path) -> Optional[date]:
        # 从 base_dir 名称推断 market
        name = base_dir.name
        market = _MARKET_MAP.get(name, name)
        return _r2_get_latest_date(market)

    def get_unique_symbols(base_dir: Path) -> List[str]:
        name = base_dir.name
        market = _MARKET_MAP.get(name, name)
        return _r2_get_unique_symbols(market)

    # 本地兼容函数 (R2 模式下保留, 用于数据迁移等)
    def _local_write_daily_bars(df, base_dir, market):
        _write_daily_bars_impl(df, base_dir, market)

    def _local_read_daily_bars(base_dir, **kwargs):
        return _read_daily_bars_impl(base_dir, **kwargs)

else:
    # 本地模式: 直接使用下面的函数实现
    def write_daily_bars(df, base_dir, market="a_stock"):
        return _write_daily_bars_impl(df, base_dir, market)

    def read_daily_bars(base_dir, symbols=None, start_date=None,
                        end_date=None, columns=None, market="a_stock"):
        return _read_daily_bars_impl(base_dir, symbols=symbols,
                                     start_date=start_date, end_date=end_date,
                                     columns=columns, market=market)

    def get_latest_date(base_dir):
        return _get_latest_date_impl(base_dir)

    def get_unique_symbols(base_dir):
        return _get_unique_symbols_impl(base_dir)


# ═══════════════════════════════════════════════════════
#  本地实现 (local 模式直接使用, R2 模式可通过 _local_* 调用)
# ═══════════════════════════════════════════════════════

def _write_daily_bars_impl(df: pd.DataFrame, base_dir: Path, market: str = "a_stock"):
    """
    增量写入日线数据到分区 Parquet。

    分区规则: {base_dir}/year=YYYY/month=MM/data.parquet

    Parameters
    ----------
    df : DataFrame
        必须包含列: trade_date, symbol, open, high, low, close, volume, amount
    base_dir : Path
        数据根目录
    market : str
        'a_stock' | 'gold_silver' | 'macro'
    """
    if df.empty:
        return

    # 确保日期列是 date 类型
    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df["year"] = pd.to_datetime(df["trade_date"]).dt.year
        df["month"] = pd.to_datetime(df["trade_date"]).dt.month
    elif "date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["date"]).dt.date
        df["year"] = pd.to_datetime(df["date"]).dt.year
        df["month"] = pd.to_datetime(df["date"]).dt.month
        df = df.drop(columns=["date"])
    else:
        raise ValueError("DataFrame 必须包含 'trade_date' 或 'date' 列")

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    for (year, month), group in df.groupby(["year", "month"]):
        partition_dir = base_dir / f"year={year}" / f"month={month}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        file_path = partition_dir / "data.parquet"

        # 去掉分区列
        save_df = group.drop(columns=["year", "month"])

        if file_path.exists():
            # 增量合并：读取已有数据，按 (trade_date, symbol) 去重
            existing = pd.read_parquet(file_path)
            if "trade_date" in existing.columns:
                existing["trade_date"] = pd.to_datetime(existing["trade_date"]).dt.date
            combined = pd.concat([existing, save_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["trade_date", "symbol"], keep="last")
            combined = combined.sort_values(["symbol", "trade_date"])
        else:
            combined = save_df.sort_values(["symbol", "trade_date"])

        table = pa.Table.from_pandas(combined)
        pq.write_table(
            table, file_path,
            compression="snappy",
            row_group_size=100_000,
        )

    logger.info(f"[{market}] 写入 {len(df)} 条记录")


def _read_daily_bars_impl(
    base_dir: Path,
    symbols: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    columns: Optional[List[str]] = None,
    market: str = "a_stock",
) -> pd.DataFrame:
    """
    读取日线数据。

    Parameters
    ----------
    base_dir : Path
        数据根目录
    symbols : list[str] or None
        股票代码列表, None = 全部
    start_date, end_date : date or None
        日期范围
    columns : list[str] or None
        需要的列, None = 全部
    market : str
        市场标识

    Returns
    -------
    DataFrame
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        logger.warning(f"[{market}] 数据目录不存在: {base_dir}")
        return pd.DataFrame()

    # 收集分区路径
    partition_dirs = sorted(base_dir.glob("year=*/month=*/"))
    if not partition_dirs:
        return pd.DataFrame()

    # 日期过滤 → 过滤分区
    filtered_dirs = partition_dirs
    if start_date:
        min_year, min_month = start_date.year, start_date.month
        filtered_dirs = [
            d for d in filtered_dirs
            if _parse_partition(d) >= (min_year, min_month)
        ]
    if end_date:
        max_year, max_month = end_date.year, end_date.month
        filtered_dirs = [
            d for d in filtered_dirs
            if _parse_partition(d) <= (max_year, max_month)
        ]

    if not filtered_dirs:
        return pd.DataFrame()

    # 读取合并
    dfs = []
    for d in filtered_dirs:
        f = d / "data.parquet"
        if not f.exists():
            continue
        read_cols = columns
        if read_cols is not None and "trade_date" not in read_cols:
            read_cols = read_cols + ["trade_date"]
        if read_cols is not None and symbols is not None and "symbol" not in read_cols:
            read_cols = read_cols + ["symbol"]

        df = pd.read_parquet(f, columns=read_cols)

        # 股票过滤
        if symbols is not None and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]

        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)

    # 精确日期过滤
    if "trade_date" in result.columns:
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.date
        if start_date:
            result = result[result["trade_date"] >= start_date]
        if end_date:
            result = result[result["trade_date"] <= end_date]

    result = result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)
    return result


def write_macro(df: pd.DataFrame, base_dir: Path):
    """写入宏观数据 (每条记录是唯一的时间序列)"""
    if STORAGE_BACKEND == "r2":
        return write_daily_bars(df.assign(symbol="MACRO"), base_dir, market="macro")
    return _write_daily_bars_impl(df.assign(symbol="MACRO"), base_dir, market="macro")


def read_macro(base_dir: Path, start_date=None, end_date=None):
    """读取宏观数据"""
    return read_daily_bars(base_dir, start_date=start_date, end_date=end_date, market="macro")


def _get_latest_date_impl(base_dir: Path) -> Optional[date]:
    """获取数据中的最新日期"""
    partition_dirs = sorted(base_dir.glob("year=*/month=*/"), reverse=True)
    for d in partition_dirs:
        f = d / "data.parquet"
        if f.exists():
            df = pd.read_parquet(f, columns=["trade_date"])
            if not df.empty:
                return pd.to_datetime(df["trade_date"].max()).date()
    return None


def get_unique_symbols(base_dir: Path) -> List[str]:
    """获取数据库中所有股票代码"""
    partition_dirs = sorted(base_dir.glob("year=*/month=*/"))
    symbols = set()
    for d in partition_dirs:
        f = d / "data.parquet"
        if f.exists():
            df = pd.read_parquet(f, columns=["symbol"])
            symbols.update(df["symbol"].unique().tolist())
    return sorted(symbols)


def _parse_partition(partition_dir: Path) -> tuple:
    """解析分区目录名: year=2024/month=3 → (2024, 3)"""
    parts = {}
    for p in partition_dir.parts:
        if "=" in p:
            k, v = p.split("=")
            parts[k] = int(v)
    return (parts.get("year", 0), parts.get("month", 0))
