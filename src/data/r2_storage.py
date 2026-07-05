"""
Cloudflare R2 存储后端 (S3-compatible)

替代本地 Parquet 文件系统，提供相同接口:
  - read_daily_bars / write_daily_bars
  - get_latest_date / get_unique_symbols

R2 免费额度: 10GB 存储 + 无出口费
本地缓存: 避免重复从 R2 下载同一分区
"""
import io
import time
import logging
from pathlib import Path
from datetime import date
from typing import Optional, List, Dict

import boto3
import pandas as pd
import pyarrow.parquet as pq
from botocore.config import Config

from config import R2_CONFIG

logger = logging.getLogger(__name__)

# ── S3 client (lazy init) ──────────────────────────────
_s3_client = None


def _get_s3():
    global _s3_client
    if _s3_client is None:
        endpoint = R2_CONFIG["endpoint_template"].format(
            account_id=R2_CONFIG["account_id"]
        )
        _s3_client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=R2_CONFIG["access_key"],
            aws_secret_access_key=R2_CONFIG["secret_key"],
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3, "mode": "standard"},
            ),
        )
    return _s3_client


def _bucket() -> str:
    return R2_CONFIG["bucket"]


def _cache_dir() -> Path:
    d = Path(R2_CONFIG["cache_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── 路径映射 ────────────────────────────────────────────

def _s3_key(market: str, partition_path: str) -> str:
    """将本地分区路径映射为 R2 key"""
    return f"{market}/{partition_path}"


def _cache_path(s3_key: str) -> Path:
    """R2 key → 本地缓存路径"""
    return _cache_dir() / s3_key


# ═══════════════════════════════════════════════════════
#  读
# ═══════════════════════════════════════════════════════

def read_daily_bars(
    market: str = "a_stock",
    symbols: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    从 R2 读取日线数据。

    策略:
      1. 列出符合日期范围的 R2 对象
      2. 优先读本地缓存, 缓存未命中则从 R2 下载
      3. 合并返回
    """
    s3 = _get_s3()
    bucket = _bucket()
    prefix = f"{market}/year="

    # 列出所有分区
    paginator = s3.get_paginator("list_objects_v2")
    all_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("data.parquet"):
                all_keys.append(obj["Key"])

    if not all_keys:
        logger.warning(f"[R2] {market}: 无数据")
        return pd.DataFrame()

    # 日期过滤 (从 key 名解析分区)
    filtered_keys = []
    for key in all_keys:
        parts = _parse_key_parts(key)
        if parts is None:
            continue
        yr, mo = parts
        if start_date and (yr, mo) < (start_date.year, start_date.month):
            continue
        if end_date and (yr, mo) > (end_date.year, end_date.month):
            continue
        filtered_keys.append(key)

    if not filtered_keys:
        return pd.DataFrame()

    # 读取每个分区
    dfs = []
    for key in filtered_keys:
        df = _read_partition(s3, bucket, key, columns)
        if df.empty:
            continue
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

    return result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _read_partition(s3, bucket: str, key: str,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
    """读单个分区 (缓存优先)"""
    cache_file = _cache_path(key)

    # 缓存命中
    if cache_file.exists():
        try:
            cols = columns
            if cols is not None and "trade_date" not in cols:
                cols = cols + ["trade_date"]
            if cols is not None and "symbol" not in cols:
                cols = cols + ["symbol"]
            return pd.read_parquet(cache_file, columns=cols)
        except Exception:
            cache_file.unlink(missing_ok=True)

    # 从 R2 下载
    try:
        resp = s3.get_object(Bucket=bucket, Key=key)
        data = resp["Body"].read()
        # 写缓存
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(data)
        return pd.read_parquet(io.BytesIO(data), columns=columns)
    except Exception as e:
        logger.warning(f"[R2] 读取失败 {key}: {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════
#  写
# ═══════════════════════════════════════════════════════

def write_daily_bars(df: pd.DataFrame, market: str = "a_stock"):
    """
    增量写入 R2。

    策略:
      1. 按 year/month 分区
      2. 对每个分区: 下载已有数据 → merge → 上传
      3. 更新本地缓存
    """
    if df.empty:
        return

    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df["year"] = pd.to_datetime(df["trade_date"]).dt.year
        df["month"] = pd.to_datetime(df["trade_date"]).dt.month

    s3 = _get_s3()
    bucket = _bucket()
    t0 = time.time()
    total_rows = 0

    for (year, month), group in df.groupby(["year", "month"]):
        partition_key = f"{market}/year={year}/month={month}/data.parquet"
        cache_file = _cache_path(partition_key)
        save_df = group.drop(columns=["year", "month"])

        # 下载已有数据并合并
        try:
            resp = s3.get_object(Bucket=bucket, Key=partition_key)
            existing = pd.read_parquet(io.BytesIO(resp["Body"].read()))
            if "trade_date" in existing.columns:
                existing["trade_date"] = pd.to_datetime(existing["trade_date"]).dt.date
            combined = pd.concat([existing, save_df], ignore_index=True)
            combined = combined.drop_duplicates(
                subset=["trade_date", "symbol"], keep="last"
            )
            combined = combined.sort_values(["symbol", "trade_date"])
        except s3.exceptions.NoSuchKey:
            combined = save_df.sort_values(["symbol", "trade_date"])

        # 上传到 R2
        buf = io.BytesIO()
        combined.to_parquet(buf, compression="snappy", index=False)
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=partition_key, Body=buf.read())

        # 更新本地缓存
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_file, compression="snappy", index=False)

        total_rows += len(save_df)

    elapsed = time.time() - t0
    logger.info(f"[R2] {market}: 写入 {total_rows} 条, {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════

def get_latest_date(market: str = "a_stock") -> Optional[date]:
    """获取 R2 中最新交易日期 (只检查当前月份分区)"""
    from datetime import date as dt_date
    today = dt_date.today()
    s3 = _get_s3()
    bucket = _bucket()

    # 尝试当前月 → 上个月 → 上上个月
    for offset in range(3):
        yr = today.year
        mo = today.month - offset
        if mo <= 0:
            yr -= 1
            mo += 12
        key = f"{market}/year={yr}/month={mo}/data.parquet"
        try:
            df = _read_partition(s3, bucket, key, columns=["trade_date"])
            if not df.empty:
                return pd.to_datetime(df["trade_date"].max()).date()
        except Exception:
            continue
    return None


def get_unique_symbols(market: str = "a_stock") -> List[str]:
    """获取 R2 中所有股票代码"""
    s3 = _get_s3()
    bucket = _bucket()
    prefix = f"{market}/year="

    symbols = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if not obj["Key"].endswith("data.parquet"):
                continue
            try:
                df = _read_partition(s3, bucket, obj["Key"], columns=["symbol"])
                symbols.update(df["symbol"].unique().tolist())
            except Exception:
                continue
    return sorted(symbols)


def _parse_key_parts(key: str) -> Optional[tuple]:
    """从 R2 key 解析 (year, month): 'a_stock/year=2024/month=3/data.parquet' → (2024, 3)"""
    try:
        parts = {}
        for segment in key.split("/"):
            if "=" in segment:
                k, v = segment.split("=")
                parts[k] = int(v)
        return (parts.get("year"), parts.get("month"))
    except (ValueError, TypeError):
        return None


def list_markets() -> List[str]:
    """列出 R2 中所有市场 (前缀)"""
    s3 = _get_s3()
    bucket = _bucket()
    paginator = s3.get_paginator("list_objects_v2")
    prefixes = set()
    for page in paginator.paginate(Bucket=bucket, Prefix="", Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            prefixes.add(p["Prefix"].rstrip("/"))
    return sorted(prefixes)
