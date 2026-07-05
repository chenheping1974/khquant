"""
GitHub Raw 存储后端

从 GitHub 仓库直接 HTTP 读取 Parquet — 不需要 clone 数据仓库!

用法 (其他项目):
    import pandas as pd
    df = pd.read_parquet(
        "https://raw.githubusercontent.com/USER/khquant-data/main/"
        "a_stock/year=2026/month=07/data.parquet"
    )

原理: pyarrow 原生支持 HTTP URL, pandas 直接读
本地缓存: 避免重复下载同一分区
"""
import json
import logging
import time
from pathlib import Path
from datetime import date
from typing import Optional, List
from urllib.request import urlopen, Request

import pandas as pd

from config import DATA_REPO

logger = logging.getLogger(__name__)

BASE = "https://raw.githubusercontent.com/{owner}/{name}/{branch}"

# GitHub API 用于列出文件 (无认证, 60次/小时, 够用)
API_BASE = "https://api.github.com/repos/{owner}/{name}/git/trees/{branch}?recursive=1"

_file_index = None  # 缓存文件列表
_index_time = 0.0
_INDEX_TTL = 3600  # 1小时


def _get_file_list(owner: str, name: str, branch: str) -> dict:
    """获取仓库文件列表 (缓存1小时)"""
    global _file_index, _index_time
    now = time.time()
    if _file_index is not None and (now - _index_time) < _INDEX_TTL:
        return _file_index

    url = API_BASE.format(owner=owner, name=name, branch=branch)
    try:
        req = Request(url, headers={"Accept": "application/vnd.github+json"})
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
        _file_index = {}
        for item in data.get("tree", []):
            if item["path"].endswith(".parquet"):
                _file_index[item["path"]] = True
        _index_time = now
        logger.info(f"[github] 文件列表: {len(_file_index)} 个 parquet")
    except Exception as e:
        logger.warning(f"[github] 获取文件列表失败: {e}")
        _file_index = {}
    return _file_index


def _partition_url(owner: str, name: str, branch: str,
                   market: str, year: int, month: int) -> str:
    """构造分区 URL"""
    return (f"{BASE.format(owner=owner, name=name, branch=branch)}/"
            f"{market}/year={year}/month={month}/data.parquet")


def read_daily_bars(
    market: str = "a_stock",
    symbols: Optional[List[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    从 GitHub Raw 读取日线数据 (HTTP, 不需 clone)

    自动缓存到 DATA_REPO['cache_dir']
    """
    owner = DATA_REPO["owner"]
    name = DATA_REPO["name"]
    branch = DATA_REPO["branch"]
    cache_dir = Path(DATA_REPO["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not owner:
        logger.error("[github] 未配置 DATA_REPO_OWNER")
        return pd.DataFrame()

    # 获取文件列表
    file_index = _get_file_list(owner, name, branch)
    if not file_index:
        # 无文件列表 → 回退: 按日期范围试读
        file_index = _generate_index(market, start_date, end_date)

    # 筛选匹配的分区
    dfs = []
    for path in sorted(file_index):
        if not path.startswith(f"{market}/"):
            continue
        parts = _parse_path(path, market)
        if parts is None:
            continue
        yr, mo = parts
        if start_date and (yr, mo) < (start_date.year, start_date.month):
            continue
        if end_date and (yr, mo) > (end_date.year, end_date.month):
            continue

        url = f"{BASE.format(owner=owner, name=name, branch=branch)}/{path}"
        df = _read_url_cached(url, cache_dir, path, columns)
        if df.empty:
            continue

        if symbols is not None and "symbol" in df.columns:
            df = df[df["symbol"].isin(symbols)]
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    if "trade_date" in result.columns:
        result["trade_date"] = pd.to_datetime(result["trade_date"]).dt.date
        if start_date:
            result = result[result["trade_date"] >= start_date]
        if end_date:
            result = result[result["trade_date"] <= end_date]

    return result.sort_values(["symbol", "trade_date"]).reset_index(drop=True)


def _read_url_cached(url: str, cache_dir: Path, path: str,
                     columns: Optional[List[str]] = None) -> pd.DataFrame:
    """读 URL, 优先本地缓存"""
    cache_file = cache_dir / path

    if cache_file.exists():
        try:
            return pd.read_parquet(cache_file, columns=columns)
        except Exception:
            cache_file.unlink(missing_ok=True)

    try:
        df = pd.read_parquet(url, columns=columns)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, compression="snappy", index=False)
        return df
    except Exception as e:
        logger.debug(f"[github] 读取失败 {url}: {e}")
        return pd.DataFrame()


def _parse_path(path: str, market: str) -> Optional[tuple]:
    """'a_stock/year=2024/month=3/data.parquet' → (2024, 3)"""
    try:
        rest = path[len(market) + 1:]  # 'year=2024/month=3/data.parquet'
        parts = {}
        for seg in rest.split("/"):
            if "=" in seg:
                k, v = seg.split("=")
                parts[k] = int(v)
        return (parts.get("year"), parts.get("month"))
    except (ValueError, TypeError):
        return None


def _generate_index(market: str, start: Optional[date],
                    end: Optional[date]) -> dict:
    """无 API 时, 按日期范围生成可能的路径"""
    from datetime import date as dt_date, timedelta
    if start is None:
        start = dt_date.today() - timedelta(days=365 * 3)
    if end is None:
        end = dt_date.today()

    paths = {}
    current = dt_date(start.year, start.month, 1)
    end_month = dt_date(end.year, end.month, 1)
    while current <= end_month:
        path = f"{market}/year={current.year}/month={current.month}/data.parquet"
        paths[path] = True
        if current.month == 12:
            current = dt_date(current.year + 1, 1, 1)
        else:
            current = dt_date(current.year, current.month + 1, 1)
    return paths


def get_latest_date(market: str = "a_stock") -> Optional[date]:
    """获取数据最新日期 (从当前月份分区读)"""
    from datetime import date as dt_date
    owner = DATA_REPO["owner"]
    name = DATA_REPO["name"]
    branch = DATA_REPO["branch"]
    cache_dir = Path(DATA_REPO["cache_dir"])

    today = dt_date.today()
    for offset in range(3):
        yr, mo = today.year, today.month - offset
        if mo <= 0:
            yr -= 1
            mo += 12
        url = _partition_url(owner, name, branch, market, yr, mo)
        path = f"{market}/year={yr}/month={mo}/data.parquet"
        df = _read_url_cached(url, cache_dir, path, columns=["trade_date"])
        if not df.empty:
            return pd.to_datetime(df["trade_date"].max()).date()
    return None


def get_unique_symbols(market: str = "a_stock") -> List[str]:
    """获取所有股票代码"""
    owner = DATA_REPO["owner"]
    name = DATA_REPO["name"]
    branch = DATA_REPO["branch"]
    cache_dir = Path(DATA_REPO["cache_dir"])

    file_index = _get_file_list(owner, name, branch)
    if not file_index:
        logger.warning("[github] 无法获取文件列表, 无法获取股票代码")
        return []

    symbols = set()
    for path in file_index:
        if not path.startswith(f"{market}/"):
            continue
        url = f"{BASE.format(owner=owner, name=name, branch=branch)}/{path}"
        df = _read_url_cached(url, cache_dir, path, columns=["symbol"])
        if not df.empty:
            symbols.update(df["symbol"].unique().tolist())

    return sorted(symbols)
