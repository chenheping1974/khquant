"""
A股股票池管理 + 垃圾股过滤器

数据源: 新浪实时行情 (hq.sinajs.cn)

Pipeline:
  1. 生成全量代码范围 → 新浪实时行情验证有效性
  2. 过滤垃圾股: ST / 低价 / 停牌
  3. 输出可交易池
"""
import logging
import pandas as pd
from datetime import date
from typing import Tuple, List

from config import GARBAGE_FILTER

logger = logging.getLogger(__name__)


def get_stock_list() -> pd.DataFrame:
    """
    全市场A股列表 (新浪实时行情验证)。

    生成全量代码范围 → 分批查询新浪实时行情 → 有效代码即为交易中股票。
    """
    logger.info("A股列表 (Sina实时行情)...")

    # 生成全量代码
    codes = _generate_all_codes()
    logger.info(f"  代码池: {len(codes)} 只")

    # 分批查询新浪实时行情
    from src.data.fetcher import fetch_sina_quotes, _to_sina_symbol

    all_dfs = []
    batch_size = 800
    for i in range(0, min(len(codes), 5000), batch_size):
        batch = codes[i:i + batch_size]
        sina_syms = [_to_sina_symbol(c) for c in batch]
        try:
            quotes = fetch_sina_quotes(sina_syms)
            if not quotes.empty:
                all_dfs.append(quotes)
        except Exception:
            continue

    if not all_dfs:
        logger.warning("新浪行情无数据, 回退纯代码列表")
        return pd.DataFrame({
            "symbol": codes, "name": "unknown",
            "close": None, "volume": None, "amount": None,
            "is_st": False,
        })

    df = pd.concat(all_dfs, ignore_index=True)
    df["is_st"] = df.get("name", "").str.contains(r"\*?ST|退市", na=False)

    # 从新浪symbol提取A股代码
    df["symbol"] = df["symbol"].str[2:]  # sh600036 → 600036

    logger.info(f"  有效: {len(df)} 只")
    return df


def filter_garbage(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """过滤垃圾股 → (可交易池, 剔除列表)"""
    cfg = GARBAGE_FILTER
    total = len(df)
    excluded = []
    mask = pd.Series(True, index=df.index)

    if cfg["exclude_st"] and "is_st" in df.columns:
        st_mask = df["is_st"]
        excluded.append(("ST/*ST", st_mask.sum()))
        mask &= ~st_mask

    if cfg["min_price"] and "close" in df.columns:
        low = (df["close"] < cfg["min_price"]) & df["close"].notna() & (df["close"] > 0)
        excluded.append((f"股价<{cfg['min_price']}元", low.sum()))
        mask &= ~low

    if cfg["exclude_suspended"] and "volume" in df.columns:
        susp = (df["volume"] <= 0) | df["close"].isna()
        excluded.append(("停牌/无成交", susp.sum()))
        mask &= ~susp

    valid = df[mask].copy()
    rejected = df[~mask].copy()

    logger.info(f"过滤: {len(valid)}/{total} 入池")
    for reason, count in excluded:
        logger.info(f"  - {reason}: {count}")

    return valid, rejected


def get_today_tradable_pool() -> Tuple[pd.DataFrame, dict]:
    """一站式: 今日可交易池 + 摘要"""
    all_ = get_stock_list()
    pool, rejected = filter_garbage(all_)
    return pool, {
        "date": date.today().isoformat(),
        "total_all": len(all_),
        "tradable": len(pool),
        "rejected": len(rejected),
    }


def _generate_all_codes() -> List[str]:
    codes = []
    for i in range(600000, 609999):
        codes.append(str(i))
    for i in range(688000, 689999):
        codes.append(str(i))
    for i in range(1, 4999):
        codes.append(str(i).zfill(6))
    for i in range(300000, 301999):
        codes.append(str(i))
    return sorted(set(codes))
