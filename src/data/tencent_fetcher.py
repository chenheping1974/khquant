"""
腾讯财经数据接口

免费、无需认证、稳定。
提供: PE(TTM), PB, 总股本, 流通股本 — 用于基本面因子。

字段映射 (88字段中关键部分):
  [1]  名称
  [3]  最新价
  [39] PE(TTM)
  [46] PB
  [44] 总市值 (有时为空, 改用 price*shares 计算)
  [72] 总股本
  [73] 流通股本
  [63] 股息率
  [31] 涨跌额
  [32] 涨跌幅

用法:
    df = fetch_valuation_batch()  # 全量
    df = fetch_valuation_batch(symbols=['600519','000001'])  # 指定
"""
import logging
import time
from typing import Optional, List

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

BASE_URL = "https://web.sqt.gtimg.cn/q="
BATCH_SIZE = 50
REQUEST_DELAY = 0.3  # 批次间延迟(秒)


def _to_tencent_code(code: str) -> str:
    """'000001' → 'sz000001', '600519' → 'sh600519'"""
    code = str(code).zfill(6)
    return f"sh{code}" if code.startswith("6") else f"sz{code}"


def fetch_valuation_batch(
    symbols: Optional[List[str]] = None,
    max_symbols: Optional[int] = None,
) -> pd.DataFrame:
    """
    批量获取A股估值数据 (PE/PB/股本)。

    Parameters
    ----------
    symbols : list or None
        股票代码列表, None=全市场 (600/000/300开头)
    max_symbols : int or None
        限制数量

    Returns
    -------
    DataFrame: symbol, pe_ttm, pb, total_shares, market_cap
    """
    if symbols is None:
        symbols = _all_a_share_codes()
    if max_symbols:
        symbols = symbols[:max_symbols]

    logger.info(f"腾讯财经: 获取 {len(symbols)} 只估值数据...")
    t0 = time.time()
    all_rows = []

    for i in range(0, len(symbols), BATCH_SIZE):
        batch = symbols[i:i + BATCH_SIZE]
        tc_codes = [_to_tencent_code(c) for c in batch]
        url = f"{BASE_URL}{','.join(tc_codes)}"

        try:
            r = requests.get(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            r.encoding = "gbk"

            for line in r.text.strip().split("\n"):
                if "=" not in line or "pv_none" in line:
                    continue
                fields = line.split('"')[1].split("~")
                if len(fields) < 74:
                    continue

                code = fields[2]  # 6位代码
                name = fields[1]
                price = _safe_float(fields[3])
                pe = _safe_float(fields[39])
                pb = _safe_float(fields[46])
                shares = _safe_float(fields[72])  # 总股本(股)
                div_yield = _safe_float(fields[63])

                if price is None or price <= 0:
                    continue

                # 市值 = 股价 × 总股本
                market_cap = price * shares if shares else None

                all_rows.append({
                    "symbol": code,
                    "name": name,
                    "close": price,
                    "pe_ttm": pe,
                    "pb": pb,
                    "total_shares": shares,
                    "market_cap": market_cap,
                    "dividend_yield": div_yield,
                })

        except Exception as e:
            logger.debug(f"  批次 {i}-{i+BATCH_SIZE}: {e}")

        # 进度
        if (i + BATCH_SIZE) % 500 == 0:
            logger.info(f"  [{min(i+BATCH_SIZE, len(symbols))}/{len(symbols)}]")

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(all_rows)

    # 清洗
    if not df.empty:
        # 负PE无意义
        df.loc[df["pe_ttm"] < 0, "pe_ttm"] = np.nan
        # 衍生因子
        df["earnings_yield"] = 1.0 / df["pe_ttm"].replace(0, np.nan)  # E/P
        df["log_market_cap"] = np.log(df["market_cap"].fillna(1e10))

    elapsed = time.time() - t0
    logger.info(f"  估值完成: {len(df)} 只, {elapsed:.1f}s")
    return df


def _all_a_share_codes() -> List[str]:
    """生成全A股代码列表"""
    codes = []
    for code in range(1, 1000000):
        c = str(code).zfill(6)
        if c.startswith(("60", "00", "30")):
            codes.append(c)
    return codes


def _safe_float(val: str) -> Optional[float]:
    """安全转float, 空字符串→None"""
    if not val or val.strip() == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
