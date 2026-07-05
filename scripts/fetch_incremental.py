#!/usr/bin/env python3
"""每日增量数据拉取 — 从新浪获取最近N天K线, 写入khquant-data仓库"""
import sys, os, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch")

SINA_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
STOCKS = 200  # 拉前200只
DATALEN = 5   # 最近5天 (增量)

# 取股票列表 (从数据仓库已有数据)
a_stock_dir = DATA_DIR / "a_stock"
existing = set()
for f in a_stock_dir.rglob("data.parquet"):
    try:
        df = pd.read_parquet(f, columns=["symbol"])
        existing.update(df["symbol"].unique().tolist())
    except: pass

symbols = sorted(existing)[:STOCKS]
logger.info(f"数据仓库已有 {len(existing)} 只, 拉取前 {len(symbols)} 只增量 (最近{DATALEN}天)")

from src.data.storage import write_daily_bars

rows = []
for i, code in enumerate(symbols):
    code = str(code).zfill(6)
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        r = requests.get(SINA_URL, params={"symbol": f"{prefix}{code}", "scale": "240", "ma": "no", "datalen": str(DATALEN)},
                        headers=HEADERS, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            continue
        for d in data:
            rows.append({"trade_date": d["day"], "symbol": code,
                         "open": float(d["open"]), "high": float(d["high"]),
                         "low": float(d["low"]), "close": float(d["close"]),
                         "volume": float(d["volume"])})
    except: pass
    if (i+1) % 50 == 0: logger.info(f"  [{i+1}/{len(symbols)}] {len(rows)}条")
    time.sleep(0.05)

if rows:
    df = pd.DataFrame(rows)
    write_daily_bars(df, a_stock_dir, "a_stock")
    logger.info(f"写入: {len(rows)}条, {df['symbol'].nunique()}只")
else:
    logger.warning("无新数据 (可能休市)")
