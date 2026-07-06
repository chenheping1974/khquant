#!/usr/bin/env python3
"""每日增量数据拉取 — 从新浪获取最近N天K线, 写入khquant-data仓库"""
import sys, os, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import requests
from datetime import date
from config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fetch")

SINA_URL = "https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData"
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"}
STOCKS = 0  # 0=全部
DATALEN = 5   # 最近5天 (增量)

# 取股票列表 (从数据仓库已有数据)
a_stock_dir = DATA_DIR / "a_stock"
existing = set()
for f in a_stock_dir.rglob("data.parquet"):
    try:
        df = pd.read_parquet(f, columns=["symbol"])
        existing.update(df["symbol"].unique().tolist())
    except: pass

# 快速检查: 当月分区最近日期, 如果已是今天则跳过
now = pd.Timestamp.now()
latest_partition = a_stock_dir / f"year={now.year}" / f"month={now.month}" / "data.parquet"
need_fetch = True
if latest_partition.exists():
    try:
        df = pd.read_parquet(latest_partition, columns=["trade_date"])
        max_date = pd.to_datetime(df["trade_date"].max()).date()
        if max_date >= date.today():  # 日期比较, 与UTC无关
            logger.info(f"数据已是最新 ({max_date}), 跳过")
            need_fetch = False
    except: pass

if need_fetch:
    symbols = sorted(existing)
    if STOCKS > 0: symbols = symbols[:STOCKS]
    logger.info(f"数据仓库 {len(existing)} 只, 拉取 {len(symbols)} 只增量")
else:
    symbols = []

from src.data.storage import write_daily_bars

batch = []
total = 0
for i, code in enumerate(symbols):
    code = str(code).zfill(6)
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        r = requests.get(SINA_URL, params={"symbol": f"{prefix}{code}", "scale": "240", "ma": "no", "datalen": str(DATALEN)},
                        headers=HEADERS, timeout=10)
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            for d in data:
                batch.append({"trade_date": d["day"], "symbol": code,
                             "open": float(d["open"]), "high": float(d["high"]),
                             "low": float(d["low"]), "close": float(d["close"]),
                             "volume": float(d["volume"])})
    except: pass

    # 每100只保存一次 (断点续传)
    if len(batch) >= 500:
        df = pd.DataFrame(batch); write_daily_bars(df, a_stock_dir, "a_stock")
        total += len(df); batch = []
        logger.info(f"  [{i+1}/{len(symbols)}] 已存{total}条")

    time.sleep(0.05)

# 最后一批
if batch:
    df = pd.DataFrame(batch); write_daily_bars(df, a_stock_dir, "a_stock")
    total += len(df)

if total > 0:
    logger.info(f"写入完成: {total}条, {len(symbols)}只")
else:
    logger.warning("无新数据 (可能休市)")
