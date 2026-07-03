#!/usr/bin/env python3
"""Full historical data fetch — saves incrementally"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.fetcher import _fetch_sina_kline, _to_sina_symbol
from src.data.storage import write_daily_bars
from config import A_STOCK_DIR

pool = pd.read_parquet('data/tradable_pool.parquet')
syms = sorted(pool['symbol'].str.zfill(6).tolist())
stocks = [s for s in syms if s.startswith(('60','00','30'))]
print(f'Target: {len(stocks)} A-shares', flush=True)

batch_size = 250
all_dfs = []
t0 = time.time()
empty_count = 0
total_rows = 0

for i, code in enumerate(stocks):
    try:
        df = _fetch_sina_kline(_to_sina_symbol(code), datalen=5000)
        if df is not None and not df.empty:
            df['trade_date'] = pd.to_datetime(df['trade_date']).dt.date
            all_dfs.append(df)
            total_rows += len(df)
            empty_count = 0
        else:
            empty_count += 1
    except Exception:
        empty_count += 1

    # Save batch incrementally
    if len(all_dfs) >= batch_size:
        batch_df = pd.concat(all_dfs, ignore_index=True)
        write_daily_bars(batch_df, A_STOCK_DIR, 'a_stock')
        elapsed = (time.time()-t0)/60
        print(f'  [{i+1}/{len(stocks)}] {total_rows} rows, {elapsed:.1f}min', flush=True)
        all_dfs = []

    if empty_count >= 200:
        print(f'  Stopped: {empty_count} empties', flush=True)
        break

    time.sleep(0.08)

# Final batch
if all_dfs:
    batch_df = pd.concat(all_dfs, ignore_index=True)
    write_daily_bars(batch_df, A_STOCK_DIR, 'a_stock')

elapsed = (time.time()-t0)/60
print(f'Done: {total_rows} rows, {elapsed:.1f}min', flush=True)
