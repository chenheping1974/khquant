#!/usr/bin/env python3
"""拉取缺失的A股数据 — 续传模式"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.data.fetcher import _fetch_sina_kline, _to_sina_symbol
from src.data.storage import write_daily_bars, get_unique_symbols
from config import A_STOCK_DIR

# 获取缺失股票列表
pool = pd.read_parquet('data/tradable_pool.parquet')
pool_syms = set(pool['symbol'].str.zfill(6).tolist())
existing = set(get_unique_symbols(A_STOCK_DIR))
missing = sorted(pool_syms - existing)
print(f'可交易池: {len(pool_syms)}, 已有: {len(existing)}, 待拉取: {len(missing)}')

batch_size = 200
all_dfs = []
t0 = time.time()
total_rows = 0
empty_count = 0
error_count = 0

for i, code in enumerate(missing):
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
        error_count += 1

    # 每batch_size保存一批
    if len(all_dfs) >= batch_size:
        batch_df = pd.concat(all_dfs, ignore_index=True)
        write_daily_bars(batch_df, A_STOCK_DIR, 'a_stock')
        elapsed = (time.time()-t0)/60
        pct = (i+1)/len(missing)*100
        print(f'  [{i+1}/{len(missing)}] {pct:.1f}% | {total_rows}行 | {elapsed:.1f}min | 连续空:{empty_count}', flush=True)
        all_dfs = []

    # 连续500只无数据才停 (之前的200太低)
    if empty_count >= 500:
        print(f'  连续{empty_count}只无数据, 判定无更多有效股票, 停止', flush=True)
        break

    time.sleep(0.08)

# 最后一批
if all_dfs:
    batch_df = pd.concat(all_dfs, ignore_index=True)
    write_daily_bars(batch_df, A_STOCK_DIR, 'a_stock')

elapsed = (time.time()-t0)/60
existing2 = set(get_unique_symbols(A_STOCK_DIR))
print(f'完成: {total_rows}行, {len(existing2)-len(existing)}只新增, {elapsed:.1f}min', flush=True)
