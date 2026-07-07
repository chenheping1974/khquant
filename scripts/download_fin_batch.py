#!/usr/bin/env python3
"""全量财报下载 — 东财批量接口, 6000只/次, 5秒"""
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, akshare as ak

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('batch')

# 8个季度
quarters = ['20230331','20230630','20230930','20231231',
            '20240331','20240630','20240930','20241231',
            '20250331','20250630','20250930','20251231','20260331']

all_data = []
t0 = time.time()
for q in quarters:
    try:
        df = ak.stock_yjbb_em(date=q)
        if df is not None and not df.empty:
            df = df.rename(columns={
                '股票代码':'symbol','每股收益':'eps','营业总收入-营业总收入':'revenue',
                '营业总收入-同比增长':'revenue_growth','净利润-净利润':'net_profit',
                '净利润-同比增长':'profit_growth','净资产收益率':'roe',
                '销售毛利率':'gross_margin','每股经营现金流量':'ocf_per_share',
                '所处行业':'industry'
            })
            df['symbol'] = df['symbol'].astype(str).str.zfill(6)
            df['report_date'] = f"{q[:4]}-{q[4:6]}-{q[6:]}"
            cols = ['symbol','report_date','eps','revenue','revenue_growth',
                    'net_profit','profit_growth','roe','gross_margin','ocf_per_share','industry']
            all_data.append(df[[c for c in cols if c in df.columns]])
            logger.info(f'{q}: {len(df)}只 ({time.time()-t0:.0f}s)')
    except Exception as e:
        logger.warning(f'{q}: {e}')
    time.sleep(1)

result = pd.concat(all_data, ignore_index=True)
result.to_parquet('.cache_fin_batch.parquet')
n = result['symbol'].nunique()
logger.info(f'完成: {len(result)}条 {n}只 ({(time.time()-t0):.0f}s)')
