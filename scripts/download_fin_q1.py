#!/usr/bin/env python3
"""下载全量Q1财报 — akshare版, 断点续传"""
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, pandas as pd, akshare as ak
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('fin_ak')

end=date.today();start=end-timedelta(days=365*3)
syms=sorted(read_daily_bars(A_STOCK_DIR,start_date=start,end_date=end,market='a_stock')['symbol'].unique())

cf='.cache_fin_infer.parquet'
try:
    old=pd.read_parquet(cf); done=set(old['symbol'].unique()); rows=old.to_dict('records')
except: done=set(); rows=[]

todo=[s for s in syms if s not in done]
logger.info(f'全量:{len(syms)} 已有:{len(done)} 待拉:{len(todo)}')
if not todo:
    logger.info('✅ 全部完成!')
    sys.exit(0)

t0=time.time()
for i,code in enumerate(todo):
    cs=str(code).zfill(6)
    try:
        df=ak.stock_financial_analysis_indicator(symbol=cs, start_year="2026")
        if df is not None and not df.empty:
            r=df.iloc[-1]
            rows.append({'symbol':cs,
                'pubDate':date.today(),'statDate':date.today(),
                'roe':float(r.get('净资产收益率(%)',np.nan))/100 if r.get('净资产收益率(%)') else np.nan,
                'gpMargin':float(r.get('主营业务利润率(%)',np.nan))/100 if r.get('主营业务利润率(%)') else np.nan,
                'npMargin':float(r.get('销售净利率(%)',np.nan))/100 if r.get('销售净利率(%)') else np.nan,
                'netProfit':float(r.get('净利润(元)',np.nan)) if r.get('净利润(元)') else np.nan,
                'debt_ratio':float(r.get('资产负债率(%)',np.nan))/100 if r.get('资产负债率(%)') else np.nan,
                'CFOtoNP':float(r.get('经营现金净流量与净利润的比率(%)',np.nan))/100 if r.get('经营现金净流量与净利润的比率(%)') else np.nan,
                'eps':float(r.get('摊薄每股收益(元)',np.nan)) if r.get('摊薄每股收益(元)') else np.nan,
            })
    except: pass
    time.sleep(1.2)  # 反爬保护
    if (i+1)%200==0:
        pd.DataFrame(rows).to_parquet(cf)
        logger.info(f'[{i+1}/{len(todo)}] {len(rows)}只 {(time.time()-t0)/60:.0f}min')

pd.DataFrame(rows).to_parquet(cf)
logger.info(f'完成:{len(rows)}只 ({(time.time()-t0)/60:.0f}min)')
logger.info(f'覆盖率:{len(rows)}/{len(syms)}')
