#!/usr/bin/env python3
"""baostock下载财报 — 稳定, 不会被封"""
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

if not hasattr(pd.DataFrame,'append'):
    pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))
import baostock as bs

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('fin_bs')

end=date.today();start=end-timedelta(days=365*3)
syms=sorted(read_daily_bars(A_STOCK_DIR,start_date=start,end_date=end,market='a_stock')['symbol'].unique())

cf='.cache_fin_infer.parquet'
try:
    old=pd.read_parquet(cf); done=set(old['symbol'].unique()); rows=old.to_dict('records')
except: done=set(); rows=[]
todo=[s for s in syms if s not in done]
logger.info(f'全量:{len(syms)} 已有:{len(done)} 待拉:{len(todo)}')
if not todo: logger.info('✅ 全部完成!'); sys.exit(0)

bs.login(); t0=time.time()
for i,code in enumerate(todo):
    cs=str(code).zfill(6);pf='sh' if cs.startswith('6') else 'sz';fc=f'{pf}.{cs}'
    # 利润表 (2023-2026)
    for y in range(2023,2027):
        for q in [1,2,3,4]:
            try:
                rs=bs.query_profit_data(code=fc,year=y,quarter=q);df=rs.get_data()
                if df is not None and not df.empty:
                    rows.append({'symbol':cs,'pubDate':pd.to_datetime(df['pubDate'].iloc[0]).date(),'statDate':pd.to_datetime(df['statDate'].iloc[0]).date(),'roe':float(df['roeAvg'].iloc[0]) if df['roeAvg'].iloc[0]!='' else np.nan,'gpMargin':float(df['gpMargin'].iloc[0]) if df['gpMargin'].iloc[0]!='' else np.nan,'npMargin':float(df['npMargin'].iloc[0]) if df['npMargin'].iloc[0]!='' else np.nan,'netProfit':float(df['netProfit'].iloc[0]) if df['netProfit'].iloc[0]!='' else np.nan,'eps':float(df['epsTTM'].iloc[0]) if df['epsTTM'].iloc[0]!='' else np.nan})
            except: pass; time.sleep(0.01)
    # 资产负债表
    for y in range(2023,2027):
        for q in [1,2,3,4]:
            try:
                rs=bs.query_balance_data(code=fc,year=y,quarter=q);df=rs.get_data()
                if df is not None and not df.empty:
                    sd=pd.to_datetime(df['statDate'].iloc[0]).date()
                    for r in rows:
                        if r['symbol']==cs and r['statDate']==sd: r['debt_ratio']=float(df['liabilityToAsset'].iloc[0]) if df['liabilityToAsset'].iloc[0]!='' else np.nan
            except: pass; time.sleep(0.01)
    # 现金流
    for y in range(2023,2027):
        for q in [1,2,3,4]:
            try:
                rs=bs.query_cash_flow_data(code=fc,year=y,quarter=q);df=rs.get_data()
                if df is not None and not df.empty:
                    sd=pd.to_datetime(df['statDate'].iloc[0]).date()
                    for r in rows:
                        if r['symbol']==cs and r['statDate']==sd: r['CFOtoNP']=float(df['CFOToNP'].iloc[0]) if df['CFOToNP'].iloc[0]!='' else np.nan
            except: pass; time.sleep(0.01)
    if (i+1)%100==0:
        pd.DataFrame(rows).to_parquet(cf); n=pd.DataFrame(rows)['symbol'].nunique()
        logger.info(f'[{i+1}/{len(todo)}] {len(rows)}条 {n}只 {(time.time()-t0)/60:.0f}min')

pd.DataFrame(rows).to_parquet(cf); bs.logout()
n=pd.read_parquet(cf)['symbol'].nunique()
logger.info(f'完成:{len(rows)}条 {n}只 ({(time.time()-t0)/60:.0f}min)')
