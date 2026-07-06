#!/usr/bin/env python3
"""下载全量Q1财报 — 断点续传, 中断后重跑自动跳过已下载的"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, pandas as pd
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

if not hasattr(pd.DataFrame,'append'):
    pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))
import baostock as bs, logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('fin_resume')

# 股票列表
end=date.today();start=end-timedelta(days=365*3)
syms=sorted(read_daily_bars(A_STOCK_DIR,start_date=start,end_date=end,market='a_stock')['symbol'].unique())

# 已有数据 (断点续传)
cache_file='.cache_fin_infer.parquet'
try:
    existing=pd.read_parquet(cache_file)
    done=set(existing['symbol'].unique())
    rows=existing.to_dict('records')
except:
    done=set(); rows=[]

todo=[s for s in syms if s not in done]
logger.info(f'全量:{len(syms)} 已有:{len(done)} 待拉:{len(todo)}')
if not todo:
    logger.info('✅ 全部完成!')
    sys.exit(0)

bs.login()
import socket
socket.setdefaulttimeout(10)  # 10秒超时
t0=time.time()
for i,code in enumerate(todo):
    # 每200只重新登录
    if i>0 and i%200==0:
        try: bs.logout()
        except: pass
        time.sleep(2)
        bs.login()
        logger.info(f'  重登录 [{i+1}/{len(todo)}]')
    cs=str(code).zfill(6);pf='sh' if cs.startswith('6') else 'sz';fc=f'{pf}.{cs}'
    # 利润表 Q1
    try:
        rs=bs.query_profit_data(code=fc,year=2026,quarter=1);df=rs.get_data()
        if df is not None and not df.empty:
            rows.append({'symbol':cs,'pubDate':pd.to_datetime(df['pubDate'].iloc[0]).date(),
                'statDate':pd.to_datetime(df['statDate'].iloc[0]).date(),
                'roe':float(df['roeAvg'].iloc[0]) if df['roeAvg'].iloc[0]!='' else np.nan,
                'gpMargin':float(df['gpMargin'].iloc[0]) if df['gpMargin'].iloc[0]!='' else np.nan,
                'npMargin':float(df['npMargin'].iloc[0]) if df['npMargin'].iloc[0]!='' else np.nan,
                'netProfit':float(df['netProfit'].iloc[0]) if df['netProfit'].iloc[0]!='' else np.nan})
    except: pass; time.sleep(0.02)
    # 资产负债表 Q1
    try:
        rs=bs.query_balance_data(code=fc,year=2026,quarter=1);df=rs.get_data()
        if df is not None and not df.empty:
            sd=pd.to_datetime(df['statDate'].iloc[0]).date()
            for r in rows:
                if r['symbol']==cs and r['statDate']==sd: r['debt_ratio']=float(df['liabilityToAsset'].iloc[0]) if df['liabilityToAsset'].iloc[0]!='' else np.nan
    except: pass; time.sleep(0.02)
    # 现金流量表 Q1
    try:
        rs=bs.query_cash_flow_data(code=fc,year=2026,quarter=1);df=rs.get_data()
        if df is not None and not df.empty:
            sd=pd.to_datetime(df['statDate'].iloc[0]).date()
            for r in rows:
                if r['symbol']==cs and r['statDate']==sd: r['CFOtoNP']=float(df['CFOToNP'].iloc[0]) if df['CFOToNP'].iloc[0]!='' else np.nan
    except: pass; time.sleep(0.02)

    # 每100只保存 (断点续传)
    if (i+1)%100==0:
        pd.DataFrame(rows).to_parquet(cache_file)
        logger.info(f'[{i+1}/{len(todo)}] {len(rows)}只 {(time.time()-t0)/60:.0f}min')

# 最终保存
pd.DataFrame(rows).to_parquet(cache_file)
bs.logout()
logger.info(f'完成:{len(rows)}只/{(time.time()-t0)/60:.0f}min')
logger.info(f'中断后重跑: python scripts/download_fin_q1.py')
