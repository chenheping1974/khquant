#!/usr/bin/env python3
"""下载全量公告数据 — 断点续传, 中断自动跳过已下载"""
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd, requests
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('ann')

end=date.today();start=end-timedelta(days=365*3)
syms=sorted(read_daily_bars(A_STOCK_DIR,start_date=start,end_date=end,market='a_stock')['symbol'].unique())

cf='.cache_announce.parquet'
try:
    old=pd.read_parquet(cf); done=set(old['symbol'].unique()); rows=old.to_dict('records')
except: done=set(); rows=[]

todo=[s for s in syms if s not in done]
logger.info(f'公告: 全量{len(syms)} 已有{len(done)} 待拉{len(todo)}')
if not todo:
    logger.info('✅ 全部完成!')
    sys.exit(0)

pos_kw=['业绩预增','中标','回购','增持','分红','送转','预盈','扭亏','重大合同','突破','获批','注册']
neg_kw=['减持','亏损','退市','立案','警示','问询','处罚','诉讼','冻结','终止','修正','下调']

t0=time.time()
for i,code in enumerate(todo):
    try:
        r=requests.get('https://np-anotice-stock.eastmoney.com/api/security/ann',
            params={'page_size':30,'page_index':1,'ann_type':'A','client_source':'web','stock_list':code},
            headers={'User-Agent':'Mozilla/5.0'},timeout=10)
        data=r.json()
        for item in data.get('data',{}).get('list',[]):
            sc=0; t=item.get('title','')
            if any(k in t for k in pos_kw): sc=1
            elif any(k in t for k in neg_kw): sc=-1
            if sc!=0: rows.append({'symbol':str(code),'pub_date':str(item['notice_date'][:10]),'score':int(sc)})
    except: pass
    if (i+1)%200==0:
        pd.DataFrame(rows).to_parquet(cf)
        logger.info(f'[{i+1}/{len(todo)}] {len(rows)}条 {pd.DataFrame(rows)["symbol"].nunique()}只 {(time.time()-t0)/60:.0f}min')
    time.sleep(0.03)

pd.DataFrame(rows).to_parquet(cf)
n=pd.DataFrame(rows)['symbol'].nunique()
logger.info(f'完成:{len(rows)}条 {n}只 ({(time.time()-t0)/60:.0f}min)')
