#!/usr/bin/env python3
"""日频推理 — 今日选股 Top30"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np, pandas as pd, json
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars
from src.data.tencent_fetcher import fetch_valuation_batch

if not hasattr(pd.DataFrame,'append'):
    pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))

# 数据
end=date.today(); start=end-timedelta(days=400)
raw=read_daily_bars(A_STOCK_DIR,start_date=start,end_date=end,market='a_stock')
syms=sorted(raw['symbol'].unique())[:300]
raw=raw[raw['symbol'].isin(syms)].copy(); raw['trade_date']=pd.to_datetime(raw['trade_date'])

val=fetch_valuation_batch(symbols=syms[:200])
val['trade_date']=pd.to_datetime(val['trade_date']) if 'trade_date' in val.columns else None

fin=pd.read_parquet('.cache_fin_200.parquet'); fin['pubDate']=pd.to_datetime(fin['pubDate'])
holder=pd.read_parquet('.cache_holder_200.parquet'); holder['end_date']=pd.to_datetime(holder['end_date'])
holder['holder_signal']=-holder.groupby('symbol')['holder_num'].transform(lambda x:x.pct_change())

# 行业
ind_map=json.load(open('.industry_cache.json'))
ind_names=json.load(open('.industry_names.json')).get('code_to_name',{})
tiers=json.load(open('.industry_names.json')).get('strategic_tier',{})

# === 快速因子计算 (仅最新日) ===
latest_date=raw['trade_date'].max()
print(f"日期: {latest_date.date()}\n")

results=[]
for sym in syms[:200]:
    grp=raw[raw['symbol']==sym].sort_values('trade_date')
    c=grp['close'].values; n=len(c)
    if n<60: continue

    # Value
    sv=val[val['symbol']==sym] if val is not None else pd.DataFrame()
    pe=float(sv['pe_ttm'].iloc[0]) if not sv.empty and not pd.isna(sv['pe_ttm'].iloc[0]) else np.nan
    pb=float(sv['pb'].iloc[0]) if not sv.empty and not pd.isna(sv['pb'].iloc[0]) else np.nan
    v_ep=1.0/pe if pe and pe>0 else np.nan
    v_bp=1.0/pb if pb and pb>0 else np.nan
    v_cfp=np.nan

    # Momentum
    mom=np.nan
    if n>=252 and c[-21]>0: mom=c[-21]/c[-252]-1

    # Reversal
    rev=np.nan
    if n>=21: rev=-(c[-1]/c[-22]-1)

    # Quality
    q_roe=q_leverage=q_fscore=np.nan
    sf=fin[fin['symbol']==sym].sort_values('pubDate')
    if not sf.empty:
        prev=sf[sf['pubDate']<=latest_date]
        if not prev.empty:
            r=prev.iloc[-1]
            q_roe=r.get('roe',np.nan)
            q_leverage=-(r.get('debt_ratio',np.nan)) if not pd.isna(r.get('debt_ratio')) else np.nan
            # F-Score simplified
            f=0
            if not pd.isna(r.get('roe')) and r['roe']>0: f+=1
            if not pd.isna(r.get('CFOtoNP')) and r['CFOtoNP']>0: f+=1
            if not pd.isna(r.get('CFOtoNP')) and r['CFOtoNP']>1: f+=1
            q_fscore=f/3.0 if f>0 else np.nan

    # Alternative
    a_visit=0
    av=pd.read_parquet('.cache_analyst_visit.parquet')
    if not av.empty:
        sv_av=av[av['symbol']==sym]
        if not sv_av.empty: a_visit=len(sv_av)

    # Strategic
    ind_code=ind_map.get(str(sym),-1)
    tier_score=float(tiers.get(str(ind_code),0) if isinstance(tiers.get(str(ind_code),0),(int,float)) else (1.0 if tiers.get(str(ind_code))==1 else 0.5 if tiers.get(str(ind_code))==2 else 0))

    # Composite (简化权重)
    score=0; n_factors=0
    for val_v,w in [(v_ep,0.1),(v_bp,0.1),(mom,0.15),(rev,0.15),(q_roe,0.2),(q_leverage,0.1),(q_fscore,0.1),(a_visit,0.05),(tier_score,0.05)]:
        if not pd.isna(val_v): score+=val_v*w; n_factors+=1

    results.append({
        'symbol':sym,'composite':score,
        'v_ep':v_ep,'v_bp':v_bp,'momentum':mom,'reversal':rev,
        'q_roe':q_roe,'q_leverage':q_leverage,'q_fscore':q_fscore,
        'a_visit':a_visit,'strategic':tier_score,
        'industry':ind_names.get(str(ind_code),f'行业{ind_code}')
    })

df=pd.DataFrame(results).dropna(subset=['composite'])
df=df.sort_values('composite',ascending=False).head(30)

# 获取股票名称 (从腾讯财经)
name_map = {}
if val is not None and 'name' in val.columns:
    name_map = dict(zip(val['symbol'], val['name']))

print(f"{'排名':<5} {'代码':<8} {'名称':<12} {'行业':<12} {'总分':>8}")
print('-'*55)
for i,(_,r) in enumerate(df.iterrows()):
    nm = name_map.get(r['symbol'], '')
    print(f"{i+1:<5} {r['symbol']:<8} {nm:<12} {r['industry']:<12} {r['composite']:>+8.3f}")

print(f"\n行业分布:")
for ind,cnt in df['industry'].value_counts().head(10).items():
    print(f"  {ind}: {cnt}只")
