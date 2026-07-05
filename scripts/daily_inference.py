#!/usr/bin/env python3
"""日频推理 — 今日选股 Top30"""
import sys, os
from pathlib import Path
# Actions 兼容: 确保项目根目录在 sys.path
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# PYTHONPATH 兜底
for p in os.environ.get("PYTHONPATH", "").split(":"):
    if p and p not in sys.path:
        sys.path.insert(0, p)

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
syms=sorted(raw['symbol'].unique())
raw=raw[raw['symbol'].isin(syms)].copy(); raw['trade_date']=pd.to_datetime(raw['trade_date'])

# PE: 尽可能多拉 (腾讯单次50只, 500只≈10秒)
N_PE = min(1000, len(syms))
val=fetch_valuation_batch(symbols=syms[:N_PE])
if val is not None and not val.empty and 'trade_date' in val.columns:
    val['trade_date']=pd.to_datetime(val['trade_date'])

# 安全加载缓存 (文件可能不存在于Actions)
def safe_read(path):
    try: return pd.read_parquet(path)
    except: return pd.DataFrame()
def safe_json(path):
    try: return json.load(open(path))
    except: return {}

fin=safe_read('.cache_fin_infer.parquet')  # Q1全量财报
if not fin.empty: fin['pubDate']=pd.to_datetime(fin['pubDate'])
holder=safe_read('.cache_holder_all.parquet')  # 全量股东户数
if not holder.empty:
    holder['end_date']=pd.to_datetime(holder['end_date'])
    holder['holder_signal']=-holder.groupby('symbol')['holder_num'].transform(lambda x:x.pct_change())

ind_map=safe_json('.industry_cache.json')
ind_names=safe_json('.industry_names.json').get('code_to_name',{})
tiers=safe_json('.industry_names.json').get('strategic_tier',{})

# === 快速因子计算 (仅最新日) ===
latest_date=raw['trade_date'].max()
# 输出到文件 + 控制台
import os; os.makedirs("results", exist_ok=True)
out_path = f"results/signals-{latest_date.date()}.txt"
fout = open(out_path, "w")

print(f"日期: {latest_date.date()}\n")
fout.write(f"khquant v3.0 选股信号 — {latest_date.date()}\n{'='*55}\n\n")

results=[]
for sym in syms[:N_PE]:  # 全量PE覆盖的股票
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
    av=safe_read('.cache_analyst_visit.parquet')
    if not av.empty and 'symbol' in av.columns:
        sv_av=av[av['symbol']==sym]
        if not sv_av.empty: a_visit=len(sv_av)

    # Strategic
    ind_code=ind_map.get(str(sym),-1)
    tier_score=float(tiers.get(str(ind_code),0) if isinstance(tiers.get(str(ind_code),0),(int,float)) else (1.0 if tiers.get(str(ind_code))==1 else 0.5 if tiers.get(str(ind_code))==2 else 0))

    results.append({
        'symbol':sym,'v_ep':v_ep,'v_bp':v_bp,'momentum':mom,'reversal':rev,
        'q_roe':q_roe,'q_leverage':q_leverage,'q_fscore':q_fscore,
        'a_visit':a_visit,'strategic':tier_score,
        'industry':ind_names.get(str(ind_code),f'行业{ind_code}')
    })

df=pd.DataFrame(results)
# Z-score标准化各因子
factor_cols = ['v_ep','v_bp','momentum','reversal','q_roe','q_leverage','q_fscore']
for col in factor_cols:
    if col in df.columns and df[col].notna().any():
        mu,sigma = df[col].mean(), df[col].std()
        if sigma and sigma>0: df[col+'_z'] = (df[col]-mu)/sigma
        else: df[col+'_z'] = 0
    else: df[col+'_z'] = 0

# 加权总分 (BlackRock: 基本面40% + 价量30% + 另类15% + 其他15%)
df['composite'] = (
    df['q_roe_z'].fillna(0)*0.20 + df['q_leverage_z'].fillna(0)*0.10 + df['q_fscore_z'].fillna(0)*0.10 +
    df['v_ep_z'].fillna(0)*0.10 + df['v_bp_z'].fillna(0)*0.10 +
    df['momentum_z'].fillna(0)*0.15 + df['reversal_z'].fillna(0)*0.15 +
    df['strategic'].fillna(0)*0.05 + df['a_visit'].apply(lambda x: min(x,50)/50*0.05)
)

df=df.dropna(subset=['composite']).sort_values('composite',ascending=False).head(30)

# 获取股票名称 (从腾讯财经)
name_map = {}
if val is not None and 'name' in val.columns:
    name_map = dict(zip(val['symbol'], val['name']))

header = f"{'排名':<5} {'代码':<8} {'名称':<12} {'行业':<12} {'总分':>8}"
print(header); fout.write(header + "\n")
print('-'*55); fout.write('-'*55 + "\n")
for i,(_,r) in enumerate(df.iterrows()):
    nm = name_map.get(r['symbol'], '')
    line = f"{i+1:<5} {r['symbol']:<8} {nm:<12} {r['industry']:<12} {r['composite']:>+8.3f}"
    print(line); fout.write(line + "\n")

ind_summary = f"\n行业分布:"
print(ind_summary); fout.write(ind_summary + "\n")
for ind,cnt in df['industry'].value_counts().head(10).items():
    line = f"  {ind}: {cnt}只"
    print(line); fout.write(line + "\n")

fout.close()
print(f"\n结果已保存: {out_path}")

# JSON 输出 (供其他项目HTTP读取)
import json
json_data = {
    "date": str(latest_date.date()),
    "updated": str(pd.Timestamp.now()),
    "method": "BlackRock 4/3/2/1, 200 stocks",
    "top30": []
}
for _, r in df.iterrows():
    json_data["top30"].append({
        "rank": int(len(json_data["top30"]) + 1),
        "symbol": r["symbol"],
        "name": name_map.get(r["symbol"], ""),
        "industry": r["industry"],
        "score": round(float(r["composite"]), 3),
        "factors": {
            "value_ep": round(float(r.get("v_ep", 0) or 0), 4),
            "value_bp": round(float(r.get("v_bp", 0) or 0), 4),
            "momentum_12m1m": round(float(r.get("momentum", 0) or 0), 4),
            "reversal_1m": round(float(r.get("reversal", 0) or 0), 4),
            "quality_roe": round(float(r.get("q_roe", 0) or 0), 4),
            "quality_leverage": round(float(r.get("q_leverage", 0) or 0), 4),
            "quality_fscore": round(float(r.get("q_fscore", 0) or 0), 2),
            "alt_visits": int(r.get("a_visit", 0) or 0),
            "strategic_tier": int(r.get("strategic", 0) or 0),
        }
    })

latest_json = "results/latest.json"
json.dump(json_data, open(latest_json, "w"), ensure_ascii=False, indent=2)
print(f"JSON已保存: {latest_json}")
