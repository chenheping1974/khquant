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

# PE: 全量拉取 (腾讯单次50只, 5000只≈100秒)
N_PE = len(syms)
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

# ── 数据新鲜度 ──
import os, datetime as dt
fresh = []
for path,name,max_d in [('.cache_fin_infer.parquet','财报',120),('.cache_holder_all.parquet','股东户数',90),
    ('.cache_announce.parquet','公告',60),('.cache_analyst_fc.parquet','分析师',30)]:
    if os.path.exists(path):
        m=dt.datetime.fromtimestamp(os.path.getmtime(path)); age=(dt.datetime.now()-m).days
        fresh.append(f'  {"✅" if age<=max_d else "⚠️过期"} {name}: {age}天前 (>{max_d}天警告)')
    else: fresh.append(f'  ❌ {name}: 缺失')
print(f"日期: {latest_date.date()}\n数据新鲜度:")
for f in fresh: print(f)
fout.write(f"khquant v3.0 选股信号 — {latest_date.date()}\n数据新鲜度:\n")
for f in fresh: fout.write(f+"\n")
fout.write(f"{'='*55}\n\n")


# ── 分析师数据 (akshare实时拉) ──
analyst_map = {}
try:
    import akshare as ak
    fc = ak.stock_profit_forecast_em()
    if fc is not None and not fc.empty:
        fc['symbol'] = fc['代码'].astype(str).str.zfill(6)
        buy_col = '机构投资评级(近六个月)-买入'
        neut_col = '机构投资评级(近六个月)-增持'
        fc['a_buy'] = fc[buy_col].fillna(0) / (fc[buy_col].fillna(0)+fc[neut_col].fillna(0)+1)
        ep25 = pd.to_numeric(fc.get('2026预测每股收益',fc.get('2025预测每股收益',pd.Series([np.nan]))),errors='coerce')
        ep26 = pd.to_numeric(fc.get('2027预测每股收益',ep25),errors='coerce')
        fc['a_eps'] = (ep26/ep25.replace(0,np.nan)-1).clip(-0.5,1.0)
        analyst_map = dict(zip(fc['symbol'], zip(fc['a_buy'], fc['a_eps'])))
        print(f"分析师(实时): {len(analyst_map)}只")
except Exception as e:
    print(f"分析师实时失败: {e}")

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
    mktcap=float(sv['market_cap'].iloc[0]) if not sv.empty and not pd.isna(sv.get('market_cap',pd.Series([np.nan])).iloc[0]) else np.nan
    v_size=-np.log(mktcap) if mktcap and mktcap>0 else np.nan  # 小盘溢价

    # Momentum
    mom=np.nan
    if n>=252 and c[-21]>0: mom=c[-21]/c[-252]-1

    # Reversal
    rev=np.nan
    if n>=21: rev=-(c[-1]/c[-22]-1)

    # Quality (推理用最新财报,不比对日期)
    q_roe=q_leverage=q_fscore=np.nan
    sf=fin[fin['symbol']==sym]
    if not sf.empty:
        r=sf.iloc[-1]  # 取最新一行
        q_roe=r.get('roe',np.nan)
        q_leverage=-(r.get('debt_ratio',np.nan)) if not pd.isna(r.get('debt_ratio')) else np.nan
        # F-Score (6指标, 缺CFO时用gpMargin+debt替代)
        f=0
        if not pd.isna(r.get('roe')) and r['roe']>0: f+=1
        if not pd.isna(r.get('CFOtoNP')) and r['CFOtoNP']>0: f+=1
        if not pd.isna(r.get('CFOtoNP')) and r['CFOtoNP']>1: f+=1
        if not pd.isna(r.get('gpMargin')) and r['gpMargin']>0.1: f+=1
        if not pd.isna(r.get('debt_ratio')) and r['debt_ratio']<0.6: f+=1
        if not pd.isna(r.get('npMargin')) and r['npMargin']>0.05: f+=1
        q_fscore=f/6.0

    # Alternative
    a_visit=0
    av=safe_read('.cache_analyst_visit.parquet')
    if not av.empty and 'symbol' in av.columns:
        sv_av=av[av['symbol']==sym]
        if not sv_av.empty: a_visit=len(sv_av)

    # Strategic
    ind_code=ind_map.get(str(sym),-1)
    tier_score=float(tiers.get(str(ind_code),0) if isinstance(tiers.get(str(ind_code),0),(int,float)) else (1.0 if tiers.get(str(ind_code))==1 else 0.5 if tiers.get(str(ind_code))==2 else 0))

    # 公告情绪: 近90天公告得分
    a_announce = 0
    ann = safe_read('.cache_announce.parquet')
    if not ann.empty and 'symbol' in ann.columns:
        sym_ann = ann[ann['symbol']==sym]
        if not sym_ann.empty and 'pub_date' in sym_ann.columns:
            sym_ann['pub_date'] = pd.to_datetime(sym_ann['pub_date'])
            recent = sym_ann[sym_ann['pub_date'] >= pd.Timestamp.now()-pd.Timedelta(days=90)]
            a_announce = recent['score'].sum() if not recent.empty else 0

    # Analyst
    a_buy, a_eps = analyst_map.get(sym, (np.nan, np.nan))

    results.append({
        'symbol':sym,'v_ep':v_ep,'v_bp':v_bp,'v_size':v_size,'mktcap_raw':mktcap,'momentum':mom,'reversal':rev,
        'q_roe':q_roe,'q_leverage':q_leverage,'q_fscore':q_fscore,
        'a_visit':a_visit,'a_buy':a_buy,'a_eps':a_eps,'a_announce':a_announce,'strategic':tier_score,
        'industry':ind_names.get(str(ind_code),f'行业{ind_code}')
    })

df=pd.DataFrame(results)
# LSY 2019: 市值最小30%的Size因子设为0 (壳污染, 不参与小盘排序)
# 但股票保留在池中, 其他因子仍可得分
if 'mktcap_raw' in df.columns and 'v_size' in df.columns:
    cap_cutoff = df['mktcap_raw'].quantile(0.30)
    n_small = (df['mktcap_raw'] < cap_cutoff).sum()
    df.loc[df['mktcap_raw'] < cap_cutoff, 'v_size'] = 0
    print(f"  Size过滤: {n_small}/{len(df)} 只设为0 (最小30%市值)")

# Z-score标准化各因子
factor_cols = ['v_ep','v_bp','v_size','momentum','reversal','q_roe','q_leverage','q_fscore','a_buy','a_eps','a_announce']
for col in factor_cols:
    if col in df.columns and df[col].notna().any():
        mu,sigma = df[col].mean(), df[col].std()
        if sigma and sigma>0: df[col+'_z'] = (df[col]-mu)/sigma
        else: df[col+'_z'] = 0
    else: df[col+'_z'] = 0

# 加权总分 (BlackRock: 基本面40% + 价量30% + 另类15% + 其他15%)
df['composite'] = (
    df['q_roe_z'].fillna(0)*0.20 + df['q_leverage_z'].fillna(0)*0.10 + df['q_fscore_z'].fillna(0)*0.10 +
    df['v_ep_z'].fillna(0)*0.08 + df['v_bp_z'].fillna(0)*0.05 + df['v_size_z'].fillna(0)*0.04 +
    df['momentum_z'].fillna(0)*0.05 + df['reversal_z'].fillna(0)*0.08 +
    df['a_buy_z'].fillna(0)*0.05 + df['a_eps_z'].fillna(0)*0.05 +
    df['a_announce_z'].fillna(0)*0.05 + df['a_visit'].apply(lambda x: min(x,30)/30*0.05) +
    df['strategic'].fillna(0)*0.05
)
# ── 因子覆盖报告 ──
print(f"\n{'='*55}")
print(f"因子覆盖报告")
print(f"{'='*55}")
print(f"股票池: {len(df)} 只 (剔除最小30%市值后)")
print(f"PE覆盖: {df['v_ep'].notna().sum()} 只")
print(f"PB覆盖: {df['v_bp'].notna().sum()} 只")
print(f"动量覆盖: {df['momentum'].notna().sum()} 只")
print(f"反转覆盖: {df['reversal'].notna().sum()} 只")
print(f"ROE覆盖: {df['q_roe'].notna().sum()} 只")
print(f"杠杆覆盖: {df['q_leverage'].notna().sum()} 只")
print(f"F-Score覆盖: {df['q_fscore'].notna().sum()} 只")
print(f"分析师买入覆盖: {df['a_buy'].notna().sum()} 只")
print(f"EPS增速覆盖: {df['a_eps'].notna().sum()} 只")
print(f"机构调研覆盖: {(df['a_visit']>0).sum()} 只")
print(f"战略行业覆盖: {(df['strategic']>0).sum()} 只")
print(f"\n因子权重:")
print(f"  基本面 40%: ROE(20%) + 杠杆(10%) + F-Score(10%)")
print(f"  价量   30%: E/P(8%) + B/P(5%) + Size(4%) + 反转(8%) + 动量(5%)")
print(f"  另类   20%: 分析师买入(5%) + EPS增速(5%) + 公告(5%) + 机构调研(5%)")
print(f"  其他   10%: 战略行业(5%) + 行业中性化(5%)")
print(f"{'='*55}\n")

# 写入文件
fout.write(f"\n{'='*55}\n因子覆盖报告\n{'='*55}\n")
fout.write(f"股票池: {len(df)} 只\n")
for name, col in [('PE','v_ep'),('PB','v_bp'),('动量','momentum'),('反转','reversal'),
    ('ROE','q_roe'),('杠杆','q_leverage'),('F-Score','q_fscore'),
    ('分析师买入','a_buy'),('EPS增速','a_eps'),('机构调研','a_visit'),('战略行业','strategic')]:
    n = df[col].notna().sum() if col != 'a_visit' else (df['a_visit']>0).sum()
    fout.write(f"  {name}: {n}/{len(df)}\n")
fout.write(f"{'='*55}\n\n")

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
    "method": "BlackRock 4/3/2/1",
    "data_freshness": {f.split(':')[0].strip().lstrip('✅⚠️❌ '): f.strip() for f in fresh},
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
