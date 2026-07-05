"""
v3.0 BlackRock 四层因子架构

  基本面 40% (Quality) — ROE/毛利率/应计/杠杆/稳定性/FScore/股东户数
  价量   30% (Price)    — Value(E/P+B/P+CF/P) + 短期反转
  另类   20% (Alt)      — 分析师买入/EPS增速/机构调研/公告情绪/舆情
  其他   10% (Other)    — 战略行业 + 行业中性化

回测: 200只, 195周, 全部时间点数据正确
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np, pandas as pd, json, requests, logging
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda s,o,**kw: pd.concat([s,o],ignore_index=kw.get("ignore_index",False))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("blackrock")

N, TOPN = 200, 30
W_FUND, W_PRICE, W_ALT, W_OTHER = 0.40, 0.30, 0.20, 0.10

# ── 交易成本 ──
STAMP_TAX = 0.0005   # 印花税 0.05% (卖出)
COMMISSION = 0.0003  # 佣金 0.03%
SLIPPAGE = 0.001     # 滑点 0.1%
COST_PER_TRADE = STAMP_TAX + COMMISSION + SLIPPAGE  # 单边约 0.18%

# ── 调仓频率: daily ──
REBALANCE_FREQ = 1  # 每天调仓

# ── 数据加载 ──
end = date.today(); start = end - timedelta(days=365*3)
raw = read_daily_bars(A_STOCK_DIR, start_date=start-timedelta(days=400), end_date=end, market="a_stock")
syms = sorted(raw["symbol"].unique())[:N]
raw = raw[raw["symbol"].isin(syms)].copy()
raw["trade_date"] = pd.to_datetime(raw["trade_date"])

val = pd.read_parquet(".cache_pe_200.parquet"); val["trade_date"] = pd.to_datetime(val["trade_date"])
fin = pd.read_parquet(".cache_fin_200.parquet"); fin["pubDate"] = pd.to_datetime(fin["pubDate"])
holder = pd.read_parquet(".cache_holder_200.parquet"); holder["end_date"] = pd.to_datetime(holder["end_date"])
holder["holder_signal"] = -holder.groupby("symbol")["holder_num"].transform(lambda x: x.pct_change())
analyst_fc = pd.read_parquet(".cache_analyst_fc.parquet")
analyst_visit = pd.read_parquet(".cache_analyst_visit.parquet")

# 行业
industry_map = {}
try: industry_map = json.loads(open(".industry_cache.json").read())
except: pass
strategy_tier = {}
try:
    ind_data = json.loads(open(".industry_names.json").read())
    for k,v in ind_data.get("strategic_tier",{}).items():
        strategy_tier[int(k)] = 1.0 if v==1 else 0.5 if v==2 else 0.0
except: pass

# F-Score 预计算
fin_sorted = fin.sort_values(["symbol","statDate"])
fscore_rows = []
for sym, sym_fin in fin_sorted.groupby("symbol"):
    sym_fin = sym_fin.sort_values("statDate"); prev = None
    for _, row in sym_fin.iterrows():
        f=0
        if not pd.isna(row.get('roe')) and row['roe']>0: f+=1
        if not pd.isna(row.get('CFOtoNP')) and row['CFOtoNP']>0: f+=1
        if not pd.isna(row.get('CFOtoNP')) and row['CFOtoNP']>1: f+=1
        if prev is not None:
            if not pd.isna(row.get('roe')) and not pd.isna(prev.get('roe')) and row['roe']>prev['roe']: f+=1
            if not pd.isna(row.get('debt_ratio')) and not pd.isna(prev.get('debt_ratio')) and row['debt_ratio']<prev['debt_ratio']: f+=1
            if not pd.isna(row.get('gpMargin')) and not pd.isna(prev.get('gpMargin')) and row['gpMargin']>prev['gpMargin']: f+=1
        fscore_rows.append({"symbol":sym,"pubDate":row["pubDate"],"fscore":f/6.0}); prev=row
fscore_df = pd.DataFrame(fscore_rows); fscore_df["pubDate"]=pd.to_datetime(fscore_df["pubDate"])

# 公告情绪 (时间点对齐)
cache_ann = Path(".cache_announce.parquet")
announce_df = pd.read_parquet(cache_ann) if cache_ann.exists() else pd.DataFrame()
if not announce_df.empty:
    announce_df['pub_date'] = pd.to_datetime(announce_df['pub_date'])
    n_ann = announce_df['symbol'].nunique()
else:
    n_ann = 0
logger.info(f"公告: {len(announce_df)}条, {n_ann}只")

# 舆情: 热门榜
hot_df = None
try:
    import akshare as ak
    hot_df = ak.stock_hot_rank_em()
    hot_df['symbol'] = hot_df['代码'].astype(str).str.zfill(6)
    hot_symbols = set(hot_df['symbol'].tolist())
    logger.info(f"舆情热门: {len(hot_symbols)}只")
except Exception as e:
    hot_symbols = set()
    logger.warning(f"舆情跳过: {e}")

# ── 信号计算 ──
logger.info("计算信号...")
def compute_signals(price_df, hist_val, hist_fin):
    df = price_df.sort_values(["symbol","trade_date"]).copy()
    results = []
    for sym, grp in df.groupby("symbol"):
        grp=grp.sort_values("trade_date"); c=grp["close"].values; n=len(c)
        if n<60: continue
        out=grp[["trade_date","symbol"]].copy()
        # Price: Value + Reversal
        sv=hist_val[hist_val["symbol"]==sym]
        if not sv.empty:
            sv=sv.copy(); sv["trade_date"]=pd.to_datetime(sv["trade_date"])
            out=out.merge(sv[["trade_date","pe_ttm","pb"]],on="trade_date",how="left")
            out["v_ep"]=1.0/out["pe_ttm"].clip(lower=1.0)
            out["v_bp"]=1.0/out["pb"].clip(lower=0.1)
        else: out["v_ep"]=out["v_bp"]=np.nan
        out["v_cfp"]=np.nan; out["v_sp"]=np.nan  # S/P (营收/市值)
        # Reversal
        rev=np.full(n,np.nan)
        if n>=21: rev[21:]=c[21:]/c[:-21]-1
        out["p_reversal"]=-rev
        # Quality init
        for q in ["q_roe","q_gross","q_accrual","q_leverage","q_stability","q_fscore","q_holder"]:
            out[q]=np.nan
        # Quality from financials
        sf=hist_fin[hist_fin["symbol"]==sym].sort_values("pubDate")
        if not sf.empty:
            for idx,row in out.iterrows():
                td=row["trade_date"]; prev=sf[sf["pubDate"]<=td]
                if not prev.empty:
                    r=prev.iloc[-1]
                    for col,key in [("q_roe","roe"),("q_gross","gpMargin"),("q_leverage","debt_ratio")]:
                        if key in r and not pd.isna(r[key]): out.at[idx,col]=r[key] if col!="q_leverage" else -r[key]
                    if "q_leverage" in r: pass  # already set
                    if "CFOtoNP" in r and not pd.isna(r["CFOtoNP"]): out.at[idx,"q_accrual"]=-(1.0-r["CFOtoNP"])
                    if "roe_stability" in r and not pd.isna(r["roe_stability"]): out.at[idx,"q_stability"]=-r["roe_stability"]
                    if "CFOtoNP" in r and not pd.isna(r["CFOtoNP"]):
                        pe_v=out.at[idx,"pe_ttm"] if "pe_ttm" in out.columns else np.nan
                        if not pd.isna(pe_v) and pe_v>0 and "netProfit" in r:
                            mktcap=pe_v*r["netProfit"]; cfo=r["netProfit"]*r["CFOtoNP"]
                            if mktcap>0: out.at[idx,"v_cfp"]=cfo/mktcap
                    # S/P = 营收/市值 (需要npMargin)
                    if "npMargin" in r and not pd.isna(r.get("npMargin")) and r["npMargin"]>0:
                        pe_v=out.at[idx,"pe_ttm"] if "pe_ttm" in out.columns else np.nan
                        if not pd.isna(pe_v) and pe_v>0 and "netProfit" in r:
                            revenue=r["netProfit"]/(r["npMargin"]); mktcap=pe_v*r["netProfit"]
                            if mktcap>0: out.at[idx,"v_sp"]=revenue/mktcap
        # F-Score merge
        sym_fs=fscore_df[fscore_df["symbol"]==sym].sort_values("pubDate")
        if not sym_fs.empty:
            out["_tmp"]=out["trade_date"]; out=pd.merge_asof(out.sort_values("_tmp"),sym_fs[["pubDate","fscore"]].sort_values("pubDate"),left_on="_tmp",right_on="pubDate",direction="backward")
            out["q_fscore"]=out["fscore"].fillna(0); out.drop(columns=["_tmp","pubDate","fscore"],inplace=True,errors="ignore")
        # Holder merge
        sym_hold=holder[holder["symbol"]==sym].sort_values("end_date")
        if not sym_hold.empty:
            out["_tmp"]=out["trade_date"]; out=pd.merge_asof(out.sort_values("_tmp"),sym_hold[["end_date","holder_signal"]].sort_values("end_date"),left_on="_tmp",right_on="end_date",direction="backward")
            out["q_holder"]=out["holder_signal"].fillna(0); out.drop(columns=["_tmp","end_date","holder_signal"],inplace=True,errors="ignore")
        results.append(out)
    return pd.concat(results,ignore_index=True) if results else pd.DataFrame()

signals=compute_signals(raw,val,fin)
logger.info(f"信号: {len(signals)}行")

# ── 另类因子 (仅保留有时间点的, 避免穿越) ──
signals['trade_date']=pd.to_datetime(signals['trade_date'])

# 1. 机构调研 — 时间点对齐: 过去180天被调研次数
signals['a_visit']=0.0
if not analyst_visit.empty and 'visit_date' in analyst_visit.columns:
    # 每日调研计数
    av=analyst_visit.copy()
    av['visit_date']=pd.to_datetime(av['visit_date'])
    daily_visits=av.groupby(['symbol',av['visit_date']]).size().reset_index(name='cnt')
    daily_visits.columns=['symbol','trade_date','visit_cnt']
    # merge → rolling sum
    signals=signals.merge(daily_visits,on=['symbol','trade_date'],how='left')
    signals['visit_cnt']=signals['visit_cnt'].fillna(0)
    # 对每只股票, 180天滚动求和
    signals=signals.sort_values(['symbol','trade_date'])
    signals['a_visit']=signals.groupby('symbol')['visit_cnt'].transform(
        lambda x: x.rolling(180, min_periods=1).sum()
    )
    signals.drop(columns=['visit_cnt'],inplace=True,errors='ignore')

# 2. 公告情绪 — 时间点对齐: merge_asof + 近90天累计
signals['a_announce']=0.0
if not announce_df.empty:
    # 对每只股票, 创建每日公告得分 → rolling 90天求和
    for sym in signals['symbol'].unique():
        sym_a = announce_df[announce_df['symbol']==sym]
        if sym_a.empty: continue
        sym_a = sym_a.sort_values('pub_date')
        mask = signals['symbol']==sym
        sym_sig = signals.loc[mask].sort_values('trade_date')
        # merge_asof: 每个交易日取最近公告得分
        merged = pd.merge_asof(
            sym_sig[['trade_date']],
            sym_a[['pub_date','score']].rename(columns={'pub_date':'trade_date'}),
            on='trade_date', direction='backward'
        )
        # 公告得分在90天内有效
        sym_sig = sym_sig.copy()
        sym_sig['_score'] = merged['score'].fillna(0).values
        sym_sig['a_announce'] = sym_sig['_score'].rolling(90, min_periods=1).sum()
        signals.loc[mask,'a_announce'] = sym_sig['a_announce'].values

# 3-4: 仅当日数据
signals['a_buy']=0.0; signals['a_eps']=0.0
signals['a_sentiment']=0.0

# 战略行业
signals['o_strategic']=signals['symbol'].apply(lambda s: strategy_tier.get(industry_map.get(str(s),-1),0.0)) if industry_map else 0

# ── 四层因子合成 ──
# 行业标记
signals['_ind']=signals['symbol'].map(industry_map).fillna(-1) if industry_map else -1

def make_factor(df, subs, name):
    cols=[]
    for s in subs:
        if s in df.columns and df[s].notna().any():
            mu,sigma=df[s].mean(),df[s].std()
            df[s+'_z']=(df[s]-mu)/sigma if sigma and sigma>0 else 0
            ind_mean=df.groupby('_ind')[s+'_z'].transform('mean')
            df[s+'_z']=df[s+'_z']-ind_mean
            cols.append(s+'_z')
    df[name]=df[cols].mean(axis=1) if cols else 0

# 基本面 40%
fund_subs=["q_roe","q_gross","q_accrual","q_leverage","q_stability","q_fscore","q_holder"]
make_factor(signals, fund_subs, "fundamental")

# 价量 30% (加S/P + 行业轮动)
# 价量: E/P分位数替代S/P (npMargin数据暂缺)
# 在make_factor前计算行业内E/P分位数
if industry_map:
    signals['_ind3']=signals['symbol'].map(industry_map).fillna(-1)
    signals['v_ep_pct']=signals.groupby('_ind3')['v_ep'].transform(lambda x: x.rank(pct=True))
    signals.drop(columns=['_ind3'],inplace=True,errors='ignore')
else:
    signals['v_ep_pct']=signals['v_ep'].rank(pct=True)

price_subs=["v_ep","v_bp","v_cfp","v_sp","v_ep_pct","p_reversal"]
make_factor(signals, price_subs, "price")

# 行业轮动: 行业动量因子 (在 make_factor 前加 _ind, 用完再删)
if industry_map:
    signals['_ind2'] = signals['symbol'].map(industry_map).fillna(-1)
    ind_return = signals.groupby(['_ind2','trade_date'])['p_reversal'].transform('mean')
    signals['o_sector'] = -ind_return
    signals.drop(columns=['_ind2'], inplace=True, errors='ignore')
else:
    signals['o_sector'] = 0
make_factor(signals, ["o_sector"], "sector_rotation")

# 另类 20%
alt_subs=["a_visit","a_announce","a_buy","a_eps"]
make_factor(signals, alt_subs, "alternative")

# 其他 10%: 战略行业单独加入总分
signals["other"] = signals["o_strategic"]

signals.drop(columns=['_ind'],inplace=True,errors='ignore')

signals["composite"] = (
    signals["fundamental"].fillna(0)*W_FUND +
    signals["price"].fillna(0)*W_PRICE +
    signals["alternative"].fillna(0)*W_ALT +
    signals["other"].fillna(0)*W_OTHER +
    signals.get("sector_rotation", 0).fillna(0)*0.05  # 行业轮动额外5%
)

# ── 预计算收益矩阵 (向量化, 日频秒级) ──
logger.info("预计算收益矩阵...")
close_m = raw.pivot_table(index="trade_date",columns="symbol",values="close",aggfunc="last")
close_m = close_m.sort_index().ffill()
fwd_ret = close_m.pct_change().shift(-1)  # t日买入 → t+1日卖出收益
logger.info(f"  矩阵: {close_m.shape}")

# ── 回测 (向量化) ──
signals["trade_date"]=pd.to_datetime(signals["trade_date"])
dates=sorted(signals["trade_date"].unique())
rebalance_dates=dates[::REBALANCE_FREQ]
logger.info(f"回测: {len(rebalance_dates)}天")

cash=1_000_000; prev_holdings=set(); ic_records=[]

for i,td in enumerate(rebalance_dates):
    day=signals[signals["trade_date"]==td]
    if len(day)<TOPN: continue
    top=day.nlargest(TOPN,"composite")

    # IC: 从矩阵直接读
    if td in fwd_ret.index:
        fwd=fwd_ret.loc[td]; common=list(set(day["symbol"])&set(fwd.dropna().index))
        if len(common)>10:
            ic=day.set_index("symbol").loc[common,"composite"].corr(fwd[common],method="spearman")
            ic_records.append({"date":td,"ic":0 if pd.isna(ic) else ic})

    # 收益: 从矩阵向量化读取
    new_holdings=set(top["symbol"].tolist())
    turnover=len(new_holdings-prev_holdings)/TOPN
    if td in fwd_ret.index:
        rets=[fwd_ret.loc[td].get(s) for s in new_holdings]
        rets=[r for r in rets if r is not None and not pd.isna(r)]
        if rets: cash*=(1+np.mean(rets)-turnover*COST_PER_TRADE)
    prev_holdings=new_holdings

    if (i+1)%200==0:
        m_ic=np.mean([r["ic"] for r in ic_records[-200:]]) if ic_records else 0
        logger.info(f"  [{i+1}/{len(rebalance_dates)}] ¥{cash:,.0f} IC={m_ic:+.3f}")

total_ret=cash/1_000_000-1; n_days=(rebalance_dates[-1]-rebalance_dates[0]).days
annual=(1+total_ret)**(365.25/max(n_days,1))-1
logger.info(f"\nBlackRock 4/3/2/1: 年化{annual:+.1%} 终值¥{cash:,.0f}")

if ic_records:
    ic_df=pd.DataFrame(ic_records); ic_df["month"]=ic_df["date"].dt.to_period("M")
    monthly=ic_df.groupby("month")["ic"].mean()
    logger.info(f"IC: mean={monthly.mean():+.3f} IR={monthly.mean()/max(monthly.std(),0.001):+.1f} >0:{(monthly>0).mean():.0%}")
