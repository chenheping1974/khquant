"""
v3.0 完整回测 — 3策略 × 全子指标

Value:     E/P + B/P + CF/P
Momentum:  价格动量 + 风险调整 + 盈利动量
Quality:   ROE + 毛利率 + 应计利润 + 杠杆 + 盈利稳定性

组合: Quality 40%, Value 30%, Momentum 30%
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import logging
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

if not hasattr(pd.DataFrame, "append"):
    pd.DataFrame.append = lambda s, o, **kw: pd.concat([s, o], ignore_index=kw.get("ignore_index", False))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v3full")

N, TOPN = 200, 30
W_V, W_Q, W_R = 0.35, 0.55, 0.10  # Value + Quality + 短期反转

# 行业映射
import json
industry_map = {}
try:
    industry_map = json.loads(open(".industry_cache.json").read())
except: pass

# 战略行业分级 (基于十四五规划 + 十五五前瞻)
# Tier1: 国家级战略核心 | Tier2: 政策支持 | Tier3: 普通
STRATEGIC_TIER1_KW = [
    '电池','半导体','芯片','集成','光伏','新能源','锂电','储能','氢能',
    '医药制造','生物制品','医疗器械','中药','化学制药','医疗服务','创新药',
    '航空航天','航天','航空','军工','船舶制造','大飞机',
    '新材料','稀土','纳米','碳纤维','超导',
    '计算机设备','软件开发','IT服务','互联网服务','人工智能','大数据',
    '机器人','工业母机','数控','自动化',
    '通信设备','通信服务','5G','6G','量子','卫星',
    '消费电子','光学光电','电子元件','电子信息','印制电路板',
]
STRATEGIC_TIER2_KW = [
    '汽车','电力','环保','风电','核电',
    '化学制品','化学原料','专用设备','仪器仪表','高端装备',
    '电网','输配电','特高压','充电桩','换电',
    '金属新材料','磁性材料','有机硅','氟化工',
    '医疗器械','体外诊断','疫苗','血液制品',
    '工程机械','重工','矿山机械','纺织机械',
    '显示面板','LED','激光','传感器',
    '智慧城市','车联网','物联网','虚拟现实','元宇宙',
]

# ── 数据加载 ──
end = date.today()
start = end - timedelta(days=365 * 3)
raw = read_daily_bars(A_STOCK_DIR, start_date=start - timedelta(days=400), end_date=end, market="a_stock")
syms = sorted(raw["symbol"].unique())[:N]
raw = raw[raw["symbol"].isin(syms)].copy()
raw["trade_date"] = pd.to_datetime(raw["trade_date"])

val = pd.read_parquet(".cache_pe_200.parquet")
val["trade_date"] = pd.to_datetime(val["trade_date"])
# baostock PE data has: trade_date, symbol, close, pe_ttm, pb

fin = pd.read_parquet(".cache_fin_200.parquet")
fin["pubDate"] = pd.to_datetime(fin["pubDate"])

# 股东户数
holder = pd.read_parquet(".cache_holder_200.parquet")
holder["end_date"] = pd.to_datetime(holder["end_date"])
holder = holder.sort_values(["symbol", "end_date"])
# 计算变化率: (当期-上期)/上期, 取反 (户数减少=利好)
holder["holder_change"] = holder.groupby("symbol")["holder_num"].transform(
    lambda x: x.pct_change()
)
# 取反: 户数减少(负变化) → 正信号
holder["holder_signal"] = -holder["holder_change"]

logger.info(f"数据: 价格{len(raw)} PE{len(val)} 财报{len(fin)}({fin['symbol'].nunique()}只)")

# ── F-Score 预计算 (每季度一次, 不逐日循环) ──
logger.info("预计算 F-Score...")
fin_sorted = fin.sort_values(["symbol", "statDate"])
fscore_rows = []
for sym, sym_fin in fin_sorted.groupby("symbol"):
    sym_fin = sym_fin.sort_values("statDate")
    prev = None
    for _, row in sym_fin.iterrows():
        f = 0
        roe_v = row.get("roe", np.nan)
        debt_v = row.get("debt_ratio", np.nan)
        cfo_np = row.get("CFOtoNP", np.nan)
        gp_v = row.get("gpMargin", np.nan)
        if not pd.isna(roe_v) and roe_v > 0: f += 1
        if not pd.isna(cfo_np) and cfo_np > 0: f += 1
        if not pd.isna(cfo_np) and cfo_np > 1: f += 1
        if prev is not None:
            prev_roe = prev.get("roe", np.nan)
            prev_debt = prev.get("debt_ratio", np.nan)
            prev_gp = prev.get("gpMargin", np.nan)
            if not pd.isna(roe_v) and not pd.isna(prev_roe) and roe_v > prev_roe: f += 1
            if not pd.isna(debt_v) and not pd.isna(prev_debt) and debt_v < prev_debt: f += 1
            if not pd.isna(gp_v) and not pd.isna(prev_gp) and gp_v > prev_gp: f += 1
        fscore_rows.append({"symbol": sym, "pubDate": row["pubDate"], "fscore": f / 6.0})
        prev = row
fscore_df = pd.DataFrame(fscore_rows)
fscore_df["pubDate"] = pd.to_datetime(fscore_df["pubDate"])

# 分析师数据
analyst_fc = pd.read_parquet(".cache_analyst_fc.parquet")
analyst_visit = pd.read_parquet(".cache_analyst_visit.parquet")

# ── 因子计算 ──
def compute_all_signals(price_df, hist_val, hist_fin):
    """计算所有子指标 → 3个独立策略信号"""
    df = price_df.sort_values(["symbol", "trade_date"]).copy()
    results = []

    for sym, grp in df.groupby("symbol"):
        grp = grp.sort_values("trade_date")
        c = grp["close"].values
        n = len(c)
        if n < 60:
            continue
        out = grp[["trade_date", "symbol"]].copy()

        # ── 日频数据 ──
        rets = np.diff(c) / np.maximum(np.abs(c[:-1]), 1e-8)
        vol_252 = np.nanstd(rets[-min(252, n):]) * np.sqrt(252) if len(rets) >= 20 else np.nan

        sv = hist_val[hist_val["symbol"] == sym]
        if not sv.empty:
            sv = sv.copy()
            sv["trade_date"] = pd.to_datetime(sv["trade_date"])
            out = out.merge(sv[["trade_date", "pe_ttm", "pb"]], on="trade_date", how="left")

            # ** Value 子指标 **
            out["v_ep"] = 1.0 / out["pe_ttm"].clip(lower=1.0)  # E/P
            out["v_bp"] = 1.0 / out["pb"].clip(lower=0.1)       # B/P
            out["v_cfp"] = np.nan  # CF/P (需要财报, 后面算)
        else:
            out["v_ep"] = out["v_bp"] = out["v_cfp"] = np.nan

        # ** Momentum 子指标 **
        mom = np.full(n, np.nan)
        mom_risk = np.full(n, np.nan)
        if n >= 252 and vol_252 and vol_252 > 0:
            for i in range(252, n):
                if c[i - 21] > 0:
                    mom[i] = c[i - 21] / c[i - 252] - 1
                    mom_risk[i] = mom[i] / vol_252
        out["m_price"] = mom
        out["m_risk_adj"] = mom_risk
        out["m_earn"] = np.nan  # 盈利动量 (需要财报)

        # 股东户数: merge_asof
        if not holder.empty:
            sym_hold = holder[holder["symbol"] == sym].sort_values("end_date")
            if not sym_hold.empty:
                out["_tmp_date"] = out["trade_date"]
                out = pd.merge_asof(
                    out.sort_values("_tmp_date"),
                    sym_hold[["end_date", "holder_signal"]].sort_values("end_date"),
                    left_on="_tmp_date", right_on="end_date", direction="backward"
                )
                out.drop(columns=["_tmp_date", "end_date"], inplace=True, errors="ignore")
                if "holder_signal" in out.columns:
                    out["q_holder"] = out["holder_signal"]
                    out.drop(columns=["holder_signal"], inplace=True)

        # F-Score: merge_asof (快速, 逐日不循环)
        if not fscore_df.empty:
            sym_fs = fscore_df[fscore_df["symbol"] == sym].sort_values("pubDate")
            if not sym_fs.empty:
                out["_tmp_date"] = out["trade_date"]
                out = pd.merge_asof(
                    out.sort_values("_tmp_date"),
                    sym_fs[["pubDate", "fscore"]].sort_values("pubDate"),
                    left_on="_tmp_date", right_on="pubDate", direction="backward"
                )
                out.drop(columns=["_tmp_date", "pubDate"], inplace=True, errors="ignore")
                if "fscore" in out.columns:
                    out["q_fscore"] = out["fscore"]
                    out.drop(columns=["fscore"], inplace=True)

        # ** Short-term Reversal (Jegadeesh 1990) **
        rev = np.full(n, np.nan)
        if n >= 21:
            rev[21:] = c[21:] / c[:-21] - 1
        out["r_short_rev"] = -rev  # 取反: 跌了会反弹

        # ** Quality 子指标 **
        out["q_roe"] = out["q_gross"] = out["q_accrual"] = out["q_leverage"] = out["q_stability"] = np.nan
        out["q_fscore"] = np.nan  # Piotroski F-Score
        out["q_holder"] = np.nan  # 股东户数变化

        sf = hist_fin[hist_fin["symbol"] == sym].sort_values("pubDate")
        if not sf.empty:
            for idx, row in out.iterrows():
                td = row["trade_date"]
                prev = sf[sf["pubDate"] <= td]
                if not prev.empty:
                    r = prev.iloc[-1]
                    # Quality: 5 indicators
                    if "roe" in r and not pd.isna(r["roe"]):
                        out.at[idx, "q_roe"] = r["roe"]
                    if "gpMargin" in r and not pd.isna(r["gpMargin"]):
                        out.at[idx, "q_gross"] = r["gpMargin"]
                    if "debt_ratio" in r and not pd.isna(r["debt_ratio"]):
                        out.at[idx, "q_leverage"] = -r["debt_ratio"]
                    # Accruals: 1 - CFO/NP (Sloan 1996)
                    if "CFOtoNP" in r and not pd.isna(r["CFOtoNP"]):
                        out.at[idx, "q_accrual"] = -(1.0 - r["CFOtoNP"])  # high accrual = bad
                    # Stability: lower std is better
                    if "roe_stability" in r and not pd.isna(r["roe_stability"]):
                        out.at[idx, "q_stability"] = -r["roe_stability"]

                    # F-Score 已在后面预计算, 此处用 merge_asof 合并

                    # CF/P for Value: (netProfit × CFOtoNP) / (close × totalShares ≈ marketCap proxy)
                    if "CFOtoNP" in r and "netProfit" in r and not pd.isna(r["CFOtoNP"]):
                        # Approximate total shares from PE = marketCap/netProfit → marketCap = PE*netProfit
                        pe_val = out.at[idx, "pe_ttm"] if "pe_ttm" in out.columns else np.nan
                        if not pd.isna(pe_val) and pe_val > 0:
                            mktcap = pe_val * r["netProfit"]
                            cfo = r["netProfit"] * r["CFOtoNP"]
                            if mktcap > 0:
                                out.at[idx, "v_cfp"] = cfo / mktcap

                    # Earnings momentum: YoY net profit growth
                    if "netProfit" in r:
                        prev_q = sf[(sf["pubDate"] <= td) & (sf["statDate"] < r["statDate"])]
                        if len(prev_q) >= 4:  # 一年前
                            prev_np = prev_q.iloc[-4]["netProfit"]
                            if prev_np and prev_np != 0:
                                out.at[idx, "m_earn"] = r["netProfit"] / prev_np - 1

        results.append(out)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ── 计算 ──
signals = compute_all_signals(raw, val, fin)
logger.info(f"信号: {len(signals)}行")

# 分析师因子 (横截面, 非时间序列)
if not analyst_fc.empty:
    signals = signals.merge(
        analyst_fc[['symbol', 'buy_ratio', 'eps_growth', 'report_count']],
        on='symbol', how='left'
    )
    signals['q_analyst_buy'] = signals['buy_ratio'].fillna(0)
    signals['q_analyst_eps'] = signals['eps_growth'].fillna(0)
    signals.drop(columns=['buy_ratio', 'eps_growth', 'report_count'], inplace=True, errors='ignore')
else:
    signals['q_analyst_buy'] = signals['q_analyst_eps'] = 0

if not analyst_visit.empty:
    signals = signals.merge(
        analyst_visit[['symbol', 'visit_times', 'avg_institutions']],
        on='symbol', how='left'
    )
    signals['q_analyst_visit'] = signals['visit_times'].fillna(0)
    signals.drop(columns=['visit_times', 'avg_institutions'], inplace=True, errors='ignore')
else:
    signals['q_analyst_visit'] = 0

# Z-score each sub-indicator → composite strategy score
value_subs = ["v_ep", "v_bp", "v_cfp"]
quality_subs = ["q_roe", "q_gross", "q_accrual", "q_leverage", "q_stability", "q_fscore", "q_strategic", "q_holder", "q_analyst_buy", "q_analyst_eps", "q_analyst_visit"]

# 战略行业得分: 从预计算映射读取 Tier1=1.0, Tier2=0.5, Tier3=0.0
strategy_tier_map = {}
try:
    ind_data = json.loads(open(".industry_names.json").read())
    tier_dict = ind_data.get("strategic_tier", {})
    # strategic_tier: {行业code(string): 1/2/3}
    for code_str, tier in tier_dict.items():
        strategy_tier_map[int(code_str)] = float(1.0 if tier == 1 else (0.5 if tier == 2 else 0.0))
except: pass

if industry_map and strategy_tier_map:
    signals["q_strategic"] = signals["symbol"].apply(
        lambda s: strategy_tier_map.get(industry_map.get(str(s), -1), 0.0)
    )
else:
    signals["q_strategic"] = 0
reversal_subs = ["r_short_rev"]

# 截面动量: 每日对所有股票的 m_price 做 Z-score
if "m_price" in signals.columns:
    signals["m_cross"] = signals.groupby("trade_date")["m_price"].transform(
        lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
    )
else:
    signals["m_cross"] = 0

# 行业动量: 每个行业的平均动量
if industry_map and "m_price" in signals.columns:
    signals["_ind"] = signals["symbol"].map(industry_map).fillna(-1)
    ind_avg = signals.groupby(["_ind", "trade_date"])["m_price"].transform("mean")
    signals["m_ind"] = ind_avg
    signals.drop(columns=["_ind"], inplace=True)
else:
    signals["m_ind"] = 0

def zscore_equal_weight(df, subs, name):
    """子指标 Z-score → 等权合成 (行业中性化)"""
    # 先标记行业
    df["_ind"] = df["symbol"].map(industry_map).fillna(-1) if industry_map else -1

    cols = []
    for s in subs:
        if s in df.columns and df[s].notna().any():
            # 行业中性化: 先在全市场Z-score, 再减去行业均值
            mu_all, sigma_all = df[s].mean(), df[s].std()
            if sigma_all and sigma_all > 0:
                df[s + "_z"] = (df[s] - mu_all) / sigma_all
            else:
                df[s + "_z"] = 0
            # 行业内去均值 (消除行业偏差)
            ind_mean = df.groupby("_ind")[s + "_z"].transform("mean")
            df[s + "_z"] = df[s + "_z"] - ind_mean
            cols.append(s + "_z")
    if cols:
        df[name] = df[cols].mean(axis=1)
    else:
        df[name] = 0
    df.drop(columns=["_ind"], inplace=True, errors="ignore")

zscore_equal_weight(signals, value_subs, "value")
zscore_equal_weight(signals, quality_subs, "quality")
zscore_equal_weight(signals, reversal_subs, "reversal")

# ── 回测 ──
signals["trade_date"] = pd.to_datetime(signals["trade_date"])
dates = sorted(signals["trade_date"].unique())
fridays = [d for d in dates if d.weekday() == 4]
logger.info(f"回测: {len(fridays)}周")

cash_v, cash_q, cash_r = 350000, 550000, 100000  # V35 Q55 R10

for i, fri in enumerate(fridays):
    day = signals[signals["trade_date"] == fri]
    if len(day) < TOPN:
        continue
    nxt = fridays[i + 1] if i < len(fridays) - 1 else dates[-1]

    def sret(df, col, n):
        t = df.nlargest(n, col)
        rets = []
        for s in t["symbol"]:
            sd = raw[(raw["symbol"] == s) & (raw["trade_date"] > fri) & (raw["trade_date"] <= nxt)]
            sd = sd.sort_values("trade_date")
            if len(sd) >= 2:
                rets.append(sd["close"].iloc[-1] / sd["close"].iloc[0] - 1)
        return np.mean(rets) if rets else 0

    cash_v *= 1 + sret(day, "value", TOPN)
    cash_q *= 1 + sret(day, "quality", TOPN)
    cash_r *= 1 + sret(day, "reversal", TOPN)

    if (i + 1) % 30 == 0:
        total = cash_v + cash_q + cash_r
        logger.info(f"  V{cash_v/350000-1:+.1%} Q{cash_q/550000-1:+.1%} R{cash_r/100000-1:+.1%} T=¥{total:,.0f}")

total = cash_v + cash_q + cash_r
total_ret = total / 1_000_000 - 1
n_days = (fridays[-1] - fridays[0]).days
annual = (1 + total_ret) ** (365.25 / max(n_days, 1)) - 1

logger.info(f"\n{'='*50}")
logger.info(f"3策略({N}只/{len(fridays)}周) — 全子指标版")
logger.info(f"  Value:     {cash_v/350000-1:+.1%}  (E/P + B/P + CF/P) [35%]")
logger.info(f"  Quality:   {cash_q/550000-1:+.1%}  (ROE+毛利率+应计+杠杆+稳定性+FScore+战略+股东户数) [55%]")
logger.info(f"  Reversal:  {cash_r/100000-1:+.1%}  (1月反转) [10%]")
logger.info(f"  年化: {annual:+.1%}")
logger.info(f"  终值: ¥{total:,.0f}")
