"""
v3.0 客观回测 — 500只随机抽样, 时间点对齐, akshare财报

随机分层抽样: 主板150 + 中小板200 + 创业板150 = 500
历史PE: 从季度EPS推算 (PE = close / EPS)
调仓: 周度
成本: 印花税0.05% + 佣金0.03% + 滑点0.1%
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import random
import logging
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("proper_bt")

# ── 1. 随机分层抽样 500 只 ──
end = date.today()
start = end - timedelta(days=365 * 3)
raw_all = read_daily_bars(A_STOCK_DIR, start_date=start - timedelta(days=400),
                          end_date=end, market="a_stock")
all_syms = sorted(raw_all["symbol"].unique())

random.seed(42)
main_board = [s for s in all_syms if s.startswith("6")]
sme_board = [s for s in all_syms if s.startswith("00")]
chinext = [s for s in all_syms if s.startswith("30")]

selected = (random.sample(main_board, min(150, len(main_board))) +
            random.sample(sme_board, min(200, len(sme_board))) +
            random.sample(chinext, min(150, len(chinext))))
logger.info(f"抽样: 主板{len(main_board[:150])} + 中小板{len(sme_board[:200])} + 创业板{len(chinext[:150])} = {len(selected)}只")

# ── 2. 下载历史财报 (akshare, 每只1次API) ──
cf = f".cache_bt_fin.pkl"
import pickle, time, akshare as ak

if Path(cf).exists():
    bt_fin = pickle.loads(Path(cf).read_bytes())
    logger.info(f"财报缓存: {len(bt_fin)}只")
else:
    bt_fin = {}
    t0 = time.time()
    for i, code in enumerate(selected):
        cs = str(code).zfill(6)
        try:
            df = ak.stock_financial_analysis_indicator(symbol=cs, start_year="2023")
            if df is not None and not df.empty:
                # 标准化列名
                df = df.rename(columns={
                    '净资产收益率(%)': 'roe', '主营业务利润率(%)': 'gross_margin',
                    '资产负债率(%)': 'debt_ratio', '摊薄每股收益(元)': 'eps',
                    '销售净利率(%)': 'net_margin', '净利润(元)': 'net_profit',
                    '经营现金净流量与净利润的比率(%)': 'cfo_np'
                })
                df['report_date'] = pd.to_datetime(df.iloc[:, 0], errors='coerce')
                keep_cols = ['report_date', 'roe', 'gross_margin', 'debt_ratio',
                             'eps', 'net_margin', 'net_profit', 'cfo_np']
                bt_fin[cs] = df[[c for c in keep_cols if c in df.columns]].copy()
        except: pass
        if (i + 1) % 100 == 0:
            logger.info(f"  [{i+1}/{len(selected)}] {len(bt_fin)}只 {(time.time()-t0)/60:.0f}min")
    Path(cf).write_bytes(pickle.dumps(bt_fin))
    logger.info(f"财报完成: {len(bt_fin)}只 ({(time.time()-t0)/60:.0f}min)")

# ── 3. 计算因子 + 回测 ──
raw = raw_all[raw_all["symbol"].isin(selected)].copy()
raw["trade_date"] = pd.to_datetime(raw["trade_date"])
logger.info(f"价格数据: {len(raw)}行")

# 预计算收益矩阵
close_m = raw.pivot_table(index="trade_date", columns="symbol", values="close", aggfunc="last")
close_m = close_m.sort_index().ffill()
fwd_ret = close_m.pct_change().shift(-1)

# 因子计算
def compute_factors(date_val):
    """对指定日期计算所有股票的四因子"""
    pd_date = pd.Timestamp(date_val)
    window = raw[(raw["trade_date"] >= pd_date - pd.Timedelta(days=400)) &
                 (raw["trade_date"] <= pd_date)]
    results = []
    for sym in selected:
        grp = window[window["symbol"] == sym].sort_values("trade_date")
        if len(grp) < 60: continue
        c = grp["close"].values; n = len(c)
        row = {"symbol": sym}

        # 价值: PE = close / latest_eps
        fin_data = bt_fin.get(str(sym).zfill(6))
        if fin_data is not None and not fin_data.empty and 'eps' in fin_data.columns:
            latest_fin = fin_data[fin_data['report_date'] <= pd_date]
            if not latest_fin.empty:
                eps = latest_fin.iloc[-1]['eps']
                if eps and eps > 0:
                    row['v_ep'] = float(eps) / c[-1]  # E/P
                row['q_roe'] = float(latest_fin.iloc[-1].get('roe', np.nan)) / 100 if latest_fin.iloc[-1].get('roe') else np.nan
                row['q_leverage'] = -float(latest_fin.iloc[-1].get('debt_ratio', 50)) / 100 if latest_fin.iloc[-1].get('debt_ratio') else np.nan

        # 动量
        if n >= 252 and c[-21] > 0: row['momentum'] = c[-21] / c[-252] - 1
        # 反转
        if n >= 21: row['reversal'] = -(c[-1] / c[-22] - 1)
        # 低波
        if n >= 252:
            rets = np.diff(c[-252:]) / np.maximum(np.abs(c[-252:-1]), 1e-8)
            row['lowvol'] = -np.nanstd(rets) * np.sqrt(252)

        results.append(row)
    return pd.DataFrame(results)

# 逐周回测
dates = sorted(close_m.index)
fridays = [d for d in dates if d.weekday() == 4]
logger.info(f"回测: {len(fridays)}周")

cash = 1_000_000
TOPN = 30
COST = 0.0018  # 单边成本
prev = set()

for i, fri in enumerate(fridays):
    factors = compute_factors(fri)
    if len(factors) < TOPN: continue

    # Z-score + 等权 (简化: momentum60% + value40%)
    for col in ['v_ep', 'momentum', 'reversal', 'lowvol', 'q_roe', 'q_leverage']:
        if col in factors.columns and factors[col].notna().any():
            mu, sigma = factors[col].mean(), factors[col].std()
            if sigma and sigma > 0: factors[col + '_z'] = (factors[col] - mu) / sigma
            else: factors[col + '_z'] = 0
        else: factors[col + '_z'] = 0

    z_cols = [c for c in factors.columns if c.endswith('_z')]
    factors['composite'] = factors[z_cols].mean(axis=1) if z_cols else 0
    top = factors.nlargest(TOPN, 'composite')
    new_holdings = set(top['symbol'].tolist())

    if fri in fwd_ret.index:
        rets = [fwd_ret.loc[fri].get(s) for s in new_holdings]
        rets = [r for r in rets if r is not None and not pd.isna(r)]
        if rets:
            turnover = len(new_holdings - prev) / TOPN
            cash *= (1 + np.mean(rets) - turnover * COST)
    prev = new_holdings

    if (i + 1) % 40 == 0:
        total_ret = cash / 1_000_000 - 1
        logger.info(f"  [{i+1}/{len(fridays)}] ¥{cash:,.0f} ({total_ret:+.1%})")

total_ret = cash / 1_000_000 - 1
n_days = (fridays[-1] - fridays[0]).days
annual = (1 + total_ret) ** (365.25 / max(n_days, 1)) - 1
logger.info(f"\n{'='*50}")
logger.info(f"客观回测 (500只随机, {len(fridays)}周)")
logger.info(f"  年化: {annual:+.1%}")
logger.info(f"  终值: ¥{cash:,.0f}")
