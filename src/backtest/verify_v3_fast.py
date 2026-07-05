"""
v3.0 快速回测 — 向量化因子 + baostock 历史PE/PB

用法:
    python src/backtest/verify_v3_fast.py --stocks 100
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("v3_fast")

# ── 因子权重 (BlackRock标准) ──
W_VALUE, W_MOMENTUM, W_QUALITY, W_LOWVOL = 0.25, 0.25, 0.40, 0.10
TOP_N = 20


def compute_daily_factors(price_df, hist_val, hist_fin):
    """向量化计算每日因子 (含 Quality)"""
    df = price_df.sort_values(['symbol', 'trade_date']).copy()
    df['trade_date'] = pd.to_datetime(df['trade_date'])

    results = []
    for sym, grp in df.groupby('symbol'):
        grp = grp.sort_values('trade_date')
        c = grp['close'].values
        n = len(c)
        if n < 60:
            continue

        out = grp[['trade_date', 'symbol']].copy()

        # Momentum + LowVol (from price)
        lb = min(252, n)
        rets = np.diff(c[-lb:]) / np.maximum(np.abs(c[-lb:-1]), 1e-8)
        out['lowvol'] = -np.nanstd(rets) * np.sqrt(252)

        mom = np.full(n, np.nan)
        if n >= 252:
            for i in range(252, n):
                if c[i-21] > 0:
                    mom[i] = c[i-21] / c[i-252] - 1
        out['momentum'] = mom

        # Value from historical PE
        if hist_val is not None and not hist_val.empty:
            sym_pe = hist_val[hist_val['symbol'] == sym]
            if not sym_pe.empty:
                sym_pe = sym_pe.copy()
                sym_pe['trade_date'] = pd.to_datetime(sym_pe['trade_date'])
                out = out.merge(sym_pe[['trade_date', 'pe_ttm']], on='trade_date', how='left')
                out['value'] = 1.0 / out['pe_ttm'].clip(lower=1.0)
        if 'value' not in out.columns:
            out['value'] = np.nan

        # Quality from point-in-time financial data
        out['quality'] = np.nan
        if hist_fin is not None and not hist_fin.empty:
            sym_fin = hist_fin[hist_fin['symbol'] == sym].sort_values('pubDate')
            if not sym_fin.empty:
                for idx, row in out.iterrows():
                    td = row['trade_date']
                    prev = sym_fin[sym_fin['pubDate'] <= td]
                    if not prev.empty:
                        latest = prev.iloc[-1]
                        # Quality = Z(ROE) + Z(-debt)
                        roe_v = latest.get('roe', np.nan)
                        debt_v = latest.get('debt_ratio', np.nan)
                        if not pd.isna(roe_v) and not pd.isna(debt_v):
                            out.at[idx, 'quality'] = roe_v - debt_v  # 简化的 quality 得分

        results.append(out)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def backtest_v3_fast(n_stocks=100, start_date=None, end_date=None):
    """快速小规模回测"""
    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=365 * 3)

    logger.info(f"v3.0 快速回测: {n_stocks}只, {start_date}~{end_date}")

    # 1. 价格数据
    raw = read_daily_bars(A_STOCK_DIR, start_date=start_date - timedelta(days=400),
                          end_date=end_date, market='a_stock')
    syms = sorted(raw['symbol'].unique())[:n_stocks]
    raw = raw[raw['symbol'].isin(syms)]
    logger.info(f"[1/4] 价格: {len(raw)}行, {len(syms)}只")

    # 2. 历史PE + 季度财报
    hist_val = pd.DataFrame()
    hist_fin = pd.DataFrame()
    try:
        from src.data.baostock_fetcher import fetch_historical_valuation
        hist_val = fetch_historical_valuation(
            symbols=syms,
            start_date=(start_date - timedelta(days=400)).strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        logger.info(f"[2a/4] 历史PE: {len(hist_val)}行")
    except Exception as e:
        logger.warning(f"  PE跳过: {e}")

    try:
        from src.data.baostock_financial import fetch_quarterly_financials
        hist_fin = fetch_quarterly_financials(symbols=syms[:200])
        if hist_fin is not None and not hist_fin.empty:
            hist_fin['pubDate'] = pd.to_datetime(hist_fin['pubDate'])
        logger.info(f"[2b/4] 历史财报: {len(hist_fin)}行")
    except Exception as e:
        logger.warning(f"  财报跳过: {e}")

    # 3. 计算每日因子
    logger.info("[3/4] 计算因子...")
    factors = compute_daily_factors(raw, hist_val, hist_fin)
    if factors.empty:
        logger.error("无因子数据")
        return

    # 加权总分 (4因子 BlackRock权重)
    for col in ['value', 'momentum', 'quality', 'lowvol']:
        if col in factors.columns and factors[col].notna().any():
            mu, sigma = factors[col].mean(), factors[col].std()
            if sigma and sigma > 0:
                factors[col + '_z'] = (factors[col] - mu) / sigma
            else:
                factors[col + '_z'] = 0
        else:
            factors[col + '_z'] = 0

    factors['composite'] = (
        factors['value_z'].fillna(0) * W_VALUE +
        factors['momentum_z'].fillna(0) * W_MOMENTUM +
        factors['quality_z'].fillna(0) * W_QUALITY +
        factors['lowvol_z'].fillna(0) * W_LOWVOL
    )

    # 4. 逐周回测
    factors['trade_date'] = pd.to_datetime(factors['trade_date'])
    raw['trade_date'] = pd.to_datetime(raw['trade_date'])
    dates = sorted(factors['trade_date'].unique())
    fridays = [d for d in dates if d.weekday() == 4]
    if not fridays:
        fridays = dates[::5]

    cash, cash_bench = 1_000_000, 1_000_000
    history = []

    for i, fri in enumerate(fridays):
        day_data = factors[factors['trade_date'] == fri]
        if len(day_data) < TOP_N:
            continue

        next_fri = fridays[i+1] if i < len(fridays)-1 else dates[-1]

        # === 策略 ===
        top = day_data.nlargest(TOP_N, 'composite')
        selected = top['symbol'].tolist()
        returns = []
        for sym in selected:
            sd = raw[(raw['symbol'] == sym) & (raw['trade_date'] > fri) & (raw['trade_date'] <= next_fri)]
            sd = sd.sort_values('trade_date')
            if len(sd) >= 2:
                returns.append(sd['close'].iloc[-1] / sd['close'].iloc[0] - 1)
        if returns:
            cash *= (1 + np.mean(returns))

        # === 等权基准 ===
        all_syms = day_data['symbol'].tolist()
        bench_returns = []
        for sym in all_syms[:100]:  # 100只采样代表基准
            sd = raw[(raw['symbol'] == sym) & (raw['trade_date'] > fri) & (raw['trade_date'] <= next_fri)]
            sd = sd.sort_values('trade_date')
            if len(sd) >= 2:
                bench_returns.append(sd['close'].iloc[-1] / sd['close'].iloc[0] - 1)
        if bench_returns:
            cash_bench *= (1 + np.mean(bench_returns))

        history.append({'date': fri, 'value': cash, 'bench': cash_bench, 'n': len(selected)})

        if (i+1) % 30 == 0:
            logger.info(f"  [{i+1}/{len(fridays)}] 策略¥{cash:,.0f} | 基准¥{cash_bench:,.0f}")

    # 4. 结果
    hist_df = pd.DataFrame(history)
    if hist_df.empty:
        return

    total_ret = hist_df['value'].iloc[-1] / hist_df['value'].iloc[0] - 1
    bench_ret = hist_df['bench'].iloc[-1] / hist_df['bench'].iloc[0] - 1 if 'bench' in hist_df.columns else 0
    n_days = (hist_df['date'].iloc[-1] - hist_df['date'].iloc[0]).days
    annual = (1+total_ret)**(365.25/max(n_days,1)) - 1
    annual_bench = (1+bench_ret)**(365.25/max(n_days,1)) - 1
    weekly = hist_df['value'].pct_change().dropna()
    sharpe = weekly.mean() / weekly.std() * np.sqrt(52) if weekly.std() > 0 else 0
    peak = hist_df['value'].cummax()
    max_dd = (hist_df['value'] - peak).min() / peak.max()
    excess = annual - annual_bench

    logger.info(f"\n{'='*50}")
    logger.info(f"v3.0 回测 ({n_stocks}只, {len(fridays)}周)")
    logger.info(f"  策略年化: {annual:.1%}  |  基准年化: {annual_bench:.1%}  |  超额: {excess:+.1%}")
    logger.info(f"  夏普: {sharpe:.2f}")
    logger.info(f"  最大回撤: {max_dd:.1%}")
    logger.info(f"  胜率: {(weekly > 0).mean():.1%}")
    logger.info(f"  策略终值: ¥{hist_df['value'].iloc[-1]:,.0f}  |  基准终值: ¥{hist_df['bench'].iloc[-1]:,.0f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stocks", type=int, default=100)
    args = p.parse_args()
    backtest_v3_fast(n_stocks=args.stocks)
