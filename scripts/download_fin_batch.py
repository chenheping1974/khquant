#!/usr/bin/env python3
"""全量财报下载 — 东财批量接口, 6000只/次, 5秒"""
import sys, time, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np, pandas as pd, akshare as ak

logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('batch')

# 8个季度
quarters = ['20230331','20230630','20230930','20231231',
            '20240331','20240630','20240930','20241231',
            '20250331','20250630','20250930','20251231','20260331']

all_data = []
t0 = time.time()
for q in quarters:
    try:
        df = ak.stock_yjbb_em(date=q)
        if df is not None and not df.empty:
            df = df.rename(columns={
                '股票代码':'symbol','每股收益':'eps','营业总收入-营业总收入':'revenue',
                '营业总收入-同比增长':'revenue_growth','净利润-净利润':'net_profit',
                '净利润-同比增长':'profit_growth','净资产收益率':'roe',
                '销售毛利率':'gross_margin','每股经营现金流量':'ocf_per_share',
                '每股净资产':'bvps','所处行业':'industry'
            })
            df['symbol'] = df['symbol'].astype(str).str.zfill(6)
            df['report_date'] = f"{q[:4]}-{q[4:6]}-{q[6:]}"
            cols = ['symbol','report_date','eps','revenue','revenue_growth',
                    'net_profit','profit_growth','roe','gross_margin','ocf_per_share',
                    'bvps','industry']
            all_data.append(df[[c for c in cols if c in df.columns]])
            logger.info(f'{q}: {len(df)}只 ({time.time()-t0:.0f}s)')
    except Exception as e:
        logger.warning(f'{q}: {e}')
    time.sleep(1)

result = pd.concat(all_data, ignore_index=True)
result.to_parquet('.cache_fin_batch.parquet')
n = result['symbol'].nunique()
logger.info(f'批量数据: {len(result)}条 {n}只 ({(time.time()-t0):.0f}s)')

# 合并: 批量数据 + 负债率 + 生成最终推理缓存
logger.info('合并负债率 & 生成 .cache_fin_infer.parquet...')
latest = result[result['report_date'] == '2026-03-31'].copy()
latest = latest.rename(columns={'net_profit':'netProfit','gross_margin':'gpMargin',
    'profit_growth':'profitGrowth','revenue_growth':'revenueGrowth'})

# 下载负债率
import requests, json, time as t
prefixes = ['000','001','002','003','300','301','600','601','603','605','688','689','920']
debt_rows = []
for pf in prefixes:
    try:
        r = requests.get("https://datacenter.eastmoney.com/securities/api/data/v1/get",
            params={"reportName":"RPT_DMSK_FN_BALANCE","columns":"SECURITY_CODE,DEBT_ASSET_RATIO",
            "filter":f'(SECURITY_CODE>="{pf}000")(SECURITY_CODE<="{pf}999")(REPORT_DATE>=\'2026-03-01\')',
            "pageNumber":1,"pageSize":2000,"sortTypes":1,"sortColumns":"SECURITY_CODE",
            "source":"HSF10","client":"PC"},headers={'User-Agent':'Mozilla/5.0'},timeout=10)
        for row in r.json().get('result',{}).get('data',[]):
            dr = row.get('DEBT_ASSET_RATIO')
            if dr and dr != '':
                debt_rows.append({'symbol':str(row['SECURITY_CODE']).zfill(6),'debt_ratio':float(dr)})
    except: pass
    t.sleep(0.3)
debt_df = pd.DataFrame(debt_rows)
latest = latest.merge(debt_df, on='symbol', how='left')
real_n = latest['debt_ratio'].notna().sum()
# 行业均值填充缺失
ind = json.load(open('.industry_names.json')); code_to_name = ind.get('code_to_name',{})
ind_map = json.load(open('.industry_cache.json'))
latest['_ind'] = latest['symbol'].map(lambda s: code_to_name.get(str(ind_map.get(str(s),-1)),'未知'))
ind_avg = latest.groupby('_ind')['debt_ratio'].mean().fillna(0.5)
latest['debt_ratio'] = latest.apply(lambda r: r['debt_ratio'] if not pd.isna(r['debt_ratio']) else ind_avg.get(r['_ind'],0.5), axis=1)
latest.drop(columns=['_ind'], inplace=True)
latest['pubDate'] = pd.Timestamp.now()
latest.to_parquet('.cache_fin_infer.parquet')
logger.info(f'推理缓存: {len(latest)}只, 真实负债率:{real_n}')
