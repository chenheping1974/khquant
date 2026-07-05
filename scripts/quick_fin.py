"""快速重下财报 (仅利润表, 加npMargin)"""
import sys; sys.path.insert(0,'.')
import numpy as np, pandas as pd, time
from datetime import date, timedelta
from config import A_STOCK_DIR
from src.data.storage import read_daily_bars

if not hasattr(pd.DataFrame,'append'):
    pd.DataFrame.append=lambda s,o,**kw:pd.concat([s,o],ignore_index=kw.get('ignore_index',False))
import baostock as bs, logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s [%(levelname)s] %(message)s')
logger=logging.getLogger('quick')

end=date.today();start=end-timedelta(days=365*3)
syms=sorted(read_daily_bars(A_STOCK_DIR,start_date=start,end_date=end,market='a_stock')['symbol'].unique())[:200]

bs.login(); rows=[]; t0=time.time()
for i,code in enumerate(syms):
    cs=str(code).zfill(6);pf='sh' if cs.startswith('6') else 'sz';fc=f'{pf}.{cs}'
    for y in range(2023,2027):
        for q in [1,2,3,4]:
            try:
                rs=bs.query_profit_data(code=fc,year=y,quarter=q);df=rs.get_data()
                if df is not None and not df.empty:
                    rows.append({'symbol':cs,
                        'pubDate':pd.to_datetime(df['pubDate'].iloc[0]).date(),
                        'statDate':pd.to_datetime(df['statDate'].iloc[0]).date(),
                        'roe':float(df['roeAvg'].iloc[0]) if df['roeAvg'].iloc[0]!='' else np.nan,
                        'gpMargin':float(df['gpMargin'].iloc[0]) if df['gpMargin'].iloc[0]!='' else np.nan,
                        'npMargin':float(df['npMargin'].iloc[0]) if df['npMargin'].iloc[0]!='' else np.nan,
                        'netProfit':float(df['netProfit'].iloc[0]) if df['netProfit'].iloc[0]!='' else np.nan})
            except: pass
            time.sleep(0.02)
    for y in range(2023,2027):
        for q in [1,2,3,4]:
            try:
                rs=bs.query_balance_data(code=fc,year=y,quarter=q);df=rs.get_data()
                if df is not None and not df.empty:
                    sd=pd.to_datetime(df['statDate'].iloc[0]).date()
                    for r in rows:
                        if r['symbol']==cs and r['statDate']==sd: r['debt_ratio']=float(df['liabilityToAsset'].iloc[0]) if df['liabilityToAsset'].iloc[0]!='' else np.nan
            except: pass; time.sleep(0.02)
    for y in range(2023,2027):
        for q in [1,2,3,4]:
            try:
                rs=bs.query_cash_flow_data(code=fc,year=y,quarter=q);df=rs.get_data()
                if df is not None and not df.empty:
                    sd=pd.to_datetime(df['statDate'].iloc[0]).date()
                    for r in rows:
                        if r['symbol']==cs and r['statDate']==sd: r['CFOtoNP']=float(df['CFOToNP'].iloc[0]) if df['CFOToNP'].iloc[0]!='' else np.nan
            except: pass; time.sleep(0.02)
    if (i+1)%50==0: logger.info(f'[{i+1}/200] {len(rows)}条 {(time.time()-t0)/60:.0f}min')

fin=pd.DataFrame(rows); fin=fin.sort_values(['symbol','statDate'])
fin['roe_stability']=fin.groupby('symbol')['roe'].transform(lambda x:x.rolling(8,min_periods=4).std())
fin.to_parquet('.cache_fin_200.parquet',index=False); bs.logout()
logger.info(f'完成:{len(fin)}条 {fin["symbol"].nunique()}只 npMargin覆盖:{fin["npMargin"].notna().sum()}/{len(fin)} {(time.time()-t0)/60:.0f}min')
