# khquant 数据源详细清单

## 1. 日频 K线 — Sina

```
接口: https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData
参数: symbol=sh600519, scale=240, ma=no, datalen=5000
返回: [{day,open,high,low,close,volume}, ...]
频率: 日, 增量模式 (datalen=1)
文件: src/data/fetcher.py → fetch_a_stock_daily()
脚本: scripts/fetch_incremental.py
```

## 2. PE/PB/市值 — 腾讯财经

```
接口: https://web.sqt.gtimg.cn/q=sh600519,sz000001
参数: 股票列表逗号分隔, 单次50只
返回: 88字段, ~分隔
  字段[3]=最新价, [39]=PE(TTM), [46]=PB, [44]=市值, [72]=总股本, [63]=股息率
频率: 实时, 每次拉取
文件: src/data/tencent_fetcher.py → fetch_valuation_batch()
推理: daily_inference.py → fetch_valuation_batch(symbols=syms[:N_PE])
```

## 3. 批量财报 — akshare stock_yjbb_em (⭐主力)

```
接口: akshare.stock_yjbb_em(date='20260331')
参数: date=YYYYMMDD (报告期)
返回: 5964只/次, 字段:
  股票代码(symbol), 每股收益(eps), 营业总收入(revenue),
  营业总收入-同比增长(revenue_growth), 净利润(net_profit),
  净利润-同比增长(profit_growth), 净资产收益率(roe),
  销售毛利率(gross_margin), 每股经营现金流量(ocf_per_share),
  所处行业(industry), 每股净资产, 最新公告日期
频率: 季度, 102秒/13季度
文件: scripts/download_fin_batch.py
工作流: .github/workflows/quarterly.yml
```

## 4. 详细财报 — akshare stock_financial_analysis_indicator (⚠️逐只,会封)

```
接口: akshare.stock_financial_analysis_indicator(symbol='600519', start_year='2023')
参数: symbol(6位代码), start_year
返回: 86字段, 含:
  净资产收益率(%), 主营业务利润率(%), 资产负债率(%),
  摊薄每股收益(元), 净利润(元), 销售净利率(%),
  经营现金净流量与净利润的比率(%)
频率: 季度, 逐只调用, 1s/只, 会封IP
文件: scripts/download_fin_q1.py (备用)
用途: 补充 debt_ratio + CFOtoNP (批量接口缺这两个)
```

## 5. 资产负债表(负债率) — 东财 datacenter

```
接口: https://datacenter.eastmoney.com/securities/api/data/v1/get
参数:
  reportName=RPT_DMSK_FN_BALANCE
  columns=SECURITY_CODE,DEBT_ASSET_RATIO
  filter=(SECURITY_CODE>="000000")(SECURITY_CODE<="000999")(REPORT_DATE>='2026-03-01')
  pageSize=2000
  source=HSF10, client=PC
返回: [{SECURITY_CODE, DEBT_ASSET_RATIO}, ...] (500条/批)
频率: 季度, 按代码前缀分批(000/002/300/600/601/603/605/688)
覆盖: 3251只
```

## 6. 股东户数 — 东财 datacenter

```
接口: https://datacenter.eastmoney.com/securities/api/data/v1/get
参数:
  reportName=RPT_F10_EH_HOLDERNUM
  columns=SECURITY_CODE,END_DATE,HOLDER_TOTAL_NUM
  filter=(SECURITY_CODE="600519")
  pageNumber=1, pageSize=20
  sortTypes=-1, sortColumns=END_DATE
  source=HSF10, client=PC
返回: [{SECURITY_CODE, END_DATE, HOLDER_TOTAL_NUM}, ...]
频率: 季度, 每只0.1s
文件: 内联在 daily_inference.py + monthly.yml
缓存: .cache_holder_all.parquet
```

## 6. 公告情绪 — 东财公告

```
接口: https://np-anotice-stock.eastmoney.com/api/security/ann
参数: page_size=30, page_index=1, ann_type=A, stock_list=600519
返回: [{title, notice_date}, ...]
关键词匹配:
  利好: 业绩预增,中标,回购,增持,分红,送转,预盈,扭亏,重大合同,突破,获批,注册
  利空: 减持,亏损,退市,立案,警示,问询,处罚,诉讼,冻结,终止,修正,下调
频率: 日, 每只0.1s
文件: scripts/download_announce.py (断点续传)
缓存: .cache_announce.parquet
工作流: .github/workflows/monthly.yml
```

## 7. 分析师评级 — akshare stock_profit_forecast_em

```
接口: akshare.stock_profit_forecast_em()
参数: 无参
返回: 2785只, 字段:
  代码, 名称, 研报数,
  机构投资评级(近六个月)-买入/增持/中性/减持/卖出,
  2025/2026/2027预测每股收益
频率: 日
推理: daily_inference.py 实时拉取
缓存: .cache_analyst_fc.parquet
```

## 8. 机构调研 — akshare stock_jgdy_tj_em

```
接口: akshare.stock_jgdy_tj_em()
参数: 无参
返回: 22278条, 字段:
  代码, 名称, 接待机构数量, 接待日期, 接待方式
频率: 日
推理: daily_inference.py 实时拉取
缓存: .cache_analyst_visit.parquet
```

## 9. 行业分类 — 东财 F10

```
接口: https://emweb.securities.eastmoney.com/PC_HSF10/CompanySurvey/CompanySurveyAjax
参数: code=SH600519 或 SZ000001
返回: jbzl.sshy (申万行业), jbzl.sszjhhy (证监会行业)
频率: 静态, 逐个查询
文件: scripts/download_industries.py
缓存: .industry_cache.json (symbol→行业code)
     .industry_names.json (code→行业名, 战略分级)
```

## 10. 舆情热度 — akshare stock_hot_rank_em

```
接口: akshare.stock_hot_rank_em()
参数: 无参
返回: 100只, 字段: 当前排名, 代码, 股票名称, 最新价, 涨跌幅
频率: 日
推理: daily_inference.py 实时拉取 (仅当日有效)
```

## 缓存文件说明

```
.cache_fin_infer.parquet  → Q1批量财报 (EPS,ROE,debt_ratio等) ~5878只
.cache_fin_batch.parquet  → 全13季批量财报                           ~110K条
.cache_holder_all.parquet → 股东户数                               4944只
.cache_announce.parquet   → 公告情绪                               3398只
.cache_analyst_fc.parquet → 分析师评级                             2359只
.cache_analyst_visit.parquet → 机构调研                            4682只
.industry_cache.json      → 股票→行业编码                          1496只
.industry_names.json      → 行业编码→名称+战略分级                   288行业
```

## 工作流说明

```
daily.yml     每晚20:00  K线增量(Sina) → 推理(腾讯PE+缓存) → 结果push
monthly.yml   每月1号     股东户数(东财datacenter) + 公告(东财公告API)
quarterly.yml 季报期周六  财报(akshare stock_yjbb_em)
```
