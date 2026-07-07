# khquant 数据源清单

## 日频数据 (每天推理需要)

| 因子 | 数据 | API | 稳定性 | 速度 | 覆盖 |
|------|------|-----|--------|------|------|
| E/P, B/P, Size | PE/PB/市值 | `腾讯财经 web.sqt.gtimg.cn/q=` | ✅ 极稳 | 50只/0.2s | 5000+ |
| 价格动量 | K线 OHLCV | `Sina money.finance.sina.com.cn` | ✅ 稳 | 1只/1s | 5000+ |
| 短期反转 | K线 OHLCV | 同上 | ✅ | | |
| 分析师买入比 | 盈利预测 | `akshare stock_profit_forecast_em()` | ⚠️ 偶尔断 | 6s/次 | 2785 |
| EPS增速 | 盈利预测 | 同上 | ⚠️ | | 2785 |
| 机构调研 | 调研数据 | `akshare stock_jgdy_tj_em()` | ⚠️ 偶尔断 | 15s/次 | 4682 |
| 舆情热度 | 热门排行 | `akshare stock_hot_rank_em()` | ⚠️ 偶尔断 | 2s/次 | 100 |

## 季度数据 (手动/Workflow 定期更新)

| 因子 | 数据 | API | 稳定性 | 速度 | 覆盖 |
|------|------|-----|--------|------|------|
| ROE, EPS, 毛利率 | 批量财报 | `akshare stock_yjbb_em(date='20260331')` | ⚠️ 偶尔断 | 6000只/5s | 5878 |
| 负债率, CFO/NP | 详细财报 | `akshare stock_financial_analysis_indicator(symbol, start_year)` | ❌ 批量会封 | 1只/1s | 1177 |
| 股东户数 | 户数数据 | `Eastmoney datacenter.eastmoney.com RPT_F10_EH_HOLDERNUM` | ✅ 稳 | 1只/0.1s | 4944 |
| 公告情绪 | 公告标题 | `Eastmoney np-anotice-stock.eastmoney.com/api/security/ann` | ✅ 稳 | 1只/0.1s | 3398 |

## 静态数据

| 因子 | 数据 | API | 覆盖 |
|------|------|-----|------|
| 行业分类 | 申万行业 | `Eastmoney emweb.securities.eastmoney.com F10 CompanySurvey` | 1496只 |
| 战略行业 | 十四五重点 | 本地映射表 (Tier1/Tier2/Tier3) | 288行业 |

## 使用方式

```
推理 (daily_inference.py):
  - 腾讯PE: 实时拉取, 5000只 ~50s
  - K线: 从 khquant-data 仓库读取
  - 财报: 读取 .cache_fin_infer.parquet (批量接口, 102s/13季)
  - 股东户数: 读取 .cache_holder_all.parquet
  - 分析师: 读取 .cache_analyst_fc.parquet
  - 机构调研: 读取 .cache_analyst_visit.parquet
  - 公告: 读取 .cache_announce.parquet
  - 行业: 读取 .industry_cache.json + .industry_names.json

回测 (verify_v3_blackrock.py):
  - 需要时间点对齐的历史 PE 和财报
  - 批量接口 stock_yjbb_em 可提供历史 EPS/ROE

更新频率:
  - 日: K线增量 (fetch_incremental.py → Sina)
  - 月: 股东户数 + 公告 (download_announce.py → Eastmoney)
  - 季: 财报 (download_fin_batch.py → akshare stock_yjbb_em)
```

## ⚠️ 避坑指南

1. **akshare stock_financial_analysis_indicator 逐只调用会封IP**
   → 用 stock_yjbb_em 批量接口替代（6000只/次）

2. **baostock 不稳定, 经常断连**
   → 不用。财报用 akshare 批量接口, PE用腾讯

3. **东财 datacenter API 需要正确的 reportName**
   → RPT_F10_EH_HOLDERNUM (股东户数) 已验证可用

4. **腾讯 PE 接口没有历史数据**
   → 回测需要历史 PE 时, 从季度 EPS + 历史股价推算
