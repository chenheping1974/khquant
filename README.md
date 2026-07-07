# khquant v3.0 — A股量化选股系统

BlackRock 4/3/2/1 因子架构，每日自动推理，输出 Top30 选股信号。

## 因子体系

```
基本面 40%                    价量 30%
  ROE          20%              E/P         8%
  杠杆率(取反)  10%              B/P         5%
  F-Score      10%              Size(取反)   4%
                                短期反转     8%
另类 20%                        动量12m1m   5%
  分析师买入   5%
  EPS增速      5%             其他 10%
  公告情绪     5%               战略行业    5%
  机构调研     5%               行业中性化  5%

总计: 40 + 30 + 20 + 10 = 100%
```

## 数据源

| 数据 | 来源 | 频率 | 覆盖 |
|------|------|------|------|
| K线 (OHLCV) | Sina API | 日 | 4944只 |
| PE/PB/市值 | 腾讯财经 | 日(实时) | 4944只 |
| 财报(ROE/负债率/毛利率等) | akshare | 季 | 4944只 |
| 股东户数 | 东方财富 | 季 | 4944只 |
| 分析师评级/预测 | akshare | 日(实时) | 2785只 |
| 机构调研 | akshare | 日(实时) | 4682只 |
| 公告情绪 | 东方财富 | 日 | 3398只 |
| 行业分类 | 东方财富 F10 | 静态 | 288个行业 |

## 自动化

```
Daily (每晚 20:00 BJT)
  K线增量拉取 → 因子计算 → Top30 → results/latest.json

Monthly (每月1号)
  股东户数全量更新 + 公告数据全量更新

Quarterly (季报期每周六: 5-6月/9月/11-12月)
  财报增量补缺
```

## 输出格式

`results/latest.json`:

```json
{
  "date": "2026-07-06",
  "updated": "2026-07-06 20:30:00",
  "method": "BlackRock 4/3/2/1",
  "data_freshness": {
    "财报": "✅ 财报: 5天前",
    "股东户数": "✅ 股东户数: 3天前"
  },
  "top30": [
    {
      "rank": 1,
      "symbol": "000338",
      "name": "潍柴动力",
      "industry": "汽车零部件",
      "score": 2.02,
      "factors": {
        "value_ep": 0.046,
        "value_bp": 0.391,
        "momentum_12m1m": 1.21,
        "reversal_1m": 0.128,
        "quality_roe": 0.033,
        "quality_leverage": -0.642,
        "quality_fscore": 0.33,
        "alt_visits": 34,
        "strategic_tier": 2
      }
    }
  ]
}
```

## 外部引用

```python
import requests
data = requests.get(
    "https://raw.githubusercontent.com/chenheping1974/khquant/main/results/latest.json"
).json()
for s in data["top30"]:
    print(f"{s['rank']}. {s['name']}({s['symbol']}) score={s['score']}")
```

## 项目结构

```
khquant/
├── src/
│   ├── features/         # 因子计算引擎
│   │   ├── factors.py    # v2.0 因子(已废弃)
│   │   ├── factors_v3.py # v3.0 四大因子
│   │   └── fundamental.py # 基本面因子
│   ├── data/             # 数据获取
│   │   ├── storage.py    # Parquet读写
│   │   ├── fetcher.py    # Sina K线
│   │   ├── tencent_fetcher.py  # 腾讯PE/PB
│   │   ├── baostock_fetcher.py # 历史PE/PB
│   │   └── baostock_financial.py # 历史财报
│   └── backtest/         # 回测
│       ├── verify_v3_blackrock.py # BlackRock框架回测
│       └── verify_v3_full.py      # 全子指标回测
├── scripts/
│   ├── daily_inference.py    # 日频推理(主入口)
│   ├── fetch_incremental.py  # K线增量拉取
│   ├── download_fin_q1.py    # 季报下载(断点续传)
│   └── download_announce.py  # 公告下载(断点续传)
├── .github/workflows/
│   ├── daily.yml       # 每日推理
│   ├── monthly.yml     # 月度数据更新
│   └── quarterly.yml   # 季度财报更新
├── docs/
│   └── FACTOR_V3_DESIGN.md  # 因子设计文档
└── results/
    └── latest.json      # 最新推理结果(供外部HTTP访问)
```

## 回测结果

```
2023-07 ~ 2026-07, 3年, 全量4943只随机抽样, 周度调仓, 扣交易成本(0.18%)

500只:     年化 +10.6%  超额 +1.4%
1000只:    年化 +11.2%  超额 +2.5%
2000只:    年化 +10.4%  超额 +1.7%
3000只:    年化 +10.4%  超额 +1.8%
全量4943只: 年化 +10.2%  超额 +1.6%  ← 最客观

等权基准: ~+8.6% 年化

所有数据均来自真实API，无前视偏差，无近似值。
```

## 许可

MIT
