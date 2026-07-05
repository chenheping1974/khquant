# khquant 因子体系 v3.0 — MSCI/BlackRock/AQR 标准

## 架构总览

```
16个散装因子  →  4个核心因子 × 多指标合成  +  行业中性化  +  风险约束
  (v2.0)                         (v3.0)
```

---

## 四个核心因子

### 1. Value (价值) — 25%

| 子指标 | 计算 | 数据源 | 参考 |
|--------|------|--------|------|
| E/P | 1/PE | 腾讯财经 PE_TTM | LSY 2019, BlackRock |
| B/P | 1/PB | 腾讯财经 PB | MSCI Barra, Fama-French |
| CF/P | 经营现金流/市值 | akshare 财报 (+总股本×股价) | BlackRock (20%权重) |

合成: Z-score(winsorized) → 行业内等权平均

### 2. Momentum (动量) — 25%

| 子指标 | 计算 | 数据源 | 参考 |
|--------|------|--------|------|
| 价格动量 | 12-1月收益 | K线 | Carhart 1997, MSCI |
| 风险调整动量 | 价格动量/252日波动率 | K线 | BlackRock, Barra |
| 盈利动量 | 净利润增速 YoY | akshare 财报 | BlackRock (50%权重!) |

合成: Z-score(winsorized) → 行业内等权平均

### 3. Quality (质量) — 40%

| 子指标 | 计算 | 数据源 | 参考 |
|--------|------|--------|------|
| ROE | 净资产收益率 | akshare 财报 | FF5, BlackRock |
| 毛利率 | (收入-成本)/收入 | akshare 财报 | Novy-Marx 2013 |
| 应计利润 | (净利润-经营CF)/总资产 | akshare 财报 | Sloan 1996, BlackRock (20%) |
| 杠杆率 | 资产负债率(取反) | akshare 财报 | MSCI Barra, AQR QMJ |
| 盈利稳定性 | 过去8季度ROE标准差(取反) | akshare 财报 | MSCI CNE6 |

合成: Z-score(winsorized) → 行业内等权平均

### 4. Low Volatility (低波动) — 10%

| 子指标 | 计算 | 数据源 | 参考 |
|--------|------|--------|------|
| 历史波动率 | 252日收益标准差(取反) | K线 | Ang et al. 2006 |
| Beta | 252日滚动Beta(取反) | K线 | BlackRock, Barra |
| 残差波动率 | 剔除市场收益后的波动(取反) | K线 | MSCI Barra CNE6 |

合成: Z-score(winsorized) → 行业内等权平均

---

## 数据处理流程

```
1. 原始数据
   ├── K线 (Sina): OHLCV
   ├── 估值 (腾讯财经): PE, PB, 总股本
   └── 财报 (akshare): ROE, 毛利率, 经营CF, 负债率, 净利润增速

2. 数据清洗
   ├── 剔除: ST, 停牌, 上市<60天, 市值最小30%
   ├── Winsorize: 3σ 截尾
   └── 缺失值: 行业内中位数填充

3. 行业中性化
   ├── 每只股票在所属行业(申万)内排名
   └── Z-score = (值 - 行业均值) / 行业标准差

4. 因子合成
   ├── 每个核心因子 = 2-5个子指标 Z-score 等权平均
   └── 最终得分 = Value×25% + Momentum×25% + Quality×40% + LowVol×10%

5. 组合构建
   ├── 每日选得分最高的 Top 30
   ├── 单行业 ≤ 5只 (行业分散)
   ├── 单票 ≤ 5% 仓位
   └── 月换手率 ≤ 25% (缓冲机制)
```

---

## 与 v2.0 的差异

| 维度 | v2.0 | v3.0 |
|------|------|------|
| 因子数 | 16 | 4 (每个多指标) |
| 因子独立性 | 未检查 | 行业中性化后自然正交 |
| 数据穿越 | 存在(回测用今日PE) | 修复(逐日滚动计算) |
| 行业控制 | 1个 industry_code | 全部行业中性化 + 组合层约束 |
| 风险控制 | 无 | 换手率+仓位+行业偏离 |
| LightGBM | 必用 | 可选(固定权重已可用) |
| 学术支撑 | 每个因子有论文 | 每个子指标+MSCI/BlackRock框架 |

---

## 实施计划

### Step 1: 数据管线
- 腾讯PE/PB → E/P, B/P (已有)
- akshare财报 → ROE, 毛利率, 经营CF, 负债率, 净利润增速, 盈利稳定性 (已有API)
- K线 → 价格动量, 波动率, Beta (已有)

### Step 2: 因子计算 (src/features/factors_v3.py)
- 4个 compute_*_factor() 函数
- 每个返回 (N_stocks,) 的 Z-score 数组
- 行业中性化函数
- Winsorize + 缺失值填充

### Step 3: 组合构建 (src/backtest/verify.py)
- 加权总分 → 选 Top 30
- 行业约束: 单行业 ≤ 5只
- 换手率缓冲: 与昨日持仓对比, 只换超出阈值部分

### Step 4: 回测验证
- 与 v2.0 对比
- 与等权基准对比
- 分年度查看稳定性
