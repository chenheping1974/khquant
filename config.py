"""
khquant — 全局配置
A股 + 现货黄金白银 量化选股与择时系统
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── 项目路径 ───────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_REPO_PATH = Path(os.getenv("KHQUANT_DATA_PATH", str(ROOT.parent / "khquant-data")))
DATA_DIR = DATA_REPO_PATH                    # 数据仓库路径 (独立于代码仓库)
A_STOCK_DIR = DATA_DIR / "a_stock"
GOLD_SILVER_DIR = DATA_DIR / "gold_silver"
MACRO_DIR = DATA_DIR / "macro"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

for d in [DATA_DIR, A_STOCK_DIR, GOLD_SILVER_DIR, MACRO_DIR, MODEL_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── 存储后端: "local" | "github" | "r2" ───────────────
STORAGE_BACKEND = os.getenv("KHQUANT_STORAGE", "local")

# 数据仓库 (独立于代码仓库, 供多项目共享)
DATA_REPO = {
    # 本地路径 (开发/回测用, git clone khquant-data 到同级目录)
    "local_path": os.getenv("KHQUANT_DATA_PATH", str(ROOT.parent / "khquant-data")),
    # GitHub 仓库 (HTTP读取用, 其他项目不需要clone)
    "owner": os.getenv("DATA_REPO_OWNER", ""),
    "name": os.getenv("DATA_REPO_NAME", "khquant-data"),
    "branch": os.getenv("DATA_REPO_BRANCH", "main"),
    "cache_dir": str(ROOT / ".data_cache"),   # HTTP读取的本地缓存
}

# Cloudflare R2 配置 (S3-compatible, 备用)
R2_CONFIG = {
    "account_id": os.getenv("R2_ACCOUNT_ID", ""),
    "access_key": os.getenv("R2_ACCESS_KEY", ""),
    "secret_key": os.getenv("R2_SECRET_KEY", ""),
    "bucket": os.getenv("R2_BUCKET", "khquant-data"),
    "endpoint_template": "https://{account_id}.r2.cloudflarestorage.com",
    "cache_dir": str(ROOT / ".r2_cache"),
}

# ── A股配置 ────────────────────────────────────────────
# 垃圾股过滤阈值
GARBAGE_FILTER = {
    "min_listing_days": 60,           # 上市天数
    "min_daily_amount": 30_000_000,   # 近20日均成交额 (3000万)
    "min_price": 2.0,                 # 最低股价 (元)
    "min_market_cap": 1_000_000_000,  # 最低总市值 (10亿)
    "exclude_st": True,               # 排除ST/*ST
    "exclude_suspended": True,        # 排除停牌
    "exclude_new_stocks": True,       # 排除次新股
}

# 指数成分股 (用于行业分布报告)
BENCHMARKS = {
    "sh000300": "沪深300",
    "sh000905": "中证500",
    "sh000852": "中证1000",
}

# ── 黄金白银配置 ───────────────────────────────────────
GOLD_SILVER_SYMBOLS = {
    # 上海金交所 (akshare)
    "AU99.99": "au99",       # 黄金现货
    "Ag(T+D)": "agtd",       # 白银T+D
    # 伦敦金 (yfinance)
    "XAUUSD": "XAUUSD=X",    # 伦敦金
    "XAGUSD": "XAGUSD=X",    # 伦敦银
}

# 宏观指标 (yfinance)
MACRO_SYMBOLS = {
    "DXY": "DX-Y.NYB",           # 美元指数
    "TIPS": "TIP",               # TIPS ETF (代理实际利率)
    "VIX": "^VIX",               # 恐慌指数
    "US10Y": "^TNX",             # 美国10年期国债收益率
    "USDCNY": "CNY=X",           # 美元/人民币
    "GLD": "GLD",                # SPDR黄金ETF
    "SLV": "SLV",                # iShares白银ETF
}

# ── LightGBM 模型标签 ──────────────────────────────────
LABEL_HORIZONS = [5, 10, 20]      # 未来N日收益率
LABEL_WEIGHTS = [0.2, 0.5, 0.3]   # 短期:中期:长期权重

# ── 特征工程 ──────────────────────────────────────────
# A股技术指标参数
TECH_PARAMS = {
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "boll_period": 20,
    "boll_std": 2,
    "atr_period": 14,
    "ma_periods": [5, 10, 20, 60],
    "vol_periods": [5, 20],
}

# ── 模型训练 ──────────────────────────────────────────
MODEL_PARAMS = {
    "objective": "lambdarank",     # 排序学习
    "metric": "ndcg",
    "boosting_type": "gbdt",
    "num_leaves": 63,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "num_threads": -1,
    "seed": 42,
}

# ── GitHub Actions 定时 ─────────────────────────────────
# 北京时间 16:00 = UTC 08:00
SCHEDULE_CRON = "0 8 * * 1-5"     # 周一至周五，北京时间16:00

# ── 数据保留天数 ───────────────────────────────────────
DATA_RETENTION_DAYS = 365 * 5      # A股保留5年历史
MACRO_RETENTION_DAYS = 365 * 10    # 宏观保留10年
