#!/usr/bin/env python3
"""
将本地 data/ 目录迁移到 Cloudflare R2

用法:
  1. 复制 .env.example → .env, 填入 R2 凭证
  2. python scripts/upload_to_r2.py              # 全部市场
  3. python scripts/upload_to_r2.py --dry-run    # 预览不上传
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import boto3
import pandas as pd
from botocore.config import Config
from config import R2_CONFIG, DATA_DIR

# ── 连接 R2 ────────────────────────────────────────────
endpoint = R2_CONFIG["endpoint_template"].format(account_id=R2_CONFIG["account_id"])
s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=R2_CONFIG["access_key"],
    aws_secret_access_key=R2_CONFIG["secret_key"],
    config=Config(signature_version="s3v4"),
)
bucket = R2_CONFIG["bucket"]

# 确保 bucket 存在
try:
    s3.head_bucket(Bucket=bucket)
    print(f"✅ Bucket '{bucket}' 已存在")
except Exception:
    s3.create_bucket(Bucket=bucket)
    print(f"✅ 创建 Bucket '{bucket}'")

# ── 上传 ────────────────────────────────────────────────
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()

for market_dir in sorted(DATA_DIR.iterdir()):
    if not market_dir.is_dir():
        continue
    market = market_dir.name
    parquets = sorted(market_dir.glob("year=*/month=*/data.parquet"))
    if not parquets:
        continue

    print(f"\n📦 {market}: {len(parquets)} 个分区")

    for fp in parquets:
        # R2 key: market/year=YYYY/month=MM/data.parquet
        rel = fp.relative_to(DATA_DIR)
        key = str(rel)  # e.g. "a_stock/year=2023/month=01/data.parquet"
        size_kb = fp.stat().st_size / 1024

        if args.dry_run:
            print(f"  [DRY-RUN] {key} ({size_kb:.0f} KB)")
            continue

        try:
            s3.upload_file(str(fp), bucket, key)
            print(f"  ✅ {key} ({size_kb:.0f} KB)")
        except Exception as e:
            print(f"  ❌ {key}: {e}")
            sys.exit(1)

print("\n🎉 迁移完成!")
print(f"   设置 KHQUANT_STORAGE=r2 后运行: python main.py --mode backtest")
