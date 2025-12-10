#!/usr/bin/env python
"""
用缓存中已有的ETF数据来演示聚类效果
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import json

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pairs_screener import PairsScreener
from app.data_cache import DataCache


def get_cached_etf_codes():
    """从缓存元数据中提取所有唯一的ETF代码"""
    cache = DataCache()
    metadata_path = Path(cache.cache_dir) / "cache_metadata.json"
    
    if not metadata_path.exists():
        print("未找到缓存元数据")
        return []
    
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"读取缓存元数据失败: {e}")
        return []
    
    # 缓存元数据是 {hash: {code, start_date, end_date, ...}} 的结构
    codes_set = set()
    for hash_key, entry in metadata.items():
        if isinstance(entry, dict) and 'code' in entry:
            codes_set.add(entry['code'])
    
    codes = sorted(list(codes_set))
    return codes


def main():
    # 获取缓存中的ETF代码
    codes = get_cached_etf_codes()
    
    if not codes:
        print("缓存中没有数据")
        return
    
    print(f"从缓存中找到 {len(codes)} 只ETF")
    print(f"代码: {codes[:10]}{'...' if len(codes) > 10 else ''}\n")
    
    # 使用缓存中最新的日期范围（250个交易日从20241202到20251210）
    start_date = "20241202"
    end_date = "20251210"
    
    print(f"日期范围: {start_date} ~ {end_date}")
    print(f"PCA参数: n_components=10")
    print(f"DBSCAN参数: eps=1.2")
    print(f"相关性阈值: min_corr=0.8")
    print(f"p值阈值: pvalue=0.1\n")
    
    # 运行筛选
    screener = PairsScreener(start_date, end_date)
    results = screener.run(
        codes,
        eps=1.2,
        n_components=10,
        min_corr=0.8,
        pvalue_threshold=0.1,
    )
    
    # 处理结果
    pairs_df = results['pairs']
    
    print("\n" + "="*60)
    print("筛选结果")
    print("="*60)
    
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('correlation', ascending=False)
        print(f"\n找到 {len(pairs_df)} 对协整配对:\n")
        print(pairs_df.to_string(index=False))
        
        # 显示Top10
        print(f"\nTop 10 最强配对:")
        for idx, (i, row) in enumerate(pairs_df.head(10).iterrows(), 1):
            print(f"{idx}. {row['stock_a']} ↔ {row['stock_b']}")
            print(f"   相关系数: {row['correlation']:.4f}, 协整p值: {row['coint_pvalue']:.6f}")
    else:
        print("未找到协整配对")


if __name__ == '__main__':
    main()
