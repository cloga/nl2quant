#!/usr/bin/env python
"""
配对交易筛选器 - 缓存示例
演示如何使用数据缓存加快重复运行
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pairs_screener import PairsScreener
from app.data_cache import DataCache


def example_with_cache():
    """演示缓存使用"""
    print("\n" + "="*70)
    print("A股配对交易筛选 - 缓存示例")
    print("="*70)
    
    # 设置日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    # 使用的股票代码
    codes = ['601398', '601939', '601288', '601166', '601328']
    
    print(f"\n参数设置:")
    print(f"  日期: {start_str} ~ {end_str}")
    print(f"  股票: {', '.join(codes)} (共{len(codes)}只)")
    
    # 第一次运行 - 会下载数据并缓存
    print(f"\n[第一次运行] 下载数据并缓存...")
    screener1 = PairsScreener(start_str, end_str)
    
    import time
    start_time = time.time()
    results1 = screener1.run(codes, eps=0.5, n_components=15)
    elapsed1 = time.time() - start_time
    
    print(f"\n耗时: {elapsed1:.2f}秒")
    pairs1 = results1['pairs']
    print(f"找到 {len(pairs1)} 对协整配对")
    
    # 显示缓存信息
    print(f"\n缓存状态:")
    screener1.cache.print_cache_info()
    
    # 第二次运行 - 从缓存读取，应该快很多
    print(f"\n[第二次运行] 从缓存读取数据...")
    screener2 = PairsScreener(start_str, end_str)
    
    start_time = time.time()
    results2 = screener2.run(codes, eps=0.5, n_components=15)
    elapsed2 = time.time() - start_time
    
    print(f"\n耗时: {elapsed2:.2f}秒")
    pairs2 = results2['pairs']
    print(f"找到 {len(pairs2)} 对协整配对")
    
    # 对比加速效果
    print(f"\n性能对比:")
    print(f"  首次运行:  {elapsed1:.2f}秒")
    print(f"  缓存运行:  {elapsed2:.2f}秒")
    if elapsed1 > 0:
        speedup = elapsed1 / elapsed2
        print(f"  加速倍数:  {speedup:.1f}x")
    
    # 验证结果一致性
    if len(pairs1) == len(pairs2):
        print(f"\n[OK] 结果验证: 两次运行的结果完全一致 ({len(pairs1)}对配对)")
    else:
        print(f"\n[WARN] 结果不一致: {len(pairs1)} vs {len(pairs2)}")


def manage_cache():
    """缓存管理示例"""
    print("\n" + "="*70)
    print("缓存管理")
    print("="*70)
    
    cache = DataCache()
    
    print("\n当前缓存信息:")
    cache.print_cache_info()
    
    print("\n缓存管理操作:")
    print("  1. 清除过期缓存 (> 24小时)")
    cache.clear_expired(max_age_hours=24)
    
    print("\n更新后的缓存信息:")
    cache.print_cache_info()


def main():
    """主程序"""
    import argparse
    
    parser = argparse.ArgumentParser(description="配对交易筛选器 - 缓存演示")
    parser.add_argument('--example', action='store_true', help='运行缓存对比示例')
    parser.add_argument('--manage', action='store_true', help='进行缓存管理')
    parser.add_argument('--clear', action='store_true', help='清除所有缓存')
    parser.add_argument('--info', action='store_true', help='显示缓存信息')
    
    args = parser.parse_args()
    
    if args.clear:
        cache = DataCache()
        cache.clear_all()
    elif args.info:
        cache = DataCache()
        cache.print_cache_info()
    elif args.manage:
        manage_cache()
    else:
        # 默认运行示例
        example_with_cache()


if __name__ == '__main__':
    main()
