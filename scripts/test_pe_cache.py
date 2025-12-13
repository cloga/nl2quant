"""测试PE缓存系统功能"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.data.pe_cache import PECache


def test_cache_metadata():
    """测试缓存元数据"""
    print("=" * 60)
    print("测试缓存元数据")
    print("=" * 60)
    
    cache = PECache()
    metadata = cache.get_metadata()
    
    print(f"最后更新: {metadata.get('last_update', '未更新')}")
    print(f"总记录数: {metadata.get('total_stocks', 0)}")
    print(f"缓存版本: {metadata.get('cache_version', 'unknown')}")
    print(f"缓存新鲜: {cache.is_cache_fresh(max_age_days=1)}")
    print()


def test_cache_data():
    """测试缓存数据加载"""
    print("=" * 60)
    print("测试缓存数据加载")
    print("=" * 60)
    
    cache = PECache()
    cache_data = cache.load_cache()
    
    print(f"缓存记录数: {len(cache_data)}")
    
    if cache_data:
        # 显示前3条
        print("\n前3条记录：")
        for i, (ts_code, data) in enumerate(list(cache_data.items())[:3]):
            print(f"\n{i+1}. {ts_code}")
            print(f"   交易日期: {data.get('trade_date')}")
            print(f"   收盘价: {data.get('close_price')}")
            print(f"   市值: {data.get('market_cap')} 亿元")
            print(f"   静态PE: {data.get('static_pe')}")
            print(f"   TTM PE: {data.get('ttm_pe')}")
            print(f"   预测PE: {data.get('forecast_pe_median')}")
    else:
        print("缓存为空")
    
    print()


def test_cache_pe_ratios():
    """测试PE对象获取"""
    print("=" * 60)
    print("测试PE对象获取")
    print("=" * 60)
    
    cache = PECache()
    
    # 随机获取一只股票
    cache_data = cache.load_cache()
    if not cache_data:
        print("缓存为空，无法测试")
        return
    
    ts_code = list(cache_data.keys())[0]
    pe_ratios = cache.get_cached_pe_ratios(ts_code)
    
    if pe_ratios:
        print(pe_ratios)
    else:
        print(f"未找到 {ts_code} 的数据")
    
    print()


def main():
    print("\n" + "=" * 60)
    print("PE缓存系统测试")
    print("=" * 60 + "\n")
    
    try:
        test_cache_metadata()
        test_cache_data()
        test_cache_pe_ratios()
        
        print("=" * 60)
        print("✅ 所有测试完成")
        print("=" * 60)
    
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
