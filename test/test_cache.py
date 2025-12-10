"""测试 Tushare 数据缓存功能"""
import time
from app.dca_backtest_engine import DCABacktestEngine

def test_cache():
    """测试缓存是否正常工作"""
    print("=" * 60)
    print("测试 Tushare 数据缓存")
    print("=" * 60)
    
    # 创建引擎实例
    engine1 = DCABacktestEngine()
    
    # 第一次查询（应该是 cache miss）
    print("\n第一次查询 000922...")
    start = time.time()
    result1 = engine1.fetch_etf_close("000922", "20150101", "20251209")
    elapsed1 = time.time() - start
    print(f"  耗时: {elapsed1:.3f}s")
    print(f"  数据长度: {len(result1) if result1 is not None else 0}")
    print(f"  缓存命中: {engine1.last_price_cache_hit}")
    print(f"  价格缓存大小: {len(engine1.PRICE_CACHE)}")
    
    # 第二次查询相同的数据（应该是 cache hit）
    print("\n第二次查询 000922（相同参数）...")
    start = time.time()
    result2 = engine1.fetch_etf_close("000922", "20150101", "20251209")
    elapsed2 = time.time() - start
    print(f"  耗时: {elapsed2:.3f}s")
    print(f"  数据长度: {len(result2) if result2 is not None else 0}")
    print(f"  缓存命中: {engine1.last_price_cache_hit}")
    print(f"  价格缓存大小: {len(engine1.PRICE_CACHE)}")
    
    # 创建新的引擎实例测试类级别缓存
    print("\n创建新的引擎实例...")
    engine2 = DCABacktestEngine()
    print(f"  新实例的缓存大小: {len(engine2.PRICE_CACHE)}")
    
    # 用新实例查询（应该仍然 hit 缓存）
    print("\n用新实例查询 000922（相同参数）...")
    start = time.time()
    result3 = engine2.fetch_etf_close("000922", "20150101", "20251209")
    elapsed3 = time.time() - start
    print(f"  耗时: {elapsed3:.3f}s")
    print(f"  数据长度: {len(result3) if result3 is not None else 0}")
    print(f"  缓存命中: {engine2.last_price_cache_hit}")
    
    # 测试带后缀的查询
    print("\n查询 000922.SH（带后缀）...")
    start = time.time()
    result4 = engine2.fetch_etf_close("000922.SH", "20150101", "20251209")
    elapsed4 = time.time() - start
    print(f"  耗时: {elapsed4:.3f}s")
    print(f"  数据长度: {len(result4) if result4 is not None else 0}")
    print(f"  缓存命中: {engine2.last_price_cache_hit}")
    
    # 显示所有缓存键
    print("\n当前缓存中的所有键:")
    for key in engine2.PRICE_CACHE.keys():
        print(f"  {key}")
    
    # 测试估值数据缓存
    print("\n" + "=" * 60)
    print("测试估值数据缓存")
    print("=" * 60)
    
    print("\n第一次查询估值数据...")
    start = time.time()
    val1 = engine1.fetch_valuation_data("000922", "20150101", "20251209")
    elapsed_val1 = time.time() - start
    print(f"  耗时: {elapsed_val1:.3f}s")
    print(f"  数据长度: {len(val1) if val1 is not None else 0}")
    print(f"  缓存命中: {engine1.last_valuation_cache_hit}")
    print(f"  估值缓存大小: {len(engine1.VALUATION_CACHE)}")
    
    print("\n第二次查询估值数据（相同参数）...")
    start = time.time()
    val2 = engine1.fetch_valuation_data("000922", "20150101", "20251209")
    elapsed_val2 = time.time() - start
    print(f"  耗时: {elapsed_val2:.3f}s")
    print(f"  数据长度: {len(val2) if val2 is not None else 0}")
    print(f"  缓存命中: {engine1.last_valuation_cache_hit}")
    
    print("\n当前估值缓存中的所有键:")
    for key in engine1.VALUATION_CACHE.keys():
        print(f"  {key}")
    
    # 总结
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    print(f"第一次查询: {elapsed1:.3f}s (无缓存)")
    print(f"第二次查询: {elapsed2:.3f}s (有缓存)")
    print(f"加速比: {elapsed1 / elapsed2:.1f}x" if elapsed2 > 0 else "N/A")
    print(f"估值第一次: {elapsed_val1:.3f}s (无缓存)")
    print(f"估值第二次: {elapsed_val2:.3f}s (有缓存)")
    print(f"估值加速比: {elapsed_val1 / elapsed_val2:.1f}x" if elapsed_val2 > 0 else "N/A")

if __name__ == "__main__":
    test_cache()
