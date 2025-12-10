#!/usr/bin/env python
"""
配对交易筛选器 - 快速测试脚本
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pairs_screener import PairsScreener


def test_basic_workflow():
    """测试基础工作流"""
    print("\n" + "="*70)
    print("测试 1: 基础工作流（使用5只银行股，180天数据）")
    print("="*70)
    
    # 使用5只主要银行股
    codes = ['601398', '601939', '601288', '601166', '601328']
    
    # 设置日期
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    print(f"\n[DATA] 测试参数:")
    print(f"  - 股票: {', '.join(codes)} (共{len(codes)}只)")
    print(f"  - 日期: {start_str} ~ {end_str}")
    print(f"  - PCA成分: 15")
    print(f"  - DBSCAN eps: 0.5")
    
    try:
        screener = PairsScreener(start_str, end_str)
        results = screener.run(codes, eps=0.5, n_components=15)
        
        pairs_df = results['pairs']
        labels = results['labels']
        
        print(f"\n[PASS] 测试成功！")
        print(f"  - 找到 {len(pairs_df)} 对协整配对")
        print(f"  - 聚类数: {len(set(labels)) - (1 if -1 in labels else 0)}")
        print(f"  - 噪音点: {list(labels).count(-1)}")
        
        if len(pairs_df) > 0:
            print(f"\n🏆 Top 3 配对:")
            for i, (idx, row) in enumerate(pairs_df.head(3).iterrows(), 1):
                print(f"  {i}. {row['stock_a']} ↔️ {row['stock_b']}")
                print(f"     相关系数: {row['correlation']:.4f}, P值: {row['coint_pvalue']:.6f}")
        
        return True
    
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("\n" + "="*70)
    print("测试 2: 参数敏感性（不同eps值的影响）")
    print("="*70)
    
    codes = ['601398', '601939', '601288', '601166', '601328']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    eps_values = [0.3, 0.5, 0.7]
    
    print(f"\n测试不同的 eps 值对聚类的影响:")
    
    try:
        screener = PairsScreener(start_str, end_str)
        
        for eps in eps_values:
            print(f"\n  eps={eps}:", end=" ")
            results = screener.run(codes, eps=eps, n_components=15)
            
            labels = results['labels']
            pairs_count = len(results['pairs'])
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"簇数={n_clusters}, 配对={pairs_count}")
        
        print(f"\n[PASS] 参数敏感性测试完成")
        return True
    
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {str(e)}")
        return False


def test_data_integrity():
    """测试数据完整性"""
    print("\n" + "="*70)
    print("测试 3: 数据完整性")
    print("="*70)
    
    codes = ['601398', '601939', '601288']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")
    
    print(f"\n检查数据质量:")
    
    try:
        screener = PairsScreener(start_str, end_str)
        
        # 获取数据
        print(f"  1. 获取股票数据...", end=" ")
        price_df = screener.fetch_stock_data(codes)
        print(f"[OK] ({price_df.shape[0]}行 × {price_df.shape[1]}列)")
        
        # 计算收益率
        print(f"  2. 计算收益率...", end=" ")
        returns_df = screener.compute_returns(price_df)
        print(f"[OK] ({returns_df.shape[0]}行 × {returns_df.shape[1]}列)")
        
        # 检查缺失值
        missing_pct = returns_df.isna().sum().sum() / (returns_df.shape[0] * returns_df.shape[1]) * 100
        print(f"  3. 缺失值检查: {missing_pct:.2f}%", end=" ")
        print("[OK]" if missing_pct < 1 else "⚠️")
        
        # PCA
        print(f"  4. PCA降维...", end=" ")
        X_pca, pca = screener.perform_pca(returns_df, n_components=10)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"[OK] (解释方差: {explained_var:.1%})")
        
        # DBSCAN
        print(f"  5. DBSCAN聚类...", end=" ")
        labels = screener.perform_dbscan(X_pca, eps=0.5)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"[OK] ({n_clusters}个簇)")
        
        print(f"\n[PASS] 数据完整性测试通过")
        return True
    
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("A股配对交易筛选器 - 功能测试")
    print("="*70)
    
    results = []
    
    # 运行测试
    results.append(("数据完整性", test_data_integrity()))
    results.append(("基础工作流", test_basic_workflow()))
    results.append(("参数敏感性", test_parameter_sensitivity()))
    
    # 总结
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    
    for test_name, passed in results:
        status = "[PASS] 通过" if passed else "[FAIL] 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("[PASS] 所有测试通过！程序可正常使用。")
    else:
        print("[FAIL] 部分测试失败，请检查错误信息。")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())


