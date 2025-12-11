#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相关性分析工具演示脚本
演示不同类型的相关性分析场景
"""

import sys
import os
from pathlib import Path

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.correlation_analyzer import CorrelationAnalyzer


def demo_pairs_trading():
    """演示：配对交易筛选"""
    print("\n" + "="*70)
    print("演示1: 配对交易筛选 (Pairs Trading)")
    print("="*70)
    print("目标: 找两个银行股，检查是否适合配对交易")
    
    analyzer = CorrelationAnalyzer()
    
    # 分析招商银行 vs 兴业银行
    print("\n分析配对: 招商银行(600036) vs 兴业银行(601166)")
    results = analyzer.cointegration_test('600036.SH', '601166.SH')
    
    print(f"\n协整性: {results['engle_granger']}")
    print(f"平稳性: {results['adf_spread']}")
    
    if results['engle_granger']['cointegrated']:
        print("\n✓ 存在协整关系，适合配对交易")
        
        spread_info = analyzer.spread_analysis('600036.SH', '601166.SH')
        print(f"\n价差分析:")
        print(f"  当前Z-Score: {spread_info['current_zscore']:.4f}")
        print(f"  极端事件频率: {spread_info['extreme_events']['percentage']:.2f}%")
        
        if abs(spread_info['current_zscore']) > 2:
            print(f"  ⚠️  当前价差处于极端水平，可考虑入场")
        else:
            print(f"  📊 当前价差处于正常水平")
    else:
        print("\n✗ 不存在协整关系，不适合配对交易")
        print("  → 建议转向其他标的对")


def demo_hedge_portfolio():
    """演示：投资组合对冲"""
    print("\n" + "="*70)
    print("演示2: 投资组合对冲 (Portfolio Hedging)")
    print("="*70)
    print("目标: 评估一只股票与一只债券基金的对冲效果")
    
    analyzer = CorrelationAnalyzer()
    
    # 分析平安银行 vs 50ETF
    print("\n分析对冲: 平安银行(000001.SZ) vs 50ETF(510050.SH)")
    
    # 线性相关分析
    linear = analyzer.pearson_correlation('000001.SZ', '510050.SH')
    print(f"\nPearson相关系数: {linear['pearson']['corr']:.4f}")
    print(f"强度: {linear['pearson']['strength']}")
    
    if linear['pearson']['corr'] < 0.3:
        print("✓ 相关性较弱，可用于对冲")
    
    # 尾部依赖分析
    tail = analyzer.tail_dependence('000001.SZ', '510050.SH')
    print(f"\n危机中的联动:")
    print(f"  左尾依赖 (暴跌时): {tail['left_tail_dependence']['probability']:.1%}")
    print(f"  风险评估: {tail['risk_assessment']}")


def demo_market_analysis():
    """演示：市场分析 - ETF联动"""
    print("\n" + "="*70)
    print("演示3: 市场分析 - ETF联动分析")
    print("="*70)
    print("目标: 分析黄金现货 vs 黄金股的最近联动关系")
    
    analyzer = CorrelationAnalyzer()
    
    # 分析黄金ETF
        print("\n分析关系: 黄金ETF(518880.SH) vs 黄金股(159562.SZ)")
    
    # 滚动相关分析
    rolling = analyzer.rolling_correlation('518880.SH', '159562.SZ', window=30)
    print(f"\n滚动相关系数分析 (30天窗口):")
    print(f"  当前: {rolling['current_correlation']:.4f}")
    print(f"  平均: {rolling['mean_correlation']:.4f}")
    print(f"  波动率: {rolling['volatility']:.4f}")
    print(f"  范围: [{rolling['min_correlation']:.4f}, {rolling['max_correlation']:.4f}]")
    
    if rolling['volatility'] < 0.15:
        print("✓ 关系稳定，长期有效")
    elif rolling['volatility'] > 0.3:
        print("⚠️  关系不稳定，需要定期监控")


def demo_beta_analysis():
    """演示：Beta系数分析"""
    print("\n" + "="*70)
    print("演示4: 风险管理 - Beta系数分析")
    print("="*70)
    print("目标: 评估某只股票相对市场的敏感度")
    
    analyzer = CorrelationAnalyzer()
    
    # 分析贵州茅台 vs 上证指数
    print("\n分析: 贵州茅台(600519) 相对 上证指数(000001.SH)的敏感度")
    
    beta = analyzer.beta_coefficient('000001.SH', '600519.SH')
    print(f"\nBeta系数: {beta['beta']:.4f}")
    print(f"Alpha (超额收益): {beta['alpha']:.6f}")
    print(f"R² (拟合度): {beta['r_squared']:.4f}")
    
    if beta['beta'] > 1.5:
        print("⚠️  高敏感度 - 贵州茅台的波动幅度远大于指数，风险较高")
    elif beta['beta'] < 0.5:
        print("✓ 低敏感度 - 贵州茅台的波动幅度小于指数，相对稳定")
    else:
        print("◈ 中等敏感度 - 贵州茅台与指数波动基本同步")


def demo_comprehensive():
    """演示：综合分析"""
    print("\n" + "="*70)
    print("演示5: 综合分析 - 完整评估")
    print("="*70)
    print("目标: 对两只股票进行6维度完整分析")
    
    analyzer = CorrelationAnalyzer()
    
    # 综合分析
    print("\n对以下标的对进行完整分析:")
    print("  标的1: 中国平安 (000001.SZ)")
    print("  标的2: 招商银行 (600036.SH)")
    
    results = analyzer.comprehensive_analysis('000001.SZ', '600036.SH')


def main():
    """运行所有演示"""
    print("\n" + "="*70)
    print("相关性分析工具 - 演示脚本")
    print("="*70)
    print("\n本脚本演示5个常见的应用场景:")
    print("1. 配对交易筛选")
    print("2. 投资组合对冲")
    print("3. ETF市场分析")
    print("4. Beta系数风险管理")
    print("5. 综合6维度分析")
    
    try:
        # 演示1: 配对交易
        demo_pairs_trading()
        
        # 演示2: 组合对冲
        demo_hedge_portfolio()
        
        # 演示3: 市场分析
        demo_market_analysis()
        
        # 演示4: Beta分析
        demo_beta_analysis()
        
        # 演示5: 综合分析
        # demo_comprehensive()  # 注释掉，因为耗时较长
        
        print("\n" + "="*70)
        print("演示完成")
        print("="*70)
        print("\n更多用法，请查看:")
        print("  - doc/CORRELATION_QUICK_START.md (快速开始)")
        print("  - doc/CORRELATION_ANALYZER_GUIDE.md (详细指南)")
        print("  - 执行: python scripts/correlation_cli.py --help")
        
    except Exception as e:
        print(f"\n✗ 演示执行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
