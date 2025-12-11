#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
相关性分析 CLI 工具
用法: python correlation_cli.py --code1 600036.SH --code2 601166.SH [--output report.md]
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.correlation_analyzer import CorrelationAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='标的相关性综合分析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 分析两只股票
  python correlation_cli.py --code1 600036.SH --code2 601166.SH
  
  # 分析股票与指数
  python correlation_cli.py --code1 000001.SZ --code2 399300.SZ --type1 stock --type2 index
  
  # 分析两只基金
  python correlation_cli.py --code1 510050.SH --code2 510180.SH --type1 fund --type2 fund
  
  # 保存报告到文件
  python correlation_cli.py --code1 600519.SH --code2 000858.SZ --output report.md
  
  # 只做某一维度的分析
  python correlation_cli.py --code1 601398.SH --code2 601939.SH --dimension pearson
        """
    )
    
    parser.add_argument(
        '--code1',
        type=str,
        required=True,
        help='第一个标的代码 (如: 600036.SH, 510050.SH, 399300.SZ)'
    )
    
    parser.add_argument(
        '--code2',
        type=str,
        required=True,
        help='第二个标的代码'
    )
    
    parser.add_argument(
        '--type1',
        type=str,
        choices=['stock', 'fund', 'index', 'auto'],
        default='auto',
        help='第一个标的类型 (默认: auto自动识别)'
    )
    
    parser.add_argument(
        '--type2',
        type=str,
        choices=['stock', 'fund', 'index', 'auto'],
        default='auto',
        help='第二个标的类型 (默认: auto自动识别)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出报告文件路径 (MD格式，可选)'
    )
    
    parser.add_argument(
        '--dimension',
        type=str,
        choices=['all', 'linear', 'cointegration', 'granger', 'rolling', 'tail'],
        default='all',
        help='分析维度 (默认: 全部)'
    )
    
    parser.add_argument(
        '--window',
        type=int,
        default=30,
        help='滚动窗口大小 (天，默认30)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("标的相关性综合分析工具")
    print("="*70)
    print(f"标的1: {args.code1} (类型: {args.type1})")
    print(f"标的2: {args.code2} (类型: {args.type2})")
    print(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")
    
    try:
        analyzer = CorrelationAnalyzer()
        
        if args.dimension == 'all':
            # 综合分析
            results = analyzer.comprehensive_analysis(
                args.code1,
                args.code2,
                report_path=args.output
            )
        
        elif args.dimension == 'linear':
            print("\n[线性相关分析]\n")
            linear_results = analyzer.pearson_correlation(args.code1, args.code2)
            beta_results = analyzer.beta_coefficient(args.code1, args.code2)
            
            print(f"Pearson相关系数: {linear_results['pearson']['corr']:.4f}")
            print(f"  强度: {linear_results['pearson']['strength']}")
            print(f"  P值: {linear_results['pearson']['p_value']:.6f}")
            print(f"\nSpearman秩相关系数: {linear_results['spearman']['corr']:.4f}")
            print(f"  P值: {linear_results['spearman']['p_value']:.6f}")
            print(f"\nBeta系数: {beta_results['beta']:.4f}")
            print(f"  {beta_results['interpretation']}")
        
        elif args.dimension == 'cointegration':
            print("\n[协整与长期均衡分析]\n")
            coint_results = analyzer.cointegration_test(args.code1, args.code2)
            spread_results = analyzer.spread_analysis(args.code1, args.code2)
            
            print(f"Engle-Granger P值: {coint_results['engle_granger']['p_value']:.6f}")
            print(f"协整关系: {'存在' if coint_results['engle_granger']['cointegrated'] else '不存在'}")
            print(f"\nADF检验P值: {coint_results['adf_spread']['p_value']:.6f}")
            print(f"平稳性: {'平稳' if coint_results['adf_spread']['stationary'] else '非平稳'}")
            print(f"\n当前Z-Score: {spread_results['current_zscore']:.4f}")
            print(f"极端事件频率: {spread_results['extreme_events']['percentage']:.2f}%")
        
        elif args.dimension == 'granger':
            print("\n[Granger因果检验]\n")
            granger_lead = analyzer.granger_causality_test(args.code1, args.code2)
            granger_lag = analyzer.granger_causality_test(args.code2, args.code1)
            
            print(f"{args.code1} → {args.code2}:")
            print(f"  {granger_lead['interpretation']}")
            print(f"\n{args.code2} → {args.code1}:")
            print(f"  {granger_lag['interpretation']}")
        
        elif args.dimension == 'rolling':
            print(f"\n[滚动相关系数分析 (窗口={args.window}天)]\n")
            rolling_results = analyzer.rolling_correlation(args.code1, args.code2, window=args.window)
            
            print(f"当前相关系数: {rolling_results['current_correlation']:.4f}")
            print(f"均值: {rolling_results['mean_correlation']:.4f}")
            print(f"标准差: {rolling_results['std_correlation']:.4f}")
            print(f"范围: [{rolling_results['min_correlation']:.4f}, {rolling_results['max_correlation']:.4f}]")
        
        elif args.dimension == 'tail':
            print("\n[极端风险与尾部依赖]\n")
            tail_results = analyzer.tail_dependence(args.code1, args.code2)
            
            print(f"左尾依赖 (暴跌): {tail_results['left_tail_dependence']['probability']:.1%}")
            print(f"右尾依赖 (暴涨): {tail_results['right_tail_dependence']['probability']:.1%}")
            print(f"风险评估: {tail_results['risk_assessment']}")
        
        print("\n" + "="*70)
        print("分析完成")
        print("="*70 + "\n")
    
    except Exception as e:
        print(f"\n✗ 分析失败: {str(e)}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
