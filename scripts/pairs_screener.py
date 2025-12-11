#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pairs Trading Screener - CLI Tool
命令行工具，用于批量筛选配对交易标的
"""

import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tushare as ts
import os

# 设置UTF-8编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    os.environ['PYTHONIOENCODING'] = 'utf-8'

# 添加项目根目录到路径，以便导入app模块
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.pairs_screener import PairsScreener
from app.config import Config


def load_predefined_pools():
    """返回预定义的股票池"""
    pools = {
        'hs300': [
            '601398', '601939', '601288', '600036', '000858', '600519',
            '600900', '600887', '000651', '000333', '000568', '601188',
            '600016', '601328', '601166', '601669', '601688', '600030',
            '000002', '000001', '001979', '600000', '601601', '601318',
            '601336', '600048', '000096', '000651', '601668', '601211',
        ],
        'banks': [
            '601398', '601939', '601288', '601169', '601658', '600016',
            '601328', '601166', '601618', '601988', '600036', '601818',
            '000001', '001979', '600000', '601818',
        ],
        'liquor': [
            '600519', '000858', '000568', '601633', '603198', '000576',
        ],
        'tech': [
            '300750', '601012', '601390', '603392', '002415', '601012',
        ],
    }
    return pools


def compute_start_date_by_trading_days(end_date_str: str, trading_days: int) -> str:
    """根据交易日数量回溯开始日期。"""
    if trading_days <= 0:
        raise ValueError("trading_days 必须为正整数")
    if not Config.TUSHARE_TOKEN:
        raise ValueError("TUSHARE_TOKEN 未配置，无法按交易日回溯")

    pro = ts.pro_api(Config.TUSHARE_TOKEN)
    # 预留安全区间，抓取足够多的日历记录
    end_dt = datetime.strptime(end_date_str, "%Y%m%d")
    start_probe = (end_dt - timedelta(days=trading_days * 2 + 30)).strftime("%Y%m%d")
    cal = pro.trade_cal(start_date=start_probe, end_date=end_date_str, is_open=1)
    cal = cal[cal["is_open"] == 1].sort_values("cal_date")
    if len(cal) < trading_days:
        raise ValueError(f"交易日样本不足: 仅有 {len(cal)} 个")
    start_cal = cal.iloc[-trading_days]["cal_date"]
    return start_cal


def load_all_a_share_codes() -> list:
    """获取全部A股代码（上市状态为L）。"""
    if not Config.TUSHARE_TOKEN:
        raise ValueError("TUSHARE_TOKEN 未配置，无法获取A股列表")
    pro = ts.pro_api(Config.TUSHARE_TOKEN)
    df = pro.stock_basic(exchange='', list_status='L', fields='ts_code,symbol,name')
    if df is None or df.empty:
        raise ValueError("未能获取到A股列表")
    return df['ts_code'].tolist()


def load_all_etf_codes(start_date_str: str | None = None) -> list:
    """获取全部ETF基金代码（上市状态为L），可按起始日期过滤未上市基金。"""
    if not Config.TUSHARE_TOKEN:
        raise ValueError("TUSHARE_TOKEN 未配置，无法获取ETF列表")
    pro = ts.pro_api(Config.TUSHARE_TOKEN)
    df = pro.fund_basic(market='E', status='L', fields='ts_code,name,fund_type,list_date')
    if df is None or df.empty:
        raise ValueError("未能获取到ETF列表")
    if start_date_str:
        df = df[df['list_date'] <= start_date_str]
    return df['ts_code'].tolist()


def main():
    parser = argparse.ArgumentParser(
        description='A股配对交易标的筛选工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用预定义的沪深300池子
  python pairs_screener.py --pool hs300 --days 365
  
  # 使用自定义股票代码
  python pairs_screener.py --codes 601398,601939,601288,600519 --days 180
  
  # 调整聚类参数
  python pairs_screener.py --pool banks --eps 0.6 --n-components 12
  
  # 保存结果到JSON
  python pairs_screener.py --pool liquor --output results.json
        """
    )
    
    parser.add_argument(
        '--pool',
        type=str,
        choices=list(load_predefined_pools().keys()),
        help='预定义股票池'
    )

    parser.add_argument(
        '--universe',
        type=str,
        choices=['all_a', 'all_etf'],
        help='全市场池：all_a=全部A股，all_etf=全部ETF（优先于 --pool）'
    )
    
    parser.add_argument(
        '--codes',
        type=str,
        help='股票代码（逗号分隔）'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=365,
        help='往前回溯天数（已弃用，推荐使用 --trade-days）'
    )

    parser.add_argument(
        '--trade-days',
        type=int,
        default=2500,
        help='往前回溯交易日数量（默认2500=约10年，优先于 --days）'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='结束日期（YYYYMMDD）'
    )
    
    parser.add_argument(
        '--eps',
        type=float,
        default=0.5,
        help='DBSCAN邻域半径（默认0.5）'
    )

    parser.add_argument(
        '--min-samples',
        type=int,
        default=5,
        help='DBSCAN min_samples（默认5，可尝试2/3降低簇质量）'
    )
    
    parser.add_argument(
        '--n-components',
        type=int,
        default=15,
        help='PCA主成分数（默认15）'
    )

    parser.add_argument(
        '--min-corr',
        type=float,
        default=0.85,
        help='协整前置相关性阈值 (默认0.85)'
    )

    parser.add_argument(
        '--pvalue',
        type=float,
        default=0.05,
        help='协整检验p值阈值 (默认0.05)'
    )

    parser.add_argument(
        '--coint-log-price',
        action='store_true',
        help='协整检验使用对数价格（推荐开启）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='输出文件路径（JSON格式）'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
        help='输出配对结果为CSV'
    )

    parser.add_argument(
        '--cluster-html',
        type=str,
        default='cluster_vis.html',
        help='聚类可视化输出HTML路径（默认 cluster_vis.html）'
    )
    
    args = parser.parse_args()
    
    # 确定日期范围
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y%m%d')
    else:
        end_date = datetime.now()

    end_str = end_date.strftime("%Y%m%d")

    # 优先使用交易日（如果没显式设置trade_days，默认值2500将被使用）
    if args.trade_days != 2500 or args.trade_days is not None:
        start_str = compute_start_date_by_trading_days(end_str, args.trade_days)
        print(f"日期范围: {start_str} ~ {end_str}（回溯 {args.trade_days} 个交易日）")
    elif args.days != 365:
        start_date = end_date - timedelta(days=args.days)
        start_str = start_date.strftime("%Y%m%d")
        print(f"日期范围: {start_str} ~ {end_str}（回溯自然日 {args.days} 天）")
    else:
        # 默认使用trade-days 2500
        start_str = compute_start_date_by_trading_days(end_str, args.trade_days)
        print(f"日期范围: {start_str} ~ {end_str}（回溯 {args.trade_days} 个交易日，约10年）")
    print(f"PCA参数: n_components={args.n_components}")
    print(f"DBSCAN参数: eps={args.eps}")
    print()

    # 确定股票代码（依赖 start_str）
    if args.universe:
        if args.universe == 'all_a':
            codes = load_all_a_share_codes()
            print(f"使用全市场A股池: {len(codes)}只股票")
        else:
            codes = load_all_etf_codes(start_str)
            print(f"使用全市场ETF池（上市<=起始日过滤）: {len(codes)}只基金")
    elif args.pool:
        codes = load_predefined_pools()[args.pool]
        print(f"使用预定义池: {args.pool} ({len(codes)}只股票)")
    elif args.codes:
        codes = [c.strip() for c in args.codes.split(',')]
        print(f"使用自定义代码: {len(codes)}只股票")
    else:
        parser.error("必须指定 --pool 或 --codes")
    
    # 运行筛选
    screener = PairsScreener(start_str, end_str)
    results = screener.run(
        codes,
        eps=args.eps,
        n_components=args.n_components,
        min_corr=args.min_corr,
        pvalue_threshold=args.pvalue,
        min_samples=args.min_samples,
        use_log_price=args.coint_log_price,
    )
    
    # 处理结果
    pairs_df = results['pairs']
    
    print("\n" + "="*60)
    print("筛选结果")
    print("="*60)
    
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('correlation', ascending=False)
        print("\n找到的协整配对:")
        print(pairs_df.to_string(index=False))
        
        # 显示Top5
        print("\nTop 5 最强配对:")
        for idx, (i, row) in enumerate(pairs_df.head(5).iterrows(), 1):
            print(f"{idx}. {row['stock_a']} ↔️ {row['stock_b']}")
            print(f"   相关系数: {row['correlation']:.4f}")
            print(f"   协整P值: {row['coint_pvalue']:.6f}")
            print(f"   协整得分: {row['coint_score']:.4f}")
            print()
    else:
        print("⚠️  未找到协整配对")
    
    # 保存结果
    if args.csv:
        pairs_df.to_csv(args.csv, index=False)
        print(f"✓ 配对结果已保存: {args.csv}")
    
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'start_date': start_str,
                'end_date': end_str,
                'stocks_count': len(codes),
                'eps': args.eps,
                'n_components': args.n_components,
            },
            'pairs_count': len(pairs_df),
            'pairs': pairs_df.to_dict('records'),
            'clusters': {
                'n_clusters': len(set(results['labels'])) - (1 if -1 in results['labels'] else 0),
                'n_noise': list(results['labels']).count(-1),
            }
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"✓ 完整结果已保存: {args.output}")

    # 保存聚类可视化
    if results.get('cluster_fig') is not None:
        html_path = args.cluster_html
        try:
            results['cluster_fig'].write_html(html_path, include_plotlyjs='cdn')
            print(f"✓ 聚类可视化已保存: {html_path}")
        except Exception as e:
            print(f"[WARN] 聚类可视化保存失败: {e}")


if __name__ == '__main__':
    main()
