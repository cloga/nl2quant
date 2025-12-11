#!/usr/bin/env python
"""
获取股票市盈率数据
包括：静态市盈率(PE-静)、动态市盈率(PE-动)、TTM市盈率(PE-TTM)
"""

import tushare as ts
import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到路径
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化Tushare
TUSHARE_TOKEN = os.getenv('TUSHARE_TOKEN')
if not TUSHARE_TOKEN:
    raise ValueError("请在.env文件中设置TUSHARE_TOKEN")

ts.set_token(TUSHARE_TOKEN)
pro = ts.pro_api()


def get_daily_basic(ts_code, start_date=None, end_date=None):
    """
    获取每日指标数据，包含市盈率等估值指标
    
    参数:
        ts_code: 股票代码，如 '600000.SH'
        start_date: 开始日期，格式 'YYYYMMDD'
        end_date: 结束日期，格式 'YYYYMMDD'
    
    返回字段:
        ts_code: 股票代码
        trade_date: 交易日期
        close: 收盘价
        turnover_rate: 换手率
        turnover_rate_f: 换手率（自由流通股）
        volume_ratio: 量比
        pe: 市盈率（总市值/净利润，亏损的PE为空）
        pe_ttm: 市盈率TTM（总市值/近12个月净利润）
        pb: 市净率（总市值/净资产）
        ps: 市销率
        ps_ttm: 市销率TTM
        dv_ratio: 股息率（%）
        dv_ttm: 股息率TTM（%）
        total_share: 总股本（万股）
        float_share: 流通股本（万股）
        free_share: 自由流通股本（万股）
        total_mv: 总市值（万元）
        circ_mv: 流通市值（万元）
    """
    df = pro.daily_basic(
        ts_code=ts_code,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,trade_date,close,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv'
    )
    return df


def get_latest_pe(ts_code):
    """
    获取最新的市盈率数据
    
    返回:
        dict包含：
        - pe: 静态市盈率（最新年报）
        - pe_ttm: TTM市盈率（近12个月）
    """
    df = pro.daily_basic(
        ts_code=ts_code,
        fields='ts_code,trade_date,close,pe,pe_ttm,pb,total_mv,circ_mv'
    )
    
    if df.empty:
        return None
    
    # 获取最新数据
    latest = df.iloc[0]
    
    result = {
        'ts_code': latest['ts_code'],
        'trade_date': latest['trade_date'],
        'close': latest['close'],
        'pe_static': latest['pe'],  # 静态市盈率
        'pe_ttm': latest['pe_ttm'],  # TTM市盈率
        'pb': latest['pb'],
        'total_mv': latest['total_mv'],  # 总市值（万元）
        'circ_mv': latest['circ_mv']     # 流通市值（万元）
    }
    
    return result


def get_pe_dynamic(ts_code, trade_date=None):
    """
    计算动态市盈率
    
    动态PE = 静态PE / (1 + 年化净利润增长率)
    
    注意：动态市盈率需要预测未来盈利，这里提供框架，
    实际使用需要结合业绩预告、行业分析等数据
    
    参数:
        ts_code: 股票代码
        trade_date: 交易日期，格式 'YYYYMMDD'
    """
    # 获取业绩预告数据
    df_forecast = pro.forecast(ts_code=ts_code)
    
    if df_forecast.empty:
        print(f"未找到 {ts_code} 的业绩预告数据")
        return None
    
    print(f"\n业绩预告数据（{ts_code}）：")
    print(df_forecast[['end_date', 'type', 'p_change_min', 'p_change_max', 'net_profit_min', 'net_profit_max']])
    
    return df_forecast


def get_fina_indicator(ts_code, period=None):
    """
    获取财务指标数据，包含更详细的盈利指标
    
    参数:
        ts_code: 股票代码
        period: 报告期，格式 'YYYYMMDD'，如20231231表示2023年年报
    
    返回字段包括：
        ts_code: 股票代码
        end_date: 报告期
        eps: 每股收益
        dt_eps: 稀释每股收益
        total_revenue_ps: 每股营业总收入
        revenue_ps: 每股营业收入
        capital_rese_ps: 每股资本公积
        surplus_rese_ps: 每股盈余公积
        undist_profit_ps: 每股未分配利润
        extra_item: 非经常性损益
        profit_dedt: 扣除非经常性损益后的净利润
        gross_margin: 销售毛利率
        current_ratio: 流动比率
        quick_ratio: 速动比率
        ...
    """
    df = pro.fina_indicator(
        ts_code=ts_code,
        period=period,
        fields='ts_code,end_date,eps,dt_eps,profit_dedt,gross_margin,netprofit_margin,roe,roa'
    )
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='获取股票市盈率数据')
    parser.add_argument('--code', type=str, required=True, help='股票代码，如 600000.SH')
    parser.add_argument('--start', type=str, help='开始日期，格式 YYYYMMDD')
    parser.add_argument('--end', type=str, help='结束日期，格式 YYYYMMDD')
    parser.add_argument('--history', action='store_true', help='获取历史数据')
    parser.add_argument('--forecast', action='store_true', help='获取业绩预告（用于计算动态PE）')
    parser.add_argument('--fina', action='store_true', help='获取财务指标')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"股票代码: {args.code}")
    print(f"{'='*60}")
    
    if args.history:
        # 获取历史数据
        print("\n获取每日估值指标...")
        df_daily = get_daily_basic(args.code, args.start, args.end)
        print(f"\n共获取 {len(df_daily)} 条数据")
        print("\n最近5个交易日数据：")
        print(df_daily.head())
        
        # 保存到CSV
        output_file = f"{args.code.replace('.', '_')}_pe_history.csv"
        df_daily.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n数据已保存到: {output_file}")
    
    else:
        # 获取最新数据
        print("\n获取最新市盈率数据...")
        result = get_latest_pe(args.code)
        
        if result:
            print(f"\n交易日期: {result['trade_date']}")
            print(f"收盘价: {result['close']:.2f} 元")
            print(f"静态市盈率(PE): {result['pe_static']:.2f}" if result['pe_static'] else "静态市盈率(PE): N/A（可能亏损）")
            print(f"TTM市盈率(PE-TTM): {result['pe_ttm']:.2f}" if result['pe_ttm'] else "TTM市盈率(PE-TTM): N/A（可能亏损）")
            print(f"市净率(PB): {result['pb']:.2f}" if result['pb'] else "市净率(PB): N/A")
            print(f"总市值: {result['total_mv']:,.0f} 万元 ({result['total_mv']/10000:.2f} 亿元)")
            print(f"流通市值: {result['circ_mv']:,.0f} 万元 ({result['circ_mv']/10000:.2f} 亿元)")
    
    if args.forecast:
        # 获取业绩预告
        print("\n获取业绩预告数据...")
        df_forecast = get_pe_dynamic(args.code)
    
    if args.fina:
        # 获取财务指标
        print("\n获取财务指标数据...")
        df_fina = get_fina_indicator(args.code)
        print(f"\n共获取 {len(df_fina)} 期财报数据")
        print("\n最近财报数据：")
        print(df_fina.head())
    
    print(f"\n{'='*60}")
    print("说明：")
    print("1. 静态市盈率(PE)：使用最新年报的净利润计算")
    print("2. TTM市盈率(PE-TTM)：使用最近12个月的净利润计算（更准确）")
    print("3. 动态市盈率(PE动)：需要结合业绩预告和盈利预测，参考 --forecast 参数")
    print("4. 市盈率为空表示公司亏损")
    print(f"{'='*60}\n")
