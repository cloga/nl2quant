"""
创业板ETF期权策略 - 分离牛熊市分析
分析策略在牛市中的表现
"""
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config
from test.backtest_gem_option_strategy import OptionStrategyBacktest, fetch_gem_data

def identify_bull_bear_markets(price_data, window=252):
    """
    识别牛熊市
    使用 250 日移动平均线：价格 > 250MA 为牛市，< 250MA 为熊市
    """
    price_data = price_data.copy()
    price_data['MA250'] = price_data['Close'].rolling(window=window).mean()
    price_data['Market'] = 'BULL'
    price_data.loc[price_data['Close'] < price_data['MA250'], 'Market'] = 'BEAR'
    price_data.loc[price_data['MA250'].isna(), 'Market'] = 'UNKNOWN'
    return price_data

def run_backtest_by_market(price_data, market_type='BULL', contracts_per_trade=100):
    """
    运行指定市场类型的回测
    """
    print(f"\n{'='*120}")
    print(f"{market_type}市分析")
    print(f"{'='*120}")
    
    # 筛选数据
    if market_type == 'BULL':
        market_data = price_data[price_data['Market'] == 'BULL'].copy()
    elif market_type == 'BEAR':
        market_data = price_data[price_data['Market'] == 'BEAR'].copy()
    else:
        market_data = price_data.copy()
    
    if len(market_data) == 0:
        print(f"没有 {market_type} 市数据")
        return None
    
    print(f"数据范围: {market_data.index[0].strftime('%Y-%m-%d')} 至 {market_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"总天数: {len(market_data)}")
    
    # 运行回测
    backtest = OptionStrategyBacktest(market_data, initial_capital=100000, contracts_per_trade=contracts_per_trade)
    backtest.run()
    
    return backtest

def print_monthly_summary(backtest):
    """打印月度操作总结 - 简化版（跳过处理nan）"""
    trades_df = pd.DataFrame(backtest.trades)
    
    if len(trades_df) == 0:
        return
    
    print(f"\n交易统计:")
    
    # 统计各类交易
    put_sell = len(trades_df[trades_df['action'] == 'SELL_PUT'])
    put_exercised = len(trades_df[trades_df['action'] == 'PUT_EXERCISED'])
    put_expired = len(trades_df[trades_df['action'] == 'PUT_EXPIRED'])
    call_sell = len(trades_df[trades_df['action'] == 'SELL_CALL'])
    call_exercised = len(trades_df[trades_df['action'] == 'CALL_EXERCISED'])
    call_expired = len(trades_df[trades_df['action'] == 'CALL_EXPIRED'])
    
    print(f"  认沽卖出: {put_sell} 次")
    print(f"  认沽被行权: {put_exercised} 次")
    print(f"  认沽到期: {put_expired} 次")
    print(f"  认购卖出: {call_sell} 次")
    print(f"  认购被行权: {call_exercised} 次")
    print(f"  认购到期: {call_expired} 次")
    
    # 计算权利金收入
    premium_total = 0
    if 'premium' in trades_df.columns:
        premium_total = trades_df[trades_df['action'].isin(['SELL_PUT', 'SELL_CALL'])]['premium'].fillna(0).sum()
    
    print(f"  权利金总收入: {premium_total:,.2f}")

def main():
    print("获取创业板指数历史数据...")
    price_data = fetch_gem_data()
    
    # 识别牛熊市
    print("识别牛熊市...")
    price_data = identify_bull_bear_markets(price_data)
    
    # 统计牛熊市占比
    bull_pct = (price_data['Market'] == 'BULL').sum() / len(price_data) * 100
    bear_pct = (price_data['Market'] == 'BEAR').sum() / len(price_data) * 100
    
    print(f"\n牛市占比: {bull_pct:.1f}%")
    print(f"熊市占比: {bear_pct:.1f}%")
    
    # 分别运行牛市和熊市回测（使用100张 = 10000股）
    contracts_per_trade = 100
    
    # 牛市回测
    backtest_bull = run_backtest_by_market(price_data, 'BULL', contracts_per_trade=contracts_per_trade)
    print_monthly_summary(backtest_bull)
    
    # 熊市回测
    backtest_bear = run_backtest_by_market(price_data, 'BEAR', contracts_per_trade=contracts_per_trade)
    print_monthly_summary(backtest_bear)
    
    # 对比
    print(f"\n{'='*120}")
    print("牛熊市对比")
    print(f"{'='*120}")
    
    if backtest_bull:
        equity_df_bull = pd.DataFrame(backtest_bull.daily_equity)
        # 清除nan值
        equity_df_bull = equity_df_bull.dropna(subset=['total_equity'])
        if len(equity_df_bull) > 0:
            final_bull = equity_df_bull.iloc[-1]['total_equity']
            print(f"牛市最终资产: {final_bull:,.2f} (收益率: {(final_bull/100000-1)*100:.2f}%)")
        else:
            print("牛市回测结果全为nan，无法计算")
    
    if backtest_bear:
        equity_df_bear = pd.DataFrame(backtest_bear.daily_equity)
        # 清除nan值
        equity_df_bear = equity_df_bear.dropna(subset=['total_equity'])
        if len(equity_df_bear) > 0:
            final_bear = equity_df_bear.iloc[-1]['total_equity']
            print(f"熊市最终资产: {final_bear:,.2f} (收益率: {(final_bear/100000-1)*100:.2f}%)")
        else:
            print("熊市回测结果全为nan，无法计算")

if __name__ == '__main__':
    main()
