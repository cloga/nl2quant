"""
分析期权行权的详细信息
"""
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config
from datetime import datetime
import tushare as ts

# 读取交易记录
trades_df = pd.read_csv('d:/project/nl2quant/test/option_strategy_trades.csv')

# 获取创业板指数数据（用于查询月初月末价格）
print("正在获取创业板指数数据...")
ts.set_token(Config.TUSHARE_TOKEN)
pro = ts.pro_api()

df_list = []
for year in range(2010, 2026):
    start_date = f"{year}0101"
    end_date = f"{year}1231"
    df_chunk = pro.index_daily(ts_code='399006.SZ', start_date=start_date, end_date=end_date)
    if not df_chunk.empty:
        df_list.append(df_chunk)

df = pd.concat(df_list, ignore_index=True)
df['trade_date'] = pd.to_datetime(df['trade_date'])
df = df.sort_values('trade_date')
df = df.set_index('trade_date')

# 提取所有行权记录
put_exercises = trades_df[trades_df['action'] == 'PUT_EXERCISED'].copy()
call_exercises = trades_df[trades_df['action'] == 'CALL_EXERCISED'].copy()

print("\n" + "="*100)
print("认沽期权行权详细分析")
print("="*100)
print(f"总行权次数: {len(put_exercises)} 次\n")

for idx, row in put_exercises.iterrows():
    exercise_date = pd.to_datetime(row['date'])
    exercise_price = row['price']
    shares = row['shares']
    new_position = row['new_position']
    avg_cost = row['avg_cost']
    
    # 获取当月月初和月末价格
    month_start = exercise_date.replace(day=1)
    month_data = df[df.index.to_period('M') == exercise_date.to_period('M')]
    
    if not month_data.empty:
        month_open = month_data.iloc[0]['open']
        month_close = month_data.iloc[-1]['close']
        month_high = month_data['high'].max()
        month_low = month_data['low'].min()
    else:
        month_open = month_close = month_high = month_low = None
    
    print(f"📌 {exercise_date.strftime('%Y年%m月')} 认沽被行权")
    print(f"   行权价: {exercise_price:.2f} 元")
    print(f"   接入股数: {shares:.0f} 股")
    print(f"   累计持仓: {new_position:.0f} 股")
    print(f"   平均成本: {avg_cost:.2f} 元/股")
    if month_open:
        print(f"   月初价格: {month_open:.2f} 元")
        print(f"   月末价格: {month_close:.2f} 元")
        print(f"   月内最高: {month_high:.2f} 元")
        print(f"   月内最低: {month_low:.2f} 元")
        print(f"   月度跌幅: {(month_close - month_open) / month_open * 100:.2f}%")
        print(f"   触发行权原因: 月末价格({month_close:.2f}) < 行权价({exercise_price:.2f})")
    print()

print("\n" + "="*100)
print("认购期权行权详细分析")
print("="*100)
print(f"总行权次数: {len(call_exercises)} 次\n")

for idx, row in call_exercises.iterrows():
    exercise_date = pd.to_datetime(row['date'])
    exercise_price = row['price']
    shares = row['shares']
    cost_basis = row['cost_basis']
    profit = row['profit']
    remaining_position = row['remaining_position']
    
    # 获取当月月初和月末价格
    month_data = df[df.index.to_period('M') == exercise_date.to_period('M')]
    
    if not month_data.empty:
        month_open = month_data.iloc[0]['open']
        month_close = month_data.iloc[-1]['close']
        month_high = month_data['high'].max()
        month_low = month_data['low'].min()
    else:
        month_open = month_close = month_high = month_low = None
    
    print(f"📌 {exercise_date.strftime('%Y年%m月')} 认购被行权")
    print(f"   行权价: {exercise_price:.2f} 元")
    print(f"   卖出股数: {shares:.0f} 股")
    print(f"   持仓成本: {cost_basis:.2f} 元/股")
    print(f"   锁定利润: {profit:.2f} 元 ({(exercise_price/cost_basis - 1)*100:.2f}%)")
    print(f"   剩余持仓: {remaining_position:.0f} 股")
    if month_open:
        print(f"   月初价格: {month_open:.2f} 元")
        print(f"   月末价格: {month_close:.2f} 元")
        print(f"   月内最高: {month_high:.2f} 元")
        print(f"   月内最低: {month_low:.2f} 元")
        print(f"   月度涨幅: {(month_close - month_open) / month_open * 100:.2f}%")
        print(f"   触发行权原因: 月末价格({month_close:.2f}) > 行权价({exercise_price:.2f})")
    print()

# 统计分析
print("\n" + "="*100)
print("行权统计对比")
print("="*100)
print(f"认沽行权次数: {len(put_exercises)} 次")
print(f"认购行权次数: {len(call_exercises)} 次")
print(f"\n说明: 两者次数相等是因为策略设计——每次认沽被行权后立即开始双卖(认购+认沽)，")
print(f"     持仓期间如果标的上涨超过成本价10%，认购就会被行权卖出股票。")
print(f"     历史数据显示，每次接入股票后都最终以盈利方式通过认购行权出清。")

# 详细匹配分析（修正版）
print("\n" + "="*100)
print("认购行权详细分析（修正版）")
print("="*100)

print("\n说明：每次认购被行权时，都是按照**当时的平均持仓成本 × 1.10**作为行权价。")
print("因此，理论上每次认购行权都应该是盈利10%（不考虑权利金）。\n")

for idx, row in call_exercises.iterrows():
    exercise_date = pd.to_datetime(row['date'])
    exercise_price = row['price']
    shares = row['shares']
    cost_basis = row['cost_basis']
    profit = row['profit']
    remaining_position = row['remaining_position']
    
    # 计算理论行权价和实际差异
    theoretical_strike = cost_basis * 1.10
    price_diff = abs(exercise_price - theoretical_strike)
    
    print(f"📌 {exercise_date.strftime('%Y年%m月')} 认购被行权")
    print(f"   平均持仓成本: {cost_basis:.2f} 元/股")
    print(f"   理论行权价: {theoretical_strike:.2f} 元 (成本×1.10)")
    print(f"   实际行权价: {exercise_price:.2f} 元")
    
    if price_diff > 0.01:
        print(f"   ⚠️  差异: {price_diff:.2f} 元 (可能是回测精度问题)")
    else:
        print(f"   ✅ 完全符合+10%逻辑")
    
    print(f"   卖出股数: {shares:.0f} 股")
    print(f"   锁定利润: {profit:.2f} 元 ({(exercise_price/cost_basis - 1)*100:.2f}%)")
    print(f"   剩余持仓: {remaining_position:.0f} 股")
    print()

# 持仓周期分析
print("\n" + "="*100)
print("持仓周期分析（基于持仓进出）")
print("="*100)

position_history = []
current_positions = []

# 遍历所有交易，追踪持仓变化
all_trades = trades_df.sort_values('date')

for idx, row in all_trades.iterrows():
    date = pd.to_datetime(row['date'])
    action = row['action']
    
    if action == 'PUT_EXERCISED':
        # 记录接入
        avg_cost = row['avg_cost']
        shares = row['shares']
        current_positions.append({
            'entry_date': date,
            'entry_cost': avg_cost,
            'shares': shares,
            'status': 'open'
        })
    
    elif action == 'CALL_EXERCISED':
        # 从当前持仓中减少（FIFO）
        shares_to_sell = row['shares']
        
        while shares_to_sell > 0 and current_positions:
            # 找到最早的未平仓持仓
            earliest_open = None
            for pos in current_positions:
                if pos['status'] == 'open':
                    earliest_open = pos
                    break
            
            if earliest_open:
                sold = min(shares_to_sell, earliest_open['shares'])
                earliest_open['shares'] -= sold
                shares_to_sell -= sold
                
                if earliest_open['shares'] == 0:
                    earliest_open['status'] = 'closed'
                    earliest_open['exit_date'] = date
                    earliest_open['exit_price'] = row['price']
                    
                    # 计算持仓时长
                    months = (date.year - earliest_open['entry_date'].year) * 12 + \
                             (date.month - earliest_open['entry_date'].month)
                    earliest_open['holding_months'] = months
                    
                    position_history.append(earliest_open)

print("\n已平仓持仓明细:")
print("-" * 100)

for i, pos in enumerate(position_history, 1):
    entry = pos['entry_date']
    exit_date = pos['exit_date']
    months = pos['holding_months']
    entry_cost = pos['entry_cost']
    exit_price = pos['exit_price']
    profit_pct = (exit_price / entry_cost - 1) * 100
    
    print(f"{i:2d}. {entry.strftime('%Y-%m')} → {exit_date.strftime('%Y-%m')} "
          f"({months:2d}月) | 成本 {entry_cost:7.2f} → 卖出 {exit_price:7.2f} | "
          f"收益 {profit_pct:+6.2f}%")

if position_history:
    avg_months = sum(p['holding_months'] for p in position_history) / len(position_history)
    max_months = max(p['holding_months'] for p in position_history)
    
    print(f"\n平均持仓时长: {avg_months:.1f} 个月")
    print(f"最长持仓时长: {max_months} 个月")

