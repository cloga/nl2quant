"""
分析通过卖认购权利金摊薄成本的效果
"""
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config

# 读取交易记录
trades_df = pd.read_csv('d:/project/nl2quant/test/option_strategy_trades.csv')
trades_df['date'] = pd.to_datetime(trades_df['date'])

print("="*100)
print("持仓成本摊薄分析")
print("="*100)

# 追踪每个持仓批次的成本变化
position_batches = []  # 存储每个持仓批次

for idx, row in trades_df.iterrows():
    action = row['action']
    date = row['date']
    
    if action == 'PUT_EXERCISED':
        # 新的持仓批次
        shares = row['shares']
        cost_per_share = row['price']
        total_cost = row['cost']
        
        position_batches.append({
            'entry_date': date,
            'entry_price': cost_per_share,
            'shares': shares,
            'initial_cost': total_cost,
            'call_premiums': [],  # 卖认购收到的权利金
            'current_cost': total_cost,  # 当前成本（随着权利金调整）
            'exit_date': None,
            'exit_price': None,
            'exit_profit': None
        })
    
    elif action == 'SELL_CALL' and position_batches:
        # 卖认购，权利金摊薄最新的持仓
        premium = row.get('premium', 0)
        if premium > 0:
            # 权利金摊薄到现有持仓
            active_batch = position_batches[-1]
            if active_batch['exit_date'] is None:  # 还未平仓
                active_batch['call_premiums'].append((date, premium))
                active_batch['current_cost'] -= premium
    
    elif action == 'CALL_EXERCISED':
        # 平仓
        if position_batches:
            active_batch = position_batches[-1]
            if active_batch['exit_date'] is None:
                active_batch['exit_date'] = date
                active_batch['exit_price'] = row.get('price', 0)
                profit = row.get('profit', 0)
                active_batch['exit_profit'] = profit

print("\n持仓批次详细分析：")
print("-"*100)

total_premium_collected = 0
for i, batch in enumerate(position_batches, 1):
    entry_date = batch['entry_date'].strftime('%Y-%m-%d')
    entry_price = batch['entry_price']
    shares = batch['shares']
    initial_cost = batch['initial_cost']
    
    # 计算卖认购累计权利金
    call_premium_total = sum(p[1] for p in batch['call_premiums'])
    total_premium_collected += call_premium_total
    
    # 计算摊薄后的成本
    adjusted_cost = initial_cost - call_premium_total
    adjusted_cost_per_share = adjusted_cost / shares if shares > 0 else 0
    
    print(f"\n📌 第 {i} 个持仓批次:")
    print(f"   接入时间: {entry_date}")
    print(f"   接入价格: {entry_price:.2f} 元/股")
    print(f"   股数: {shares:.0f} 股")
    print(f"   初始成本: {initial_cost:,.2f} 元")
    
    if batch['call_premiums']:
        print(f"   卖认购数: {len(batch['call_premiums'])} 次")
        print(f"   卖认购总权利金: {call_premium_total:,.2f} 元 ({call_premium_total/initial_cost*100:.2f}% of 初始成本)")
    else:
        print(f"   卖认购数: 0 次")
    
    print(f"   摊薄后成本: {adjusted_cost:,.2f} 元")
    print(f"   摊薄后成本/股: {adjusted_cost_per_share:.2f} 元")
    print(f"   成本降幅: {(1 - adjusted_cost_per_share/entry_price)*100:.2f}%")
    
    if batch['exit_date']:
        exit_date = batch['exit_date'].strftime('%Y-%m-%d')
        exit_price = batch['exit_price']
        exit_profit = batch['exit_profit']
        
        # 基于摊薄成本计算收益
        profit_per_share = exit_price - adjusted_cost_per_share
        total_profit = profit_per_share * shares
        
        print(f"   退出时间: {exit_date}")
        print(f"   退出价格: {exit_price:.2f} 元/股")
        print(f"   持仓时长: {(batch['exit_date'] - batch['entry_date']).days} 天")
        print(f"   基于初始成本的利润: {exit_profit:,.2f} 元")
        print(f"   基于摊薄成本的利润: {total_profit:,.2f} 元")
        print(f"   基于摊薄成本的收益率: {(exit_price/adjusted_cost_per_share - 1)*100:.2f}%")
    else:
        print(f"   状态: 未平仓（仍持有）")
        print(f"   持仓时长: {(pd.Timestamp.now() - batch['entry_date']).days} 天")

print("\n" + "="*100)
print("总体摊薄效果:")
print("="*100)
print(f"累计权利金收入: {total_premium_collected:,.2f} 元")
print(f"平均每次持仓批次降低成本: {total_premium_collected/max(len(position_batches), 1):,.2f} 元")

# 计算如果全部出清，基于摊薄成本的总收益
if position_batches:
    last_batch = position_batches[-1]
    if last_batch['exit_date']:
        print(f"\n✅ 已完成平仓: {len([b for b in position_batches if b['exit_date']])} 批次")
    else:
        call_premium_total = sum(p[1] for p in last_batch['call_premiums'])
        print(f"\n⏳ 仍有持仓未平仓")
        print(f"   初始成本: {last_batch['initial_cost']:,.2f} 元")
        print(f"   已摊薄: {call_premium_total:,.2f} 元")
        print(f"   实际成本: {last_batch['initial_cost'] - call_premium_total:,.2f} 元")
        print(f"   需要达到的行权价（+10%）: {last_batch['entry_price'] * 1.1:.2f} 元")
