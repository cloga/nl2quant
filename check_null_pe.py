#!/usr/bin/env python
"""检查 PE 为空的股票原因"""
import pandas as pd

df = pd.read_csv('data/pe_ratios_20251212.csv')
print("总记录数:", len(df))
print("\nPE 为空的股票：")
null_pe = df[(df['pe_static'].isna()) | (df['pe_ttm'].isna()) | (df['pe_dynamic'].isna())]
print(null_pe[['symbol', 'name', 'pe_static', 'pe_ttm', 'pe_dynamic', 'net_profit']].to_string())
print(f"\n共 {len(null_pe)} 条 PE 为空")
