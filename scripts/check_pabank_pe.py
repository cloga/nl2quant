#!/usr/bin/env python
"""检查平安银行的券商PE预测数据"""
import tushare as ts
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()

df = pro.report_rc(ts_code='000001.SZ')
df['report_date'] = pd.to_datetime(df['report_date'])
df_recent = df[df['report_date'] >= '2024-09-01'].sort_values('report_date', ascending=False)

print('平安银行最近3个月的券商报告PE预测:')
print(df_recent[['report_date', 'org_name', 'rating', 'pe', 'eps']].head(15))
print(f'\nPE中位数: {df_recent["pe"].median():.2f}')
print(f'PE平均数: {df_recent["pe"].mean():.2f}')
print(f'\n有效PE数量: {df_recent["pe"].notna().sum()}')
