#!/usr/bin/env python
"""测试report_rc接口"""
import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv()
ts.set_token(os.getenv('TUSHARE_TOKEN'))
pro = ts.pro_api()

print('='*60)
print('测试 report_rc 接口（券商投研报告）')
print('文档: https://tushare.pro/document/2?doc_id=203')
print('='*60)

try:
    df = pro.report_rc(ts_code='600000.SH')
    print(f'\n✓ 可用 - 获取到 {len(df)} 条记录')
    
    if not df.empty:
        print(f'\n字段列表: {df.columns.tolist()}')
        print('\n最新5条记录:')
        print(df.head()[['title', 'researcher', 'org_name', 'report_date']])
        
except Exception as e:
    print(f'\n✗ 不可用 - {e}')
