
import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

code = '000002.SZ'
period = '20250930'

print(f"Fetching data for {code} at {period}...")

# 1. Income
print("\n--- Income ---")
df_inc = pro.income(ts_code=code, period=period, fields='ts_code,total_revenue,revenue,n_income_attr_p')
print(df_inc)

# 2. Balance Sheet
print("\n--- Balance Sheet ---")
df_bal = pro.balancesheet(ts_code=code, period=period, fields='ts_code,total_assets,total_liab,total_cur_assets')
print(df_bal)












# Test generic query for balancesheet
print("\n--- Generic Query Balance Sheet ---")
try:
    df_gen = pro.query('balancesheet', period=period, fields='ts_code,total_cur_assets,total_liab')
    print(f"Fetched {len(df_gen)} rows.")
    print(df_gen.head())
except Exception as e:
    print(f"Error: {e}")

