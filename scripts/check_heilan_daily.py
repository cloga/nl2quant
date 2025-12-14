
import tushare as ts
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

code = '600398.SH'
print(f"Checking daily basic for {code}...")
df = pro.daily_basic(ts_code=code, trade_date='20251212', fields='dv_ratio,dv_ttm')
print(df)
