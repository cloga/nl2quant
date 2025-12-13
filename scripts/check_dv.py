
import tushare as ts
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

# Check fields for daily_basic
try:
    df = pro.daily_basic(ts_code='000001.SZ', limit=1, fields='ts_code,dv_ratio,dv_ttm')
    print(df)
except Exception as e:
    print(e)
