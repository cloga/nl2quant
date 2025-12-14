
import tushare as ts
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime, timedelta

load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

# Try to fetch dividends for the last 1 year globally
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Fetching dividends from {start_date.strftime('%Y%m%d')} to {end_date.strftime('%Y%m%d')}...")

try:
    # Try single fetch
    code = "600519.SH"
    print(f"Fetching for {code}...")
    df = pro.dividend(ts_code=code, fields='ts_code,end_date,div_proc,cash_div_tax')
    print(f"Fetched {len(df)} rows.")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
