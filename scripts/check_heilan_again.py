
import tushare as ts
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

code = '600398.SH' # Heilan Home
print(f"Checking data for {code}...")

df_div = pro.dividend(ts_code=code, fields='end_date,div_proc,cash_div_tax')
df_impl = df_div[df_div['div_proc'] == '实施'].sort_values('end_date', ascending=False)
print(df_impl.head(10))
