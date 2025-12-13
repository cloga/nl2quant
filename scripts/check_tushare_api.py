
import tushare as ts
import os
import sys
from dotenv import load_dotenv

# Load .env from current directory
load_dotenv()
token = os.getenv("TUSHARE_TOKEN")
if not token:
    print("No token found")
    sys.exit(1)

ts.set_token(token)
pro = ts.pro_api()

try:
    print("Fetching report_rc by single date...")
    df = pro.report_rc(report_date='20251028', fields='ts_code,report_date,quarter,np')
    if df is not None and not df.empty:
        print(f"Success! Found {len(df)} records.")
        print("Sample:", df.head().to_dict('records'))
    else:
        print("DataFrame is empty")
except Exception as e:
    print(f"Error: {e}")
