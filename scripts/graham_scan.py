"""
Graham Valuation Scan for A-Shares.

Calculates Benjamin Graham's value investing metrics for all A-share stocks:
1. Net Current Asset Value (NCAV) / Net-Net:
   - NCAV = Current Assets - Total Liabilities
   - Signal: Price < 2/3 * NCAV per share
2. Graham Number:
   - V = Sqrt(22.5 * EPS * BVPS)
   - Signal: Price < Graham Number (or PE * PB < 22.5)
3. Defensive Investor Criteria (Simplified):
   - Current Ratio > 2.0
   - Long-term Debt < Working Capital

Usage:
  python scripts/graham_scan.py
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import tushare as ts
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from app.utils.rate_limiter import GLOBAL_LIMITER

def get_pro():
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        print("Error: TUSHARE_TOKEN not found in .env")
        sys.exit(1)
    ts.set_token(token)
    return ts.pro_api()

def _ts_call(func, **kwargs):
    """Wrapper for Tushare calls with rate limiting."""
    max_retries = 3
    for i in range(max_retries):
        try:
            GLOBAL_LIMITER.acquire()
            return func(**kwargs)
        except Exception as e:
            if "每分钟最多" in str(e) or "500" in str(e):
                time.sleep(10 * (i + 1))
            elif i == max_retries - 1:
                raise e
            else:
                time.sleep(1)
    return None

def get_latest_trade_date(pro):
    today = datetime.now().strftime('%Y%m%d')
    df = _ts_call(pro.trade_cal, exchange='SSE', is_open='1', end_date=today, limit=1, fields='cal_date')
    return df['cal_date'].iloc[0]

def get_financial_period(pro, trade_date):
    """Determine the latest likely financial reporting period."""
    # Simple heuristic: 
    # May 1 -> Q1 (0331)
    # Sep 1 -> Q2 (0630)
    # Nov 1 -> Q3 (0930)
    # May 1 (next year) -> Annual (1231)
    
    dt = datetime.strptime(trade_date, '%Y%m%d')
    year = dt.year
    md = dt.month * 100 + dt.day
    
    if md < 501:
        # Before May 1st, use Q3 of previous year (Annual might not be fully out, but let's try Q3 first or Annual of prev year)
        # Actually, by May 1st, Annual and Q1 must be out.
        # Let's just try to fetch the latest available period for a major stock to determine the period.
        pass
    
    # Better approach: Fetch balance sheet for a big bank (e.g., 601398.SH) to see latest period
    df = _ts_call(pro.balancesheet, ts_code='601398.SH', limit=1, fields='end_date')
    if not df.empty:
        return df['end_date'].iloc[0]
    return '20240930' # Fallback

def fetch_data(pro):
    print("Fetching latest trade date...")
    trade_date = get_latest_trade_date(pro)
    print(f"Latest trade date: {trade_date}")

    print("Fetching daily basic data (Price, PE, PB)...")
    # daily_basic gives us: close, pe, pe_ttm, pb, total_share, total_mv, circ_mv
    df_daily = _ts_call(pro.daily_basic, trade_date=trade_date, fields='ts_code,trade_date,close,pe,pe_ttm,pb,total_share,total_mv')
    
    print("Determining latest financial period...")
    period = get_financial_period(pro, trade_date)
    print(f"Using financial period: {period}")

    print("Fetching balance sheet (Assets, Liabilities)...")
    # Need: total_cur_assets, total_liab, total_non_cur_liab (or total_liab - total_cur_liab)
    # Note: Tushare balancesheet fields: total_cur_assets, total_cur_liab, total_liab
    # We fetch for all stocks in that period.
    # Since pro.balancesheet requires ts_code, we batch fetch using ts_codes from df_daily
    ts_codes = df_daily['ts_code'].tolist()
    
    df_bal_list = []
    df_fina_list = []
    
    batch_size = 50
    total_stocks = len(ts_codes)
    
    print(f"Fetching financials for {total_stocks} stocks in batches of {batch_size}...")
    
    for i in range(0, total_stocks, batch_size):
        batch_codes = ",".join(ts_codes[i:i+batch_size])
        
        # Balance Sheet
        try:
            df_b = _ts_call(pro.balancesheet, ts_code=batch_codes, period=period, fields='ts_code,total_cur_assets,total_cur_liab,total_liab,total_assets')
            if df_b is not None and not df_b.empty:
                df_bal_list.append(df_b)
        except Exception as e:
            print(f"Error fetching balancesheet batch {i}: {e}")

        # Financial Indicators
        try:
            df_f = _ts_call(pro.fina_indicator, ts_code=batch_codes, period=period, fields='ts_code,eps,bps,current_ratio')
            if df_f is not None and not df_f.empty:
                df_fina_list.append(df_f)
        except Exception as e:
            print(f"Error fetching fina_indicator batch {i}: {e}")
            
        # Simple progress indicator
        if (i // batch_size) % 5 == 0:
            print(f"Processed {min(i+batch_size, total_stocks)}/{total_stocks}...")

    if df_bal_list:
        df_bal = pd.concat(df_bal_list, ignore_index=True)
        df_bal = df_bal.drop_duplicates(subset=['ts_code'], keep='first')
    else:
        df_bal = pd.DataFrame(columns=['ts_code', 'total_cur_assets', 'total_cur_liab', 'total_liab', 'total_assets'])

    if df_fina_list:
        df_fina = pd.concat(df_fina_list, ignore_index=True)
        df_fina = df_fina.drop_duplicates(subset=['ts_code'], keep='first')
    else:
        df_fina = pd.DataFrame(columns=['ts_code', 'eps', 'bps', 'current_ratio'])

    return df_daily, df_bal, df_fina, trade_date, period

def calculate_graham_metrics(df_daily, df_bal, df_fina):
    print("Merging data...")
    df = pd.merge(df_daily, df_bal, on='ts_code', how='left')
    df = pd.merge(df, df_fina, on='ts_code', how='left')

    print("Calculating Graham metrics...")
    
    # 1. NCAV (Net Current Asset Value)
    # NCAV = Current Assets - Total Liabilities
    # NCAV per share = NCAV / Total Shares
    # Tushare total_share is in 10k? No, daily_basic total_share is in 'ten thousand shares' (万股).
    # Financials are usually in Yuan (CNY).
    # We need to align units.
    # daily_basic: total_share (万股) -> * 10000 -> shares
    # balancesheet: total_cur_assets (CNY), total_liab (CNY)
    
    df['total_shares_exact'] = df['total_share'] * 10000
    df['ncav'] = df['total_cur_assets'] - df['total_liab']
    df['ncav_per_share'] = df['ncav'] / df['total_shares_exact']
    
    # NCAV Ratio = Price / NCAV per share
    # If Price < 0.67 * NCAV, it's a "Net-Net"
    df['ncav_ratio'] = df['close'] / df['ncav_per_share']
    
    # 2. Graham Number
    # V = Sqrt(22.5 * EPS * BVPS)
    # Tushare 'bps' is Book Value Per Share. 'eps' is Earnings Per Share.
    # Handle negatives for sqrt
    df['graham_number'] = np.nan
    mask_pos = (df['eps'] > 0) & (df['bps'] > 0)
    df.loc[mask_pos, 'graham_number'] = np.sqrt(22.5 * df.loc[mask_pos, 'eps'] * df.loc[mask_pos, 'bps'])
    
    df['price_to_graham'] = df['close'] / df['graham_number']
    
    # 3. PE * PB
    df['pe_pb'] = df['pe'] * df['pb']
    
    # 4. Defensive Criteria
    # Current Ratio (already in df_fina)
    # Debt to Working Capital? 
    # Working Capital = Current Assets - Current Liabilities
    # Long Term Debt approx = Total Liab - Total Cur Liab
    df['working_capital'] = df['total_cur_assets'] - df['total_cur_liab']
    df['long_term_debt'] = df['total_liab'] - df['total_cur_liab']
    df['ltd_to_working_capital'] = df['long_term_debt'] / df['working_capital']

    return df

def main():
    pro = get_pro()
    
    df_daily, df_bal, df_fina, trade_date, period = fetch_data(pro)
    
    result_df = calculate_graham_metrics(df_daily, df_bal, df_fina)
    
    # Filter columns for output
    cols = [
        'ts_code', 'trade_date', 'close', 'pe', 'pb', 
        'ncav_per_share', 'ncav_ratio', 
        'graham_number', 'price_to_graham', 'pe_pb',
        'current_ratio', 'ltd_to_working_capital',
        'eps', 'bps'
    ]
    
    # Add stock name
    print("Fetching stock names...")
    df_basic = _ts_call(pro.stock_basic, exchange='', list_status='L', fields='ts_code,name,industry')
    result_df = pd.merge(result_df, df_basic, on='ts_code', how='left')
    
    final_cols = ['ts_code', 'name', 'industry'] + cols[1:]
    final_df = result_df[final_cols].copy()
    
    # Sort by Price/Graham (Ascending) - cheapest relative to Graham Number
    final_df = final_df.sort_values('price_to_graham', ascending=True)
    
    # Save
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f'graham_valuation_{trade_date}.csv'
    
    final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Saved Graham valuation report to {output_file}")
    
    # Print some insights
    print("\n=== Graham Net-Nets (Price < 2/3 NCAV) ===")
    net_nets = final_df[ (final_df['ncav_ratio'] > 0) & (final_df['ncav_ratio'] < 0.67) ]
    if not net_nets.empty:
        print(net_nets[['ts_code', 'name', 'close', 'ncav_per_share', 'ncav_ratio']].head(10).to_string(index=False))
    else:
        print("No Net-Nets found.")

    print("\n=== Undervalued by Graham Number (Price < Graham Number) ===")
    undervalued = final_df[ (final_df['price_to_graham'] < 1.0) & (final_df['price_to_graham'] > 0) ]
    print(f"Found {len(undervalued)} stocks trading below Graham Number.")
    print(undervalued[['ts_code', 'name', 'close', 'graham_number', 'price_to_graham', 'pe_pb']].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
