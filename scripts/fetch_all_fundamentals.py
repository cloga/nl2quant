"""
Fetch Comprehensive Fundamentals for A-Shares.

Fetches and merges:
1. Stock Basic (Name, Industry, List Date)
2. Daily Basic (Price, PE, PB, Market Cap, Turnover, Dividend Yield)
3. Financial Indicators (ROE, ROA, Gross Margin, Net Margin, Debt Ratio, etc.)
4. Income Statement (Revenue, Net Income, Growth)
5. Balance Sheet (Total Assets, Equity, Cash, Debt)

Output: data/fundamentals_YYYYMMDD.csv
"""

import os
import sys
import time
import pandas as pd
import tushare as ts
from datetime import datetime
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
                print(f"Failed after retries: {e}")
                return None
            else:
                time.sleep(1)
    return None

def get_latest_trade_date(pro):
    today = datetime.now().strftime('%Y%m%d')
    df = _ts_call(pro.trade_cal, exchange='SSE', is_open='1', end_date=today, limit=1, fields='cal_date')
    return df['cal_date'].iloc[0] if df is not None and not df.empty else today

def get_financial_period(pro):
    # Fetch a major bank to check latest report period
    # Try a few major stocks to be sure
    for code in ['601398.SH', '600519.SH', '000001.SZ']:
        try:
            df = _ts_call(pro.balancesheet, ts_code=code, limit=1, fields='end_date')
            if df is not None and not df.empty:
                return df['end_date'].iloc[0]
        except:
            continue
            
    # Fallback logic
    now = datetime.now()
    year = now.year
    md = now.month * 100 + now.day
    if md < 501: return f"{year-1}1231" # Annual
    if md < 901: return f"{year}0331" # Q1
    if md < 1101: return f"{year}0630" # Semi-Annual
    return f"{year}0930" # Q3

def get_period_offset(period, years_offset):
    """Calculate period string for N years ago."""
    try:
        p_date = datetime.strptime(period, "%Y%m%d")
        new_year = p_date.year - years_offset
        return f"{new_year}{p_date.month:02d}{p_date.day:02d}"
    except:
        return None

def get_trade_date_offset(pro, date_str, years_offset=1):
    """Get the nearest trading date N years ago."""
    try:
        d = datetime.strptime(date_str, "%Y%m%d")
        target_d = d.replace(year=d.year - years_offset)
        target_str = target_d.strftime('%Y%m%d')
        
        # Find nearest trade date (look back 10 days)
        # Tushare trade_cal returns sorted by date desc by default
        df = _ts_call(pro.trade_cal, exchange='SSE', is_open='1', end_date=target_str, limit=1, fields='cal_date')
        if df is not None and not df.empty:
            return df['cal_date'].iloc[0]
    except Exception as e:
        print(f"Error calculating offset date: {e}")
    return None

def batch_fetch(pro, api_func, ts_codes, period, fields, batch_size=50):
    results = []
    total = len(ts_codes)
    func_name = getattr(api_func, '__name__', str(api_func))
    print(f"Fetching {func_name} for {total} stocks...")
    
    for i in range(0, total, batch_size):
        batch = ts_codes[i:i+batch_size]
        codes_str = ",".join(batch)
        try:
            df = _ts_call(api_func, ts_code=codes_str, period=period, fields=fields)
            if df is not None and not df.empty:
                results.append(df)
            else:
                # Debug: Print if empty for first few batches
                if i < 100:
                    print(f"  Batch {i} returned empty.")
        except Exception as e:
            print(f"Error batch {i}: {e}")
        
        if (i // batch_size) % 10 == 0:
            print(f"  Progress: {min(i+batch_size, total)}/{total}")
            
    if results:
        return pd.concat(results, ignore_index=True).drop_duplicates(subset=['ts_code'], keep='first')
    
    if fields:
        cols = fields.split(',')
        return pd.DataFrame(columns=cols)
    return pd.DataFrame()

def main():
    pro = get_pro()
    
    print("1. Getting Trade Date & Financial Period...")
    trade_date = get_latest_trade_date(pro)
    period = get_financial_period(pro)
    print(f"Trade Date: {trade_date}, Financial Period: {period}")
    
    print("2. Fetching Stock List...")
    df_basic = _ts_call(pro.stock_basic, exchange='', list_status='L', fields='ts_code,symbol,name,area,industry,market,list_date')
    if df_basic is None or df_basic.empty:
        print("Error: Failed to fetch stock list.")
        return
    ts_codes = df_basic['ts_code'].tolist()
    print(f"Total Stocks: {len(ts_codes)}")
    
    print("3. Fetching Daily Basic (Valuation)...")
    # daily_basic doesn't support batch by ts_code well for all history, but for specific date it returns all.
    # So we just query by trade_date.
    # Added dv_ttm for rolling dividend yield
    df_daily = _ts_call(pro.daily_basic, trade_date=trade_date, fields='ts_code,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_mv,circ_mv')
    
    if df_daily is None or df_daily.empty:
        print("Error: Failed to fetch daily basic data.")
        return

    # Use dv_ttm if available, otherwise fallback to dv_ratio
    if 'dv_ttm' in df_daily.columns:
        df_daily['dv_ratio'] = df_daily['dv_ttm'].fillna(df_daily['dv_ratio'])
    
    print("3.1 Fetching Daily Basic 1 Year Ago (for Strict TTM Growth)...")
    trade_date_1y = get_trade_date_offset(pro, trade_date, 1)
    print(f"  Trade Date 1Y Ago: {trade_date_1y}")
    
    df_daily_1y = pd.DataFrame()
    if trade_date_1y:
        df_daily_1y = _ts_call(pro.daily_basic, trade_date=trade_date_1y, fields='ts_code,close,pe_ttm')
        if df_daily_1y is not None and not df_daily_1y.empty:
            df_daily_1y = df_daily_1y.rename(columns={'close': 'close_1y', 'pe_ttm': 'pe_ttm_1y'})
        else:
            df_daily_1y = pd.DataFrame()

    print("3.2 Fetching Daily Basic 3 Years Ago (for 3Y CAGR)...")
    trade_date_3y = get_trade_date_offset(pro, trade_date, 3)
    print(f"  Trade Date 3Y Ago: {trade_date_3y}")
    
    df_daily_3y = pd.DataFrame()
    if trade_date_3y:
        df_daily_3y = _ts_call(pro.daily_basic, trade_date=trade_date_3y, fields='ts_code,close,pe_ttm')
        if df_daily_3y is not None and not df_daily_3y.empty:
            df_daily_3y = df_daily_3y.rename(columns={'close': 'close_3y', 'pe_ttm': 'pe_ttm_3y'})
        else:
            df_daily_3y = pd.DataFrame()

    print("4. Fetching Financial Indicators (Ratios)...")
    # fields: roe, roa, gross_margin, net_profit_margin (net_income_of_total_revenue?), debt_to_assets, current_ratio
    # Tushare fields: 
    # roe, roe_dt (deducted), roa
    # gross_margin, net_profit_margin is not direct, use 'profit_to_gr' (Net Profit / Operating Revenue)? No, 'net_profit_margin' exists in some docs but let's check.
    # fina_indicator fields: 'gross_margin', 'net_profit_margin' might not be exact names.
    # 'grossprofit_margin', 'netprofit_margin'?
    # Let's use standard ones: 'gross_margin', 'net_profit_to_total_revenue' (net_income_of_total_revenue)
    # 'debt_to_assets', 'current_ratio', 'quick_ratio'
    # 'eps', 'bps'
    # Note: Tushare uses 'profit_to_gr' for Net Profit Margin usually.
    # We include 'gross_margin' but will validate it.
    # Added 'ca_turn' (Current Asset Turnover) to derive Current Assets.
    # Added 'tr_yoy' (Revenue Growth YoY) and 'netprofit_yoy' (Net Profit Growth YoY)
    fina_fields = 'ts_code,eps,dt_eps,bps,roe,roe_dt,roa,gross_margin,profit_to_gr,debt_to_assets,current_ratio,quick_ratio,ocf_to_debt,ca_turn,tr_yoy,netprofit_yoy'
    
    # fina_indicator supports batching well
    df_fina = batch_fetch(pro, pro.fina_indicator, ts_codes, period, fina_fields, batch_size=50)
    
    # Rename profit_to_gr to net_profit_margin if present
    if 'profit_to_gr' in df_fina.columns:
        df_fina = df_fina.rename(columns={'profit_to_gr': 'net_profit_margin'})
    
    print("5. Fetching Historical EPS (for Growth Calculation)...")
    # Fetch EPS from 3 years ago (for 3Y CAGR)
    # Note: For 3Y Growth, we still use the reporting period EPS (CAGR of cumulative EPS)
    # as it is a standard way to measure long term growth.
    period_3y_ago = get_period_offset(period, 3)
    print(f"  Fetching EPS for {period_3y_ago} (3 years ago)...")
    
    # Use smaller batch size to avoid timeouts/errors
    df_eps_3y = batch_fetch(pro, pro.fina_indicator, ts_codes, period_3y_ago, 'ts_code,eps', batch_size=50)
    if not df_eps_3y.empty:
        df_eps_3y = df_eps_3y.rename(columns={'eps': 'eps_3y_ago'})
        print(f"  Fetched {len(df_eps_3y)} records for 3Y ago EPS.")
    else:
        print("  Warning: No data fetched for 3Y ago EPS.")
    
    print("6. Merging Data...")
    # Base is stock_basic
    df = df_basic.merge(df_daily, on='ts_code', how='left')
    df = df.merge(df_fina, on='ts_code', how='left')
    
    if not df_daily_1y.empty:
        df = df.merge(df_daily_1y, on='ts_code', how='left')

    if not df_daily_3y.empty:
        df = df.merge(df_daily_3y, on='ts_code', how='left')

    if not df_eps_3y.empty:
        df = df.merge(df_eps_3y, on='ts_code', how='left')

    # --- Derived Metrics ---
    print("7. Calculating Derived Metrics...")
    
    # Ensure numeric types
    numeric_cols = ['total_mv', 'pe', 'pe_ttm', 'pb', 'ps', 'ps_ttm', 'gross_margin']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    # 1. Revenue (Derived) = Market Cap / PS
    # Use ps_ttm if available, else ps
    df['ps_used'] = df['ps_ttm'].fillna(df['ps'])
    df['total_revenue'] = df['total_mv'] * 10000 / df['ps_used'] # total_mv is in 10k
    
    # 2. Net Income (Derived) = Market Cap / PE
    # Use pe_ttm if available, else pe
    df['pe_used'] = df['pe_ttm'].fillna(df['pe'])
    df['n_income_attr_p'] = df['total_mv'] * 10000 / df['pe_used']
    
    # 3. Gross Margin Validation
    # If gross_margin > 100, it might be absolute value (Gross Profit).
    # If so, recalculate percentage: Gross Profit / Revenue * 100
    mask_abs_gm = df['gross_margin'] > 100
    if mask_abs_gm.any():
        print(f"  Found {mask_abs_gm.sum()} rows with absolute Gross Margin. Converting to %...")
        df.loc[mask_abs_gm, 'gross_margin'] = (df.loc[mask_abs_gm, 'gross_margin'] / df.loc[mask_abs_gm, 'total_revenue']) * 100
        
    # 4. EPS and BPS (Derived fallback)
    # If eps/bps from fina_indicator is missing, derive from PE/PB
    if 'eps' not in df.columns: df['eps'] = None
    if 'bps' not in df.columns: df['bps'] = None
    
    df['eps_derived'] = df['close'] / df['pe_used']
    df['bps_derived'] = df['close'] / df['pb']
    
    df['eps'] = df['eps'].fillna(df['eps_derived'])
    df['bps'] = df['bps'].fillna(df['bps_derived'])
    
    # 5. NCAV (Approximate)
    # We derive Total Liabilities and Current Assets to calculate NCAV.
    # Total Shares = Total MV * 10000 / Close
    df['total_shares'] = df['total_mv'] * 10000 / df['close']
    
    # 6. Calculate Growth Rates
    # 6.1 EPS Growth (TTM) - Strict TTM Calculation
    # Current TTM EPS = Close / PE_TTM
    # Prior TTM EPS = Close_1Y / PE_TTM_1Y
    
    # Ensure numeric
    for col in ['pe_ttm', 'close', 'pe_ttm_1y', 'close_1y']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Calculate TTM EPS
    # Avoid division by zero if PE is 0 or NaN
    df['eps_ttm_current'] = df.apply(lambda x: x['close'] / x['pe_ttm'] if (pd.notnull(x['pe_ttm']) and x['pe_ttm'] != 0) else None, axis=1)
    df['eps_ttm_1y'] = df.apply(lambda x: x['close_1y'] / x['pe_ttm_1y'] if (pd.notnull(x['pe_ttm_1y']) and x['pe_ttm_1y'] != 0) else None, axis=1)
    
    # Calculate Growth
    mask_valid_ttm = (df['eps_ttm_1y'].abs() > 0.001)
    df.loc[mask_valid_ttm, 'eps_growth_ttm'] = (
        (df.loc[mask_valid_ttm, 'eps_ttm_current'] - df.loc[mask_valid_ttm, 'eps_ttm_1y']) / 
        df.loc[mask_valid_ttm, 'eps_ttm_1y'].abs()
    ) * 100
    
    # Fallback to netprofit_yoy if calculation failed or data missing
    df['eps_growth_ttm'] = df['eps_growth_ttm'].fillna(df['netprofit_yoy'])

    # 6.2 EPS Growth (3-Year CAGR)
    # Prefer TTM EPS for 3Y Growth if available, as it handles seasonality and is more robust.
    # Calculate TTM EPS for 3 years ago
    if 'close_3y' in df.columns and 'pe_ttm_3y' in df.columns:
        df['close_3y'] = pd.to_numeric(df['close_3y'], errors='coerce')
        df['pe_ttm_3y'] = pd.to_numeric(df['pe_ttm_3y'], errors='coerce')
        df['eps_ttm_3y'] = df.apply(lambda x: x['close_3y'] / x['pe_ttm_3y'] if (pd.notnull(x['pe_ttm_3y']) and x['pe_ttm_3y'] != 0) else None, axis=1)
    else:
        df['eps_ttm_3y'] = None

    # Use TTM EPS for 3Y Growth if both current and 3Y ago TTM EPS are available
    mask_valid_3y_ttm = (df['eps_ttm_current'] > 0) & (df['eps_ttm_3y'] > 0)
    
    df.loc[mask_valid_3y_ttm, 'eps_growth_3y'] = (
        (df.loc[mask_valid_3y_ttm, 'eps_ttm_current'] / df.loc[mask_valid_3y_ttm, 'eps_ttm_3y']) ** (1/3) - 1
    ) * 100

    # Fallback to Reported EPS if TTM not available
    # CAGR = (Ending Value / Beginning Value) ^ (1 / n) - 1
    if 'eps' in df.columns and 'eps_3y_ago' in df.columns:
        df['eps'] = pd.to_numeric(df['eps'], errors='coerce')
        df['eps_3y_ago'] = pd.to_numeric(df['eps_3y_ago'], errors='coerce')
        
        # Only calculate if both are positive to avoid complex numbers or invalid growth logic
        # And only if not already calculated by TTM
        mask_valid_3y_reported = (df['eps'] > 0) & (df['eps_3y_ago'] > 0) & (df['eps_growth_3y'].isna())
        
        df.loc[mask_valid_3y_reported, 'eps_growth_3y'] = (
            (df.loc[mask_valid_3y_reported, 'eps'] / df.loc[mask_valid_3y_reported, 'eps_3y_ago']) ** (1/3) - 1
        ) * 100
    
    # If still None, leave as None

    # Equity (Parent) = BPS * Shares
    df['total_equity'] = df['bps'] * df['total_shares']
    
    # Total Assets = Equity / (1 - debt_to_assets%)
    # debt_to_assets is in %, so divide by 100
    if 'debt_to_assets' in df.columns:
        df['debt_ratio'] = pd.to_numeric(df['debt_to_assets'], errors='coerce') / 100
        # Avoid division by zero or negative equity implication
        # If debt_ratio >= 1, equity would be <= 0, which is possible for distressed firms.
        # But formula Equity / (1 - D/A) blows up if D/A -> 1.
        # Let's cap D/A at 0.99 for calculation stability or handle separately.
        df['total_assets_derived'] = df['total_equity'] / (1 - df['debt_ratio'])
        df['total_liab'] = df['total_assets_derived'] * df['debt_ratio']
    else:
        df['total_liab'] = None

    # Current Assets = Revenue / ca_turn
    # ca_turn is Current Asset Turnover.
    if 'ca_turn' in df.columns:
        df['ca_turn'] = pd.to_numeric(df['ca_turn'], errors='coerce')
        # Avoid division by zero
        df['total_cur_assets'] = df.apply(
            lambda row: row['total_revenue'] / row['ca_turn'] if (pd.notnull(row['ca_turn']) and row['ca_turn'] > 0) else None, 
            axis=1
        )
    else:
        df['total_cur_assets'] = None
    
    # Add Period Column
    df['report_period'] = period
    
    # Save
    out_dir = Path('data')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'fundamentals_{trade_date}.csv'
    df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"Saved comprehensive fundamentals to {out_path}")

if __name__ == "__main__":
    main()
