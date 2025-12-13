import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
from pathlib import Path
from datetime import datetime

st.set_page_config(page_title="A-Share Scanner", page_icon="🔍", layout="wide")

st.title("🔍 A-Share Market Scanner")

# --- Update Data Button ---
col_title, col_update = st.columns([3, 1])
with col_title:
    st.markdown("Comprehensive fundamental data analysis and screening for A-Share companies.")

with col_update:
    if st.button("🔄 Update Data Now"):
        with st.spinner("Fetching latest data from Tushare... This may take a few minutes."):
            try:
                # Run the fetch script
                script_path = Path("scripts/fetch_all_fundamentals.py")
                python_executable = sys.executable
                result = subprocess.run(
                    [python_executable, str(script_path)], 
                    capture_output=True, 
                    text=True,
                    check=True
                )
                st.success("Data updated successfully!")
                st.cache_data.clear() # Clear cache to reload new data
                st.rerun()
            except subprocess.CalledProcessError as e:
                st.error(f"Update failed: {e.stderr}")
            except Exception as e:
                st.error(f"An error occurred: {e}")

# --- Indicator Explanations ---
with st.expander("ℹ️ 指标解释 (Indicator Definitions)"):
    st.markdown("""
    ### 估值指标 (Valuation)
    - **PE (TTM)**: 市盈率 (滚动)，股价 / 最近12个月每股收益。衡量估值高低，越低通常越便宜。
    - **PB**: 市净率 (最新)，股价 / 每股净资产。衡量股价相对于净资产的溢价。
    - **Div Yield (TTM) %**: 股息率 (滚动)，过去12个月每股股息 / 股价。衡量现金分红回报率。
    - **Graham Num**: 格雷厄姆数值，即 $\\sqrt{22.5 \\times EPS \\times BVPS}$。源自格雷厄姆的防御型投资标准（PE<15 且 PB<1.5，乘积为 22.5）。股价低于此值即视为具有安全边际。
    - **NCAV/Share**: 每股净流动资产价值，Calculated as $(Current Assets - Total Liabilities) / Total Shares$。深度价值投资指标，股价低于此值通常被认为是极度低估。
      > **⚠️ 注意**: 银行及部分金融类公司因会计准则差异（不区分流动/非流动资产），无法计算 NCAV，该指标会显示为 N/A。
    - **Price/Graham**: 股价与格雷厄姆数值的比率。小于1表示股价低于格雷厄姆数值。
    - **Price/NCAV**: 股价与NCAV的比率。小于1表示股价低于净流动资产价值。

    ### 盈利能力 (Profitability)
    - **ROE %**: 加权净资产收益率，净利润 / 净资产。衡量公司运用自有资本的效率。
    - **Net Margin %**: 净利率，净利润 / 营业收入。衡量每一元收入能带来多少净利润。
    - **Gross Margin %**: 毛利率，(营业收入 - 营业成本) / 营业收入。反映产品或服务的直接盈利能力。

    ### 财务数据 (Financials)
    - **Market Cap (B)**: 总市值 (亿元)。
    - **Revenue (B)**: 营业收入 (亿元)。*注：部分数据可能由市值和PS估算得出。*
    - **Net Profit (B)**: 净利润 (亿元)。*注：部分数据可能由市值和PE估算得出。*
    - **Report Period**: 数据来源的财报期 (YYYYMMDD)。
    """)

# --- Data Loading ---
def get_latest_file_info():
    data_dir = Path("data")
    files = list(data_dir.glob("fundamentals_*.csv"))
    if not files:
        return None, 0
    latest_file = sorted(files)[-1]
    return latest_file, latest_file.stat().st_mtime

@st.cache_data
def load_data(file_path, mtime):
    if file_path is None:
        return None, None
        
    # Extract date from filename
    try:
        file_date_str = file_path.stem.split('_')[-1]
        file_date = datetime.strptime(file_date_str, "%Y%m%d").strftime("%Y-%m-%d")
    except:
        file_date = "Unknown"
        
    df = pd.read_csv(file_path)
    
    # Calculate Net Profit Margin if missing
    if 'net_profit_margin' not in df.columns:
        if 'n_income_attr_p' in df.columns and 'total_revenue' in df.columns:
            df['net_profit_margin'] = (df['n_income_attr_p'] / df['total_revenue'] * 100).round(2)
        else:
            df['net_profit_margin'] = None

    # --- Graham & Value Metrics Calculation ---
    import numpy as np
    
    # 1. Graham Number = Sqrt(22.5 * EPS * BVPS)
    # Ensure eps and bps are numeric
    df['eps'] = pd.to_numeric(df['eps'], errors='coerce')
    df['bps'] = pd.to_numeric(df['bps'], errors='coerce')
    
    # Only calculate if EPS > 0 and BPS > 0
    mask_graham = (df['eps'] > 0) & (df['bps'] > 0)
    df.loc[mask_graham, 'graham_number'] = np.sqrt(22.5 * df.loc[mask_graham, 'eps'] * df.loc[mask_graham, 'bps'])
    
    # 2. NCAV Per Share = (Current Assets - Total Liabilities) / Total Shares
    # Total Shares = (Total MV * 10000) / Close
    # We use derived total_cur_assets and total_liab from the script if available
    if 'total_cur_assets' in df.columns and 'total_liab' in df.columns:
        # Ensure numeric
        df['total_cur_assets'] = pd.to_numeric(df['total_cur_assets'], errors='coerce')
        df['total_liab'] = pd.to_numeric(df['total_liab'], errors='coerce')
        
        df['total_shares'] = (df['total_mv'] * 10000) / df['close']
        
        # Calculate NCAV
        df['ncav_per_share'] = (df['total_cur_assets'] - df['total_liab']) / df['total_shares']
    else:
        df['ncav_per_share'] = None

    # 3. Price Ratios
    df['price_to_graham'] = df['close'] / df['graham_number']
    df['price_to_ncav'] = df['close'] / df['ncav_per_share']
            
    return df, file_date

latest_file, file_mtime = get_latest_file_info()
df, data_date = load_data(latest_file, file_mtime)

if df is None:
    st.error("No data found. Please run `scripts/fetch_all_fundamentals.py` first or click Update Data.")
    st.stop()

st.info(f"📅 Data Last Updated: **{data_date}**")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Industry Filter
industries = sorted(df['industry'].dropna().unique())
selected_industries = st.sidebar.multiselect("Industry", industries)

# Search Filter
search_term = st.sidebar.text_input("Search", placeholder="Code or Name (e.g. 000001 or 平安)")

# --- Advanced Filters (Collapsible) ---
with st.sidebar.expander("💰 Valuation & Size Filters", expanded=False):
    # Market Cap
    enable_mv = st.checkbox("Filter by Market Cap")
    min_mv = int(df['total_mv'].min() / 10000)
    max_mv = int(df['total_mv'].max() / 10000)
    if enable_mv:
        mv_range = st.slider("Market Cap (Billion CNY)", min_mv, max_mv, (min_mv, max_mv))

    # PE
    enable_pe = st.checkbox("Filter by PE (TTM)")
    if enable_pe:
        pe_range = st.slider("PE Range", -200.0, 200.0, (0.0, 50.0))
    
    # PB
    enable_pb = st.checkbox("Filter by PB")
    if enable_pb:
        pb_range = st.slider("PB Range", -10.0, 20.0, (0.0, 5.0))

    # Graham
    enable_graham = st.checkbox("Filter by Price/Graham")
    if enable_graham:
        pg_range = st.slider("Price/Graham", 0.0, 5.0, (0.0, 1.0))
        
    # NCAV
    enable_ncav = st.checkbox("Filter by Price/NCAV")
    if enable_ncav:
        pncav_range = st.slider("Price/NCAV", 0.0, 5.0, (0.0, 1.0))

with st.sidebar.expander("📈 Profitability Filters", expanded=False):
    # ROE
    enable_roe = st.checkbox("Filter by ROE")
    if enable_roe:
        roe_range = st.slider("ROE %", -100.0, 100.0, (0.0, 30.0))
        
    # Div Yield
    enable_dv = st.checkbox("Filter by Div Yield")
    if enable_dv:
        dv_range = st.slider("Div Yield %", 0.0, 20.0, (0.0, 10.0))

# --- Filtering Logic ---
filtered_df = df.copy()

# Search Filter
if search_term:
    filtered_df = filtered_df[
        filtered_df['ts_code'].str.contains(search_term, case=False, na=False) | 
        filtered_df['name'].str.contains(search_term, case=False, na=False)
    ]

if selected_industries:
    filtered_df = filtered_df[filtered_df['industry'].isin(selected_industries)]

# Market Cap
if enable_mv:
    filtered_df = filtered_df[
        (filtered_df['total_mv'] / 10000 >= mv_range[0]) & 
        (filtered_df['total_mv'] / 10000 <= mv_range[1])
    ]

# PE TTM
if enable_pe:
    filtered_df = filtered_df[
        (filtered_df['pe_ttm'] >= pe_range[0]) & 
        (filtered_df['pe_ttm'] <= pe_range[1])
    ]

# PB
if enable_pb:
    filtered_df = filtered_df[
        (filtered_df['pb'] >= pb_range[0]) & 
        (filtered_df['pb'] <= pb_range[1])
    ]

# Price/Graham
if enable_graham:
    filtered_df = filtered_df[
        (filtered_df['price_to_graham'] >= pg_range[0]) & 
        (filtered_df['price_to_graham'] <= pg_range[1])
    ]

# Price/NCAV
if enable_ncav:
    filtered_df = filtered_df[
        (filtered_df['price_to_ncav'] >= pncav_range[0]) & 
        (filtered_df['price_to_ncav'] <= pncav_range[1])
    ]

# ROE
if enable_roe:
    filtered_df = filtered_df[
        (filtered_df['roe'] >= roe_range[0]) & 
        (filtered_df['roe'] <= roe_range[1])
    ]

# Dividend Yield
if enable_dv:
    filtered_df = filtered_df[
        (filtered_df['dv_ratio'] >= dv_range[0]) & 
        (filtered_df['dv_ratio'] <= dv_range[1])
    ]

# --- Main Display ---
st.subheader(f"Filtered Results: {len(filtered_df)} Companies")

# Display Columns
display_cols = [
    'ts_code', 'name', 'industry', 'report_period', 'close', 
    'pe_ttm', 'pb', 'dv_ratio', 
    'graham_number', 'price_to_graham', 'ncav_per_share', 'price_to_ncav',
    'total_mv', 'roe', 'gross_margin', 'net_profit_margin',
    'total_revenue', 'n_income_attr_p'
]

# Format columns for display
display_df = filtered_df[display_cols].copy()
display_df['total_mv'] = (display_df['total_mv'] / 10000).round(2) # Billions
display_df['total_revenue'] = (display_df['total_revenue'] / 1e8).round(2) # Billions
display_df['n_income_attr_p'] = (display_df['n_income_attr_p'] / 1e8).round(2) # Billions
display_df = display_df.round(2)

st.dataframe(
    display_df,
    column_config={
        "ts_code": "Code",
        "name": "Name",
        "industry": "Industry",
        "report_period": "Report Period",
        "close": "Price",
        "pe_ttm": "PE (TTM)",
        "pb": "PB",
        "dv_ratio": "Div Yield (TTM) %",
        "graham_number": "Graham Num",
        "price_to_graham": "Price/Graham",
        "ncav_per_share": "NCAV/Share",
        "price_to_ncav": "Price/NCAV",
        "total_mv": "Market Cap (B)",
        "roe": "ROE %",
        "gross_margin": "Gross Margin %",
        "net_profit_margin": "Net Margin %",
        "total_revenue": "Revenue (B)",
        "n_income_attr_p": "Net Profit (B)"
    },
    use_container_width=True,
    height=600
)

# --- Detailed View ---
st.divider()
st.subheader("🏢 Company Details")

selected_code = st.selectbox("Select Company for Details", filtered_df['ts_code'].unique())

if selected_code:
    company = df[df['ts_code'] == selected_code].iloc[0]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Name", company['name'])
        st.metric("Industry", company['industry'])
        st.metric("Price", company['close'])
    with col2:
        st.metric("PE (TTM)", round(company['pe_ttm'], 2))
        st.metric("PB", round(company['pb'], 2))
        st.metric("Div Yield", f"{company['dv_ratio']}%")
    with col3:
        st.metric("ROE", f"{round(company['roe'], 2)}%")
        st.metric("ROA", f"{round(company['roa'], 2)}%")
        st.metric("Debt/Assets", f"{round(company['debt_to_assets'], 2)}%")
    with col4:
        st.metric("Revenue", f"{round(company['total_revenue']/1e8, 2)} B")
        st.metric("Net Income", f"{round(company['n_income_attr_p']/1e8, 2)} B")
        st.metric("Cash", f"{round(company['money_cap']/1e8, 2)} B")

    st.markdown("#### 🧠 Graham Valuation Metrics")
    graham_cols = st.columns(4)
    with graham_cols[0]:
        gn = round(company['graham_number'], 2) if pd.notnull(company['graham_number']) else "N/A"
        st.metric("Graham Number", gn)
    with graham_cols[1]:
        pg = round(company['price_to_graham'], 2) if pd.notnull(company['price_to_graham']) else "N/A"
        st.metric("Price / Graham", pg)
    with graham_cols[2]:
        ncav = round(company['ncav_per_share'], 2) if pd.notnull(company['ncav_per_share']) else "N/A"
        st.metric("NCAV / Share", ncav)
    with graham_cols[3]:
        pncav = round(company['price_to_ncav'], 2) if pd.notnull(company['price_to_ncav']) else "N/A"
        st.metric("Price / NCAV", pncav)

    st.markdown("#### Financial Health")
    health_data = {
        "Metric": ["Current Ratio", "Quick Ratio", "Gross Margin", "Net Margin", "EPS", "BPS"],
        "Value": [
            round(company['current_ratio'], 2),
            round(company['quick_ratio'], 2),
            f"{round(company['gross_margin'], 2)}%",
            f"{round(company['net_profit_margin'], 2)}%",
            company['eps'],
            company['bps']
        ]
    }
    st.table(pd.DataFrame(health_data))

