import streamlit as st
import pandas as pd
import plotly.express as px
import subprocess
import sys
import io
import os
from pathlib import Path
from datetime import datetime
from app.agents.analyst import stock_analysis_agent
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="A-Share Scanner", page_icon="🔍", layout="wide")

def create_pdf(company_info, analysis_text):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Register Chinese Font
    font_path = "C:\\Windows\\Fonts\\simhei.ttf"
    font_name = 'Helvetica' # Default fallback
    
    if os.path.exists(font_path):
        try:
            pdfmetrics.registerFont(TTFont('SimHei', font_path))
            font_name = 'SimHei'
        except Exception as e:
            print(f"Failed to load Chinese font: {e}")
    
    # Create custom style for Chinese
    style_normal = ParagraphStyle(
        'ChineseNormal',
        parent=styles['Normal'],
        fontName=font_name,
        fontSize=10,
        leading=14,
        spaceAfter=10,
        wordWrap='CJK' # Important for Chinese line breaking
    )
    
    style_heading = ParagraphStyle(
        'ChineseHeading',
        parent=styles['Heading1'],
        fontName=font_name,
        fontSize=16,
        leading=20,
        spaceAfter=12
    )

    story = []
    
    # Title
    story.append(Paragraph(f"Company Analysis: {company_info.get('name', '')} ({company_info.get('ts_code', '')})", style_heading))
    story.append(Spacer(1, 12))
    
    # Basic Info
    info_text = f"""
    <b>Industry:</b> {company_info.get('industry', '')}<br/>
    <b>Price:</b> {company_info.get('close', '')}<br/>
    <b>PE (TTM):</b> {company_info.get('pe_ttm', '')}<br/>
    <b>PB:</b> {company_info.get('pb', '')}<br/>
    <b>ROE:</b> {company_info.get('roe', '')}%<br/>
    """
    story.append(Paragraph(info_text, style_normal))
    story.append(Spacer(1, 12))
    
    # Analysis
    if analysis_text:
        story.append(Paragraph("Analyst Agent Report", style_heading))
        story.append(Spacer(1, 12))
        
        # Split by paragraphs
        paragraphs = analysis_text.split('\n')
        for p in paragraphs:
            if not p.strip():
                continue
                
            # Basic formatting
            # Bold **text** -> <b>text</b>
            parts = p.split('**')
            formatted_p = ""
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    formatted_p += f"<b>{part}</b>"
                else:
                    formatted_p += part
            
            story.append(Paragraph(formatted_p, style_normal))
            story.append(Spacer(1, 6))

    doc.build(story)
    buffer.seek(0)
    return buffer

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
    - **Intrinsic Value**: 内在价值 (成长型)，Calculated as $EPS \times (8.5 + 2g)$。其中 $g$ 为预期增长率 (Expected Annual Growth Rate of EPS)，可在左侧边栏调整。格雷厄姆原意指未来7-10年的平均增长率，本系统默认使用分析师对当年的EPS增长预期作为参考。
    - **NCAV/Share**: 每股净流动资产价值，Calculated as $(Current Assets - Total Liabilities) / Total Shares$。深度价值投资指标，股价低于此值通常被认为是极度低估。
      > **⚠️ 注意**: 银行及部分金融类公司因会计准则差异（不区分流动/非流动资产），无法计算 NCAV，该指标会显示为 N/A。
    - **Price/Graham**: 股价与格雷厄姆数值的比率。小于1表示股价低于格雷厄姆数值。
    - **Price/NCAV**: 股价与NCAV的比率。小于1表示股价低于净流动资产价值。

    ### 盈利能力 (Profitability)
    - **ROE %**: 加权净资产收益率，净利润 / 净资产。衡量公司运用自有资本的效率。
    - **Net Margin %**: 净利率，净利润 / 营业收入。衡量每一元收入能带来多少净利润。
    - **Gross Margin %**: 毛利率，(营业收入 - 营业成本) / 营业收入。反映产品或服务的直接盈利能力。

    ### 成长与预期 (Growth & Forecasts)
    - **EPS (TTM)**: 每股收益 (滚动)，最近12个月的每股净利润。
    - **EPS Growth (TTM) %**: 每股收益同比增长率 (滚动)，(当前TTM EPS - 去年同期TTM EPS) / |去年同期TTM EPS|。反映剔除季节性后的真实每股增长趋势。
    - **EPS Growth (3Y) %**: 每股收益3年复合增长率 (CAGR)，反映过去3年的长期增长趋势。
    - **Rev Growth %**: 营业收入同比增长率 (Year-over-Year)。
    - **Profit Growth %**: 净利润同比增长率 (Year-over-Year)。

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

# --- Intrinsic Value Settings ---
st.sidebar.markdown("#### 🧠 Intrinsic Value Settings")

# Ensure columns exist
for col in ['tr_yoy', 'netprofit_yoy', 'eps_growth_ttm', 'eps_growth_3y']:
    if col not in df.columns:
        df[col] = None
        
g_source = st.sidebar.selectbox(
    "Growth Rate Source", 
    ["TTM Growth Rate", "Historical 3-Year Growth Rate", "Manual Input"],
    index=1,
    help="Select the source for 'g' in Intrinsic Value = EPS * (8.5 + 2g)"
)

if g_source == "Manual Input":
    growth_rate = st.sidebar.slider("Expected Growth Rate (g) %", 0.0, 30.0, 0.0)
    df['calc_growth_rate'] = growth_rate
elif g_source == "TTM Growth Rate":
    # Use eps_growth_ttm if available, else fallback to netprofit_yoy
    if 'eps_growth_ttm' in df.columns:
        df['calc_growth_rate'] = pd.to_numeric(df['eps_growth_ttm'], errors='coerce').fillna(0)
    else:
        df['calc_growth_rate'] = pd.to_numeric(df['netprofit_yoy'], errors='coerce').fillna(0)
    st.sidebar.caption("Using 'EPS Growth (TTM)'.")
elif g_source == "Historical 3-Year Growth Rate":
    if 'eps_growth_3y' in df.columns:
        df['calc_growth_rate'] = pd.to_numeric(df['eps_growth_3y'], errors='coerce').fillna(0)
        st.sidebar.caption("Using 'EPS Growth (3-Year CAGR)'.")
    else:
        # Fallback
        df['calc_growth_rate'] = pd.to_numeric(df['netprofit_yoy'], errors='coerce').fillna(0)
        st.sidebar.warning("⚠️ 3-Year Growth data missing. Using Last Year Profit Growth as proxy.")

# Calculate Intrinsic Value dynamically
# V = EPS * (8.5 + 2g)
df['intrinsic_value'] = df['eps'] * (8.5 + 2 * df['calc_growth_rate'])
df['price_to_intrinsic'] = df['close'] / df['intrinsic_value']

st.sidebar.markdown("---")

# --- Sidebar Filters ---
st.sidebar.header("Filters")

# Reset Filters Button
if st.sidebar.button("🔄 Reset / Show All", help="Clear all filters to show all data"):
    st.session_state.filter_industry = []
    st.session_state.filter_search = ""
    st.session_state.filter_enable_mv = False
    st.session_state.filter_enable_pe = False
    st.session_state.filter_enable_pb = False
    st.session_state.filter_enable_graham = False
    st.session_state.filter_enable_iv = False
    st.session_state.filter_enable_ncav = False
    st.session_state.filter_enable_roe = False
    st.session_state.filter_enable_dv = False
    st.rerun()

# Industry Filter
industries = sorted(df['industry'].dropna().unique())
selected_industries = st.sidebar.multiselect("Industry", industries, key="filter_industry")

# Search Filter
search_term = st.sidebar.text_input("Search", placeholder="Code or Name (e.g. 000001 or 平安)", key="filter_search")

# --- Advanced Filters (Collapsible) ---
with st.sidebar.expander("💰 Valuation & Size Filters", expanded=True):
    # Market Cap
    enable_mv = st.checkbox("Filter by Market Cap", key="filter_enable_mv")
    min_mv = int(df['total_mv'].min() / 10000)
    max_mv = int(df['total_mv'].max() / 10000)
    if enable_mv:
        mv_range = st.slider("Market Cap (Billion CNY)", min_mv, max_mv, (min_mv, max_mv), key="filter_mv_range")

    # PE
    enable_pe = st.checkbox("Filter by PE (TTM)", key="filter_enable_pe")
    if enable_pe:
        pe_range = st.slider("PE Range", -200.0, 200.0, (0.0, 50.0), key="filter_pe_range")
    
    # PB
    enable_pb = st.checkbox("Filter by PB", key="filter_enable_pb")
    if enable_pb:
        pb_range = st.slider("PB Range", -10.0, 20.0, (0.0, 5.0), key="filter_pb_range")

    # Graham
    enable_graham = st.checkbox("Filter by Price/Graham", value=True, key="filter_enable_graham")
    if enable_graham:
        pg_range = st.slider("Price/Graham", 0.0, 5.0, (0.0, 1.0), key="filter_pg_range")

    # Intrinsic Value Filter
    enable_iv = st.checkbox("Filter by Price/Intrinsic Value", value=True, key="filter_enable_iv")
    if enable_iv:
        piv_range = st.slider("Price/Intrinsic Value", 0.0, 5.0, (0.0, 1.0), key="filter_piv_range")
        
    # NCAV
    enable_ncav = st.checkbox("Filter by Price/NCAV", value=True, key="filter_enable_ncav")
    if enable_ncav:
        pncav_range = st.slider("Price/NCAV", 0.0, 5.0, (0.0, 1.0), key="filter_pncav_range")

with st.sidebar.expander("📈 Profitability Filters", expanded=True):
    # ROE
    enable_roe = st.checkbox("Filter by ROE", value=True, key="filter_enable_roe")
    if enable_roe:
        roe_range = st.slider("ROE %", -100.0, 100.0, (0.0, 30.0), key="filter_roe_range")
        
    # Div Yield
    enable_dv = st.checkbox("Filter by Div Yield", key="filter_enable_dv")
    if enable_dv:
        dv_range = st.slider("Div Yield %", 0.0, 20.0, (0.0, 10.0), key="filter_dv_range")

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

# Price/Intrinsic Value
if enable_iv:
    filtered_df = filtered_df[
        (filtered_df['price_to_intrinsic'] >= piv_range[0]) & 
        (filtered_df['price_to_intrinsic'] <= piv_range[1])
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

# Ensure new columns exist (for compatibility with old data)
for col in ['tr_yoy', 'netprofit_yoy', 'eps_growth_ttm', 'eps_growth_3y', 'eps_ttm_current']:
    if col not in filtered_df.columns:
        filtered_df[col] = None

# Display Columns
display_cols = [
    'ts_code', 'name', 'industry', 'report_period', 'close', 
    'pe_ttm', 'pb', 'dv_ratio', 
    'graham_number', 'price_to_graham', 
    'intrinsic_value', 'price_to_intrinsic',
    'ncav_per_share', 'price_to_ncav',
    'eps_ttm_current', 'eps_growth_ttm', 'eps_growth_3y',
    'tr_yoy', 'netprofit_yoy', 
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
        "intrinsic_value": "Intrinsic Value",
        "price_to_intrinsic": "Price/Intrinsic",
        "ncav_per_share": "NCAV/Share",
        "price_to_ncav": "Price/NCAV",
        "eps_ttm_current": "EPS (TTM)",
        "eps_growth_ttm": "EPS Growth (TTM) %",
        "eps_growth_3y": "EPS Growth (3Y) %",
        "tr_yoy": "Rev Growth %",
        "netprofit_yoy": "Profit Growth %",
        "total_mv": "Market Cap (B)",
        "net_profit_margin": "Net Margin %",
        "total_revenue": "Revenue (B)",
        "n_income_attr_p": "Net Profit (B)"
    },
    width="stretch",
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

    st.markdown("#### 💎 Intrinsic Value (Growth Model)")
    iv_cols = st.columns(4)
    with iv_cols[0]:
        iv = round(company['intrinsic_value'], 2) if pd.notnull(company['intrinsic_value']) else "N/A"
        st.metric("Intrinsic Value", iv, help="Calculated as EPS * (8.5 + 2g)")
    with iv_cols[1]:
        piv = round(company['price_to_intrinsic'], 2) if pd.notnull(company['price_to_intrinsic']) else "N/A"
        st.metric("Price / Intrinsic", piv)
    with iv_cols[2]:
        g_used = round(company['calc_growth_rate'], 2) if pd.notnull(company['calc_growth_rate']) else "N/A"
        st.metric("Growth Rate Used", f"{g_used}%")
    with iv_cols[3]:
        st.caption(f"Source: {g_source}")

    st.markdown("#### Financial Health")
    health_data = {
        "Metric": ["Current Ratio", "Quick Ratio", "Gross Margin", "Net Margin", "EPS", "BPS"],
        "Value": [
            str(round(company['current_ratio'], 2)),
            str(round(company['quick_ratio'], 2)),
            f"{round(company['gross_margin'], 2)}%",
            f"{round(company['net_profit_margin'], 2)}%",
            str(company['eps']),
            str(company['bps'])
        ]
    }
    st.table(pd.DataFrame(health_data).astype(str))

    # --- All Indicators ---
    with st.expander("📋 View All Indicators (Raw Data)", expanded=True):
        # Transpose to show as Key-Value pairs
        st.dataframe(company.to_frame(name="Value").astype(str), use_container_width=True, height=500)

    # --- Price History Chart ---
    st.markdown("#### 📈 Price History (Last 1 Year)")
    
    @st.cache_data(ttl=3600)
    def fetch_price_history(ts_code):
        import tushare as ts
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        token = os.getenv("TUSHARE_TOKEN")
        if not token:
            return None
            
        ts.set_token(token)
        # pro = ts.pro_api() # Not needed for ts.pro_bar
        
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=365)
        
        try:
            # Use ts.pro_bar to get forward-adjusted prices (adj='qfq')
            df_price = ts.pro_bar(
                ts_code=ts_code, 
                adj='qfq',
                start_date=start_date.strftime('%Y%m%d'), 
                end_date=end_date.strftime('%Y%m%d')
            )
            if df_price is not None and not df_price.empty:
                df_price['trade_date'] = pd.to_datetime(df_price['trade_date'])
                return df_price.sort_values('trade_date')
        except Exception as e:
            st.error(f"Error fetching price history: {e}")
            return None
        return None

    with st.spinner("Fetching price history (Forward Adjusted)..."):
        df_history = fetch_price_history(selected_code)
        
    if df_history is not None and not df_history.empty:
        fig = px.line(
            df_history, 
            x='trade_date', 
            y='close', 
            title=f"{company['name']} ({selected_code}) - Daily Close Price (Forward Adjusted)",
            labels={'trade_date': 'Date', 'close': 'Price (CNY)'}
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Could not fetch price history data.")

    st.markdown("---")
    st.subheader("🧐 Analyst Agent 解读")
    st.caption("点击后，Analyst Agent 会基于上方报告给出要点总结与风险提示。（生成过程可能需要 3-5 分钟，请耐心等待）")

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}

    current_analysis = st.session_state.analysis_results.get(selected_code)

    col_btn, col_opt = st.columns([3, 1])
    with col_opt:
        force_update = st.checkbox("强制更新", key="force_update_scanner", help="忽略缓存，重新生成分析结果")

    with col_btn:
        if st.button("生成解读", use_container_width=True):
            # Convert company series to dict
            company_dict = company.to_dict()
            analysis_text = stock_analysis_agent(company_dict, df_history, force_update=force_update)
            if analysis_text:
                st.session_state.analysis_results[selected_code] = analysis_text
                st.rerun()

    if current_analysis:
        with st.expander("🧐 Analyst Agent (Cached)", expanded=True):
            st.markdown(current_analysis)
        
        pdf_buffer = create_pdf(company.to_dict(), current_analysis)
        st.download_button(
            label="📄 Download Analysis PDF",
            data=pdf_buffer,
            file_name=f"{selected_code}_analysis.pdf",
            mime="application/pdf",
            use_container_width=True
        )

