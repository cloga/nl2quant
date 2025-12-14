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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

st.set_page_config(page_title="A-Share Scanner", page_icon="🔍", layout="wide")

def create_pdf(company_info, analysis_text, chart_bytes=None):
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
    
    # Basic Info & Financials
    # Helper to safely format
    def fmt(val, suffix=""):
        if pd.isna(val) or val == "N/A": return "N/A"
        try:
            return f"{float(val):.2f}{suffix}"
        except:
            return str(val)

    info_text = f"""
    <b>Industry:</b> {company_info.get('industry', '')} &nbsp;&nbsp;&nbsp; <b>Price:</b> {company_info.get('close', '')}<br/>
    <b>PE (TTM):</b> {fmt(company_info.get('pe_ttm'))} &nbsp;&nbsp;&nbsp; <b>PB:</b> {fmt(company_info.get('pb'))}<br/>
    <b>ROE:</b> {fmt(company_info.get('roe'), '%')} &nbsp;&nbsp;&nbsp; <b>Div Yield:</b> {fmt(company_info.get('dv_ratio'), '%')}<br/>
    <br/>
    <b>Graham Number:</b> {fmt(company_info.get('graham_number'))} &nbsp;&nbsp;&nbsp; <b>Price/Graham:</b> {fmt(company_info.get('price_to_graham'))}<br/>
    <b>NCAV:</b> {fmt(company_info.get('ncav_per_share'))} &nbsp;&nbsp;&nbsp; <b>Price/NCAV:</b> {fmt(company_info.get('price_to_ncav'))}<br/>
    <br/>
    <b>EPS (TTM):</b> {fmt(company_info.get('eps_ttm_current'))} &nbsp;&nbsp;&nbsp; <b>EPS Growth (TTM):</b> {fmt(company_info.get('eps_growth_ttm'), '%')}<br/>
    """
    story.append(Paragraph(info_text, style_normal))
    story.append(Spacer(1, 12))

    # Price Chart
    if chart_bytes:
        story.append(Paragraph("Price Trend (1 Year)", style_heading))
        story.append(Spacer(1, 6))
        try:
            img = ReportLabImage(io.BytesIO(chart_bytes), width=450, height=250)
            story.append(img)
            story.append(Spacer(1, 12))
        except Exception as e:
            story.append(Paragraph(f"Error adding chart: {e}", style_normal))
    
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
    - **Intrinsic Value**: 内在价值 (成长型)，计算公式为 $V = EPS \times (8.5 + 2g)$。
        - **$g$ 的定义**: 预期年化增长率的**整数值**（如增长 10% 则 $g=10$）。
        - **默认来源**: 系统默认优先使用 **3年复合增长率 (3Y CAGR)** 以平滑短期波动。可在侧边栏切换为 TTM 增长率或手动输入。
        - **适用范围**: 该公式适用于增长率在 **0% - 15%** 之间的稳健型公司。
        - **⚠️ 局限性**: 
            1. **高增长失真**: 当 $g > 20$ 时，公式会给出极高的估值。**系统默认开启 "Cap Growth Rate at 15%" 选项，将 $g$ 限制在 15 以内以修正此问题。**
            2. **基数效应**: 若历史增长率（如 3年 CAGR）是基于低基数（如疫情期间）计算的恢复性高增长，直接套用会导致估值严重虚高。
            3. **建议**: 对于高增长或周期性反弹个股，建议在左侧选择 "Manual Input" 并输入保守的长期增长率（如 8-12%）。
        
        **不同类型公司的估值参考表**:

        | 公司类型 | 特征 | 推荐关注的指标 (Graham Class) | 为什么？ |
        | :--- | :--- | :--- | :--- |
        | **困境反转/烟蒂股** | 亏损或微利，但资产极其廉价 | **💎 Deep Value (Net-Net)** | 看重清算价值 (NCAV)，不指望它增长，只求它不破产。 |
        | **成熟蓝筹/防御股** | 增长缓慢 (0-5%)，分红稳定，资产扎实 | **🛡️ Defensive Value** | 看重资产安全边际 (Graham Number)，兼顾适度盈利。 |
        | **稳健成长股** | 利润稳定增长 (5-15%)，ROE 较高 | **🚀 Growth Value** | **这才是内在价值公式 ($V = EPS \times (8.5 + 2g)$) 的主场。** 市场愿意为它的增长支付溢价。 |
        | **妖股/概念股** | 极高增长 (>30%) 或 纯题材 | **☁️ Premium / Watch** | 格雷厄姆模型通常会认为它们“高估”，因为模型偏保守，无法捕捉爆发性增长。 |
        | **银行/保险/券商** | 高负债经营，资产负债表特殊 | **PB (市净率) / 股息率** | **不适用 NCAV 或 Intrinsic Value**。银行无流动资产概念，保险负债为准备金。请关注 PB 和分红。 |

    - **NCAV/Share**: 每股净流动资产价值，Calculated as $(Current Assets - Total Liabilities) / Total Shares$。深度价值投资指标，股价低于此值通常被认为是极度低估。
      > **⚠️ 注意**: 银行、保险及部分金融类公司因会计准则差异（不区分流动/非流动资产），无法计算 NCAV，该指标会显示为 N/A。这是正常现象。
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

    ### 智能分类 (Graham Classification)
    系统根据估值指标自动将股票划分为以下 5 类（优先级从高到低）。
    **注意**: 若开启 "Enable Strict Quality Checks" (默认开启)，分类将包含额外的财务健康检查。

    1. **💎 Deep Value (Net-Net)**: 深度价值股。
       - **基础条件**: 股价 < NCAV (净流动资产价值)。
       - **严格模式 (Strict)**: 需同时满足 **资产负债率 < 50%**。
       - **含义**: 极度低估，买入价格低于公司的清算价值（流动资产减去所有负债）。这是格雷厄姆最经典的“捡烟蒂”策略。
    2. **🛡️ Defensive Value**: 防御型价值股。
       - **基础条件**: 股价 < Graham Number (格雷厄姆数值)。
       - **严格模式 (Strict)**: 需同时满足 **流动比率 > 1.2** 且 **资产负债率 < 50%** 且 **ROE > 0**。
       - **含义**: 符合格雷厄姆防御型投资标准，兼顾了盈利能力和资产安全边际。
    3. **🚀 Growth Value**: 成长型价值股。
       - **基础条件**: 股价 < Intrinsic Value (内在价值)。
       - **严格模式 (Strict)**: 需同时满足 **ROE > 8%** 且 **资产负债率 < 60%**。
       - **含义**: 基于成长性模型计算出的低估。这类股票可能市净率较高，但高增长率支撑了其内在价值。*(受左侧 Growth Rate 设置影响)*
    4. **⚠️ Distressed / Loss Making**: 困境/亏损股。
       - **条件**: 亏损 (EPS < 0) 或 资不抵债 (BPS < 0)。
       - **含义**: 基本面恶化，无法使用常规公式估值。
    5. **☁️ Premium / Watch**: 溢价/观察股。
       - **条件**: 股价高于上述所有估值指标。
       - **含义**: 市场给予了溢价，可能处于高估状态，或者拥有极高的护城河/成长性（超出了模型的捕捉范围）。
    6. **📉 Declining / Negative Growth**: 衰退/负增长股。
       - **条件**: 内在价值计算结果为负数 (即 $g < -4.25$)。
       - **含义**: 公司处于业绩下滑通道，且股价尚未低到足以进入 "Defensive Value" 区间。需警惕价值陷阱。

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
        df['total_mv'] = pd.to_numeric(df['total_mv'], errors='coerce')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        
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

# --- Industry Thermometer ---
with st.expander("📊 Industry Thermometer (行业景气度)", expanded=False):
    st.markdown("Based on **Revenue Growth** + **Profit Growth** (Median of industry). Only industries with >= 10 companies are shown.")
    
    # Filter industries with enough data
    ind_counts = df['industry'].value_counts()
    major_inds = ind_counts[ind_counts >= 10].index
    df_major = df[df['industry'].isin(major_inds)].copy()
    
    # Ensure columns are numeric
    for col in ['tr_yoy', 'netprofit_yoy', 'roe', 'pe_ttm', 'total_mv']:
        if col in df_major.columns:
            df_major[col] = pd.to_numeric(df_major[col], errors='coerce')
            
    # Aggregate
    grp = df_major.groupby('industry')[['tr_yoy', 'netprofit_yoy', 'roe', 'pe_ttm', 'total_mv']].median().reset_index()
    grp['Prosperity_Score'] = grp['tr_yoy'] + grp['netprofit_yoy']
    grp['Company_Count'] = grp['industry'].map(ind_counts)
    
    # Plot
    col_chart, col_table = st.columns([3, 2])
    
    with col_chart:
        fig_ind = px.scatter(
            grp, 
            x='pe_ttm', 
            y='Prosperity_Score', 
            size='Company_Count', 
            color='roe',
            hover_name='industry',
            text='industry',
            title="Industry Prosperity vs Valuation",
            labels={'pe_ttm': 'PE (TTM) Median', 'Prosperity_Score': 'Prosperity (Rev+Profit Growth)', 'roe': 'ROE Median'},
            height=450
        )
        fig_ind.update_traces(textposition='top center')
        # Add quadrants
        fig_ind.add_vline(x=30, line_dash="dash", line_color="gray")
        fig_ind.add_hline(y=30, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_ind, use_container_width=True)
    
    with col_table:
        st.subheader("🔥 Top 10 High Prosperity")
        top_10 = grp.sort_values('Prosperity_Score', ascending=False).head(10)
        st.dataframe(
            top_10[['industry', 'Prosperity_Score', 'tr_yoy', 'netprofit_yoy', 'roe', 'pe_ttm']],
            column_config={
                "industry": "Industry",
                "Prosperity_Score": st.column_config.NumberColumn("Score", format="%.1f"),
                "tr_yoy": st.column_config.NumberColumn("Rev Gr %", format="%.1f%%"),
                "netprofit_yoy": st.column_config.NumberColumn("Prof Gr %", format="%.1f%%"),
                "roe": st.column_config.NumberColumn("ROE %", format="%.1f%%"),
                "pe_ttm": st.column_config.NumberColumn("PE", format="%.1f"),
            },
            hide_index=True,
            use_container_width=True
        )

    st.markdown("""
    #### 📖 如何解读该图表 (How to Read)
    - **X轴 (PE TTM)**: 估值水平。越靠左越便宜，越靠右越贵。
    - **Y轴 (Prosperity Score)**: 景气度得分，计算公式为 `营收增速 + 净利润增速`。越靠上景气度越高。
    - **气泡大小**: 该行业包含的公司数量。
    - **颜色深浅 (ROE)**: 盈利质量。颜色越亮/深，代表行业整体 ROE 越高。
    
    **四大象限解读**:
    1. **↖️ 左上角 (高景气 + 低估值)**: **黄金机会区**。行业处于高速增长期，但市场尚未给予高估值（可能是预期差或被错杀）。
    2. **↗️ 右上角 (高景气 + 高估值)**: **热门/泡沫区**。市场已充分定价其高增长，需警惕业绩不及预期带来的杀估值风险。
    3. **↙️ 左下角 (低景气 + 低估值)**: **烟蒂/防御区**。行业成熟或衰退，缺乏增长，但价格便宜。适合寻找高分红或困境反转机会。
    4. **↘️ 右下角 (低景气 + 高估值)**: **风险/高估区**。业绩平平但价格昂贵，需极度谨慎。
    """)

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

# Default manual_g for Company Details view if slider is hidden
manual_g = 5.0

if g_source == "Manual Input":
    manual_g = st.sidebar.slider("Manual Growth Rate (g) %", 0.0, 30.0, 5.0, key="manual_g_slider", help="Used for 'Manual' scenario in Company Details, and as main source if 'Manual Input' is selected above.")
    df['calc_growth_rate'] = manual_g
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

# Store raw growth rate for display purposes
df['raw_growth_rate'] = df['calc_growth_rate']

# Cap Growth Rate Option
enable_cap = st.sidebar.checkbox(
    "Cap Growth Rate at 15%", 
    value=True, 
    help="Limit 'g' to 15% to avoid unrealistic valuations for high-growth companies (Graham formula limitation)."
)

if enable_cap:
    df['calc_growth_rate'] = df['calc_growth_rate'].clip(upper=15)

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
    st.session_state.filter_graham_class = "All"
    st.session_state.filter_enable_mv = False
    st.session_state.filter_enable_pe = False
    st.session_state.filter_enable_pb = False
    st.session_state.filter_enable_graham = False
    st.session_state.filter_enable_iv = False
    st.session_state.filter_enable_ncav = False
    st.session_state.filter_enable_roe = False
    st.session_state.filter_enable_dv = False
    st.rerun()

# Graham Class Filter
graham_classes = ["All", "💎 Deep Value (Net-Net)", "🛡️ Defensive Value", "🚀 Growth Value", "☁️ Premium / Watch", "📉 Declining / Negative Growth", "⚠️ Distressed / Loss Making"]
selected_class = st.sidebar.selectbox("Graham Class", graham_classes, index=2, key="filter_graham_class")

# Strict Mode Option
enable_strict_class = st.sidebar.checkbox(
    "Enable Strict Quality Checks", 
    value=True, 
    help="Add financial health constraints (ROE, Debt, Current Ratio) to Graham Class definitions."
)

# Default Strict Thresholds
strict_roe_min = 8.0
strict_debt_max = 60.0
strict_cr_min = 1.2
strict_debt_max_def = 50.0
strict_debt_max_deep = 50.0

if enable_strict_class:
    with st.sidebar.expander("⚙️ Configure Strict Criteria", expanded=True):
        # Dynamic sliders based on selected class
        if "Growth Value" in selected_class:
            st.caption("Criteria for **Growth Value**:")
            strict_roe_min = st.slider("Min ROE (%)", 0.0, 30.0, 8.0, key="strict_roe_growth")
            strict_debt_max = st.slider("Max Debt Ratio (%)", 0.0, 100.0, 60.0, key="strict_debt_growth")
        
        elif "Defensive Value" in selected_class:
            st.caption("Criteria for **Defensive Value**:")
            strict_cr_min = st.slider("Min Current Ratio", 0.0, 5.0, 1.2, key="strict_cr_def")
            strict_debt_max_def = st.slider("Max Debt Ratio (%)", 0.0, 100.0, 50.0, key="strict_debt_def")
            # ROE > 0 is usually hard constraint for defensive, but we can expose it if needed.
            # Let's keep it simple for now.
            
        elif "Deep Value" in selected_class:
            st.caption("Criteria for **Deep Value**:")
            strict_debt_max_deep = st.slider("Max Debt Ratio (%)", 0.0, 100.0, 50.0, key="strict_debt_deep")
            
        else:
            st.caption("Select a specific class above to configure its strict criteria.")
            st.info("Default settings are applied to all classes.")

# --- Graham Classification (Dynamic) ---
# Default to "Premium" (Price > All Metrics)
df['graham_class'] = "☁️ Premium / Watch"

# Distressed: Negative Earnings or Equity (Graham Number is NaN)
mask_distressed = (df['graham_number'].isna())
df.loc[mask_distressed, 'graham_class'] = "⚠️ Distressed / Loss Making"

# Declining: Intrinsic Value < 0 (Negative Growth)
# Applied after Distressed (so Distressed takes precedence if EPS < 0)
# But before Value categories (so Value categories can override if cheap enough)
mask_declining = (df['intrinsic_value'] < 0) & (~mask_distressed)
df.loc[mask_declining, 'graham_class'] = "📉 Declining / Negative Growth"

# Growth Value: Price < Intrinsic Value
# (Applied first, can be overridden by stricter categories)
# Must ensure Intrinsic Value is positive (Price/Intrinsic > 0) to avoid negative valuations being counted as "undervalued"
mask_growth = (df['price_to_intrinsic'] < 1) & (df['price_to_intrinsic'] > 0)
if enable_strict_class:
    # Strict: ROE > X% AND Debt Ratio < Y%
    mask_growth &= (df['roe'].fillna(0) > strict_roe_min) & (df['debt_to_assets'].fillna(100) < strict_debt_max)

df.loc[mask_growth, 'graham_class'] = "🚀 Growth Value"

# Defensive Value: Price < Graham Number
# (Overrides Growth Value as it's a stricter/safer asset-based standard)
mask_defensive = (df['price_to_graham'] < 1)
if enable_strict_class:
    # Strict: Current Ratio > X AND Debt Ratio < Y% AND Profitable
    mask_defensive &= (df['current_ratio'].fillna(0) > strict_cr_min) & (df['debt_to_assets'].fillna(100) < strict_debt_max_def) & (df['roe'].fillna(0) > 0)

df.loc[mask_defensive, 'graham_class'] = "🛡️ Defensive Value"

# Deep Value (Net-Net): Price < NCAV
# (Highest Priority: The deepest form of value)
mask_deep = (df['price_to_ncav'] < 1) & (df['price_to_ncav'] > 0)
if enable_strict_class:
    # Strict: Debt Ratio < Z%
    mask_deep &= (df['debt_to_assets'].fillna(100) < strict_debt_max_deep)

df.loc[mask_deep, 'graham_class'] = "💎 Deep Value (Net-Net)"

# Search Filter
search_term = st.sidebar.text_input("Search", placeholder="Code or Name (e.g. 000001 or 平安)", key="filter_search")

# Industry Filter
industries = sorted(df['industry'].dropna().unique())
selected_industries = st.sidebar.multiselect("Industry", industries, key="filter_industry")

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
    enable_graham = st.checkbox("Filter by Price/Graham", value=False, key="filter_enable_graham")
    if enable_graham:
        pg_range = st.slider("Price/Graham", 0.0, 5.0, (0.0, 1.0), key="filter_pg_range")

    # Intrinsic Value Filter
    enable_iv = st.checkbox("Filter by Price/Intrinsic Value", value=False, key="filter_enable_iv")
    if enable_iv:
        piv_range = st.slider("Price/Intrinsic Value", 0.0, 5.0, (0.0, 1.0), key="filter_piv_range")
        
    # NCAV
    enable_ncav = st.checkbox("Filter by Price/NCAV", value=False, key="filter_enable_ncav")
    if enable_ncav:
        pncav_range = st.slider("Price/NCAV", 0.0, 5.0, (0.0, 1.0), key="filter_pncav_range")

with st.sidebar.expander("📈 Profitability Filters", expanded=True):
    # ROE
    enable_roe = st.checkbox("Filter by ROE", value=False, key="filter_enable_roe")
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

if selected_class != "All":
    filtered_df = filtered_df[filtered_df['graham_class'] == selected_class]

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
    'ts_code', 'name', 'industry', 'graham_class', 'report_period', 'close', 
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

# Initialize session state for selection
if 'details_code' not in st.session_state:
    st.session_state.details_code = None
if 'last_table_selection' not in st.session_state:
    st.session_state.last_table_selection = []

event = st.dataframe(
    display_df,
    column_config={
        "ts_code": "Code",
        "name": "Name",
        "industry": "Industry",
        "graham_class": "Graham Class",
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
    height=600,
    on_select="rerun",
    selection_mode="single-row"
)

# Handle Table Selection
if event.selection.rows != st.session_state.last_table_selection:
    st.session_state.last_table_selection = event.selection.rows
    if len(event.selection.rows) > 0:
        selected_row_idx = event.selection.rows[0]
        st.session_state.details_code = display_df.iloc[selected_row_idx]['ts_code']

# --- Detailed View ---
st.divider()
st.subheader("🏢 Company Details")

# Determine default index for selectbox
codes = filtered_df['ts_code'].unique()
default_index = 0
if st.session_state.details_code in codes:
    default_index = list(codes).index(st.session_state.details_code)

selected_code = st.selectbox("Select Company for Details", codes, index=default_index)

# Update session state if selectbox changes
if selected_code != st.session_state.details_code:
    st.session_state.details_code = selected_code

if selected_code:
    company = df[df['ts_code'] == selected_code].iloc[0]
    
    # --- Compact View ---
    # Row 1: Basic & Valuation
    r1c1, r1c2, r1c3, r1c4, r1c5, r1c6 = st.columns(6)
    with r1c1: st.metric("Name", company['name'], company['ts_code'])
    with r1c2: st.metric("Price", company['close'])
    with r1c3: st.metric("PE (TTM)", round(company['pe_ttm'], 2) if pd.notnull(company['pe_ttm']) else "N/A")
    with r1c4: st.metric("PB", round(company['pb'], 2) if pd.notnull(company['pb']) else "N/A")
    with r1c5: st.metric("ROE", f"{round(company['roe'], 2)}%" if pd.notnull(company['roe']) else "N/A")
    with r1c6: st.metric("Div Yield", f"{company['dv_ratio']}%" if pd.notnull(company['dv_ratio']) else "N/A")

    # Row 2: Financials & Growth
    r2c1, r2c2, r2c3, r2c4, r2c5, r2c6 = st.columns(6)
    with r2c1: st.metric("Industry", company['industry'])
    with r2c2: st.metric("Revenue", f"{round(company['total_revenue']/1e8, 2)} B" if pd.notnull(company['total_revenue']) else "N/A")
    with r2c3: st.metric("Net Income", f"{round(company['n_income_attr_p']/1e8, 2)} B" if pd.notnull(company['n_income_attr_p']) else "N/A")
    with r2c4: st.metric("Debt/Assets", f"{round(company['debt_to_assets'], 2)}%" if pd.notnull(company['debt_to_assets']) else "N/A")
    with r2c5: 
        val = company.get('eps_ttm_current')
        st.metric("EPS (TTM)", round(val, 3) if pd.notnull(val) else "N/A")
    with r2c6:
        g_ttm = company.get('eps_growth_ttm')
        g_3y = company.get('eps_growth_3y')
        st.metric(
            "EPS Gr (TTM)", 
            f"{round(g_ttm, 1)}%" if pd.notnull(g_ttm) else "N/A",
            delta=f"3Y: {round(g_3y, 1)}%" if pd.notnull(g_3y) else None,
            delta_color="off"
        )

    st.divider()
    st.markdown("#### 🧠 Graham Valuation Metrics")
    
    # Financial Industry Warning
    if company['industry'] in ['银行', '保险', '证券']:
        st.warning(f"⚠️ **Note for {company['industry']} Industry**: Standard valuation metrics like NCAV and Intrinsic Value may not be applicable due to special accounting standards (e.g., no 'Current Assets' distinction). Please focus on PB and Dividend Yield.")

    graham_cols = st.columns(4)
    with graham_cols[0]:
        gn = round(company['graham_number'], 2) if pd.notnull(company['graham_number']) else "N/A"
        st.metric("Graham Number", gn)
    with graham_cols[1]:
        pg = company.get('price_to_graham')
        if pd.notnull(pg):
            st.metric("Price / Graham", round(pg, 2))
            if pg > 1:
                target_price = company.get('graham_number')
                if pd.notnull(target_price):
                    drop_needed = (target_price - company['close']) / company['close'] * 100
                    st.caption(f"Defensive Entry: **{target_price:.2f}** ({drop_needed:.1f}%)")
        else:
            st.metric("Price / Graham", "N/A")
    with graham_cols[2]:
        ncav = round(company['ncav_per_share'], 2) if pd.notnull(company['ncav_per_share']) else "N/A"
        st.metric("NCAV / Share", ncav)
    with graham_cols[3]:
        pncav = round(company['price_to_ncav'], 2) if pd.notnull(company['price_to_ncav']) else "N/A"
        st.metric("Price / NCAV", pncav)

    st.markdown("#### 💎 Intrinsic Value Scenarios (Growth Model)")
    
    # Helper to calculate IV
    def calc_iv(eps, g):
        # V = EPS * (8.5 + 2g)
        # Apply 15% cap if enabled, but only for non-manual/non-zero scenarios usually?
        # User request: "Triggered 15% cap... give hint".
        # Let's apply cap logic consistent with global setting for TTM and 3Y.
        # For Manual, we use as is. For Zero, g=0.
        if enable_cap and g > 15:
            g_eff = 15
        else:
            g_eff = g
        
        # Floor protection for negative growth
        # If g < -4.25, multiplier is negative.
        # Let's floor multiplier at 0 or 0.1 to avoid negative value?
        # Or just let it be negative but display N/A?
        # User previously saw negative value.
        # Let's use the raw formula but handle display.
        val = eps * (8.5 + 2 * g_eff)
        return val, g_eff

    # Prepare data for 4 scenarios
    eps_val = company['eps']
    
    # 1. Zero Growth
    iv_0, _ = calc_iv(eps_val, 0)
    
    # 2. TTM Growth
    g_ttm = company.get('eps_growth_ttm', 0)
    if pd.isna(g_ttm): g_ttm = company.get('netprofit_yoy', 0)
    g_ttm = float(g_ttm) if pd.notnull(g_ttm) else 0
    iv_ttm, g_ttm_eff = calc_iv(eps_val, g_ttm)
    
    # 3. 3-Year Growth
    g_3y = company.get('eps_growth_3y', 0)
    if pd.isna(g_3y): g_3y = company.get('netprofit_yoy', 0)
    g_3y = float(g_3y) if pd.notnull(g_3y) else 0
    iv_3y, g_3y_eff = calc_iv(eps_val, g_3y)
    
    # 4. Manual Growth
    # manual_g is from sidebar
    iv_manual, _ = calc_iv(eps_val, manual_g) # Manual usually not capped? Or should it be? Let's assume manual is manual.
    # Actually calc_iv applies cap if enable_cap is True. 
    # If user manually sets 20%, and cap is on, it will be 15%. 
    # Maybe manual should override cap? 
    # Let's stick to calc_iv logic for consistency, or maybe pass 'apply_cap=False' for manual.
    # Let's assume manual overrides cap.
    iv_manual_raw = eps_val * (8.5 + 2 * manual_g)

    # Display in columns
    scenarios = [
        {"label": "Zero Growth (g=0%)", "val": iv_0, "g_disp": "0%"},
        {"label": "TTM Growth", "val": iv_ttm, "g_disp": f"{round(g_ttm_eff, 2)}%" + (" (Capped)" if enable_cap and g_ttm > 15 else "")},
        {"label": "3-Year Avg Growth", "val": iv_3y, "g_disp": f"{round(g_3y_eff, 2)}%" + (" (Capped)" if enable_cap and g_3y > 15 else "")},
        {"label": f"Manual (g={manual_g}%)", "val": iv_manual_raw, "g_disp": f"{manual_g}%"}
    ]
    
    iv_cols = st.columns(4)

    # Helper to determine class for a specific scenario
    def get_scenario_class(company, iv):
        # Default
        cls = "☁️ Premium"
        reason = "Price > Valuation Metrics"

        # 1. Deep Value (Highest Priority)
        # Priority Check: If the main table already classified it as Deep Value, respect that.
        if "Deep Value" in str(company.get('graham_class', '')):
            cls = "💎 Deep Value (Net-Net)"
            reason = "Price < NCAV"
            return cls, reason

        # Manual Check for Deep Value
        try:
            p_ncav = float(company.get('price_to_ncav', float('inf')))
        except:
            p_ncav = float('inf')
            
        if pd.notnull(p_ncav) and p_ncav < 1 and p_ncav > 0:
            is_deep = True
            fail_reasons = []
            if enable_strict_class:
                try:
                    debt = float(company.get('debt_to_assets', 100))
                except:
                    debt = 100.0
                if debt >= strict_debt_max_deep:
                    is_deep = False
                    fail_reasons.append(f"Debt {debt:.1f}% >= {strict_debt_max_deep}%")
            
            if is_deep:
                return "💎 Deep Value (Net-Net)", "Price < NCAV"
            # If failed strict, we continue to check other classes, but remember failure
            # Actually, if it fails strict Deep Value, it might still be Defensive or Growth?
            # Yes.

        # 2. Defensive Value (Overrides Growth)
        p_graham = company.get('price_to_graham', float('inf'))
        if pd.notnull(p_graham) and p_graham < 1:
            is_defensive = True
            fail_reasons = []
            if enable_strict_class:
                cr = company.get('current_ratio', 0) if pd.notnull(company.get('current_ratio')) else 0
                debt = company.get('debt_to_assets', 100) if pd.notnull(company.get('debt_to_assets')) else 100
                roe = company.get('roe', 0) if pd.notnull(company.get('roe')) else 0
                
                if cr <= strict_cr_min:
                    is_defensive = False
                    fail_reasons.append(f"CR {cr:.2f} <= {strict_cr_min}")
                if debt >= strict_debt_max_def:
                    is_defensive = False
                    fail_reasons.append(f"Debt {debt:.1f}% >= {strict_debt_max_def}%")
                if roe <= 0:
                    is_defensive = False
                    fail_reasons.append(f"ROE {roe:.1f}% <= 0")
            
            if is_defensive:
                return "🛡️ Defensive Value", "Price < Graham Number"
            # If failed strict, continue to Growth

        # 3. Distressed (If not Deep or Defensive)
        if pd.isna(company.get('graham_number')):
            return "⚠️ Distressed", "EPS or BPS is negative"

        # 4. Declining (New)
        if iv < 0:
            cls = "📉 Declining"
            reason = "Intrinsic Value < 0 (g < -4.25%)"
            return cls, reason
        
        # 5. Growth Value (Depends on IV)
        if iv > 0:
            p_iv = company['close'] / iv
            if p_iv < 1:
                is_growth = True
                fail_reasons = []
                if enable_strict_class:
                    roe = company.get('roe', 0) if pd.notnull(company.get('roe')) else 0
                    debt = company.get('debt_to_assets', 100) if pd.notnull(company.get('debt_to_assets')) else 100
                    if roe <= strict_roe_min:
                        is_growth = False
                        fail_reasons.append(f"ROE {roe:.1f}% <= {strict_roe_min}%")
                    if debt >= strict_debt_max:
                        is_growth = False
                        fail_reasons.append(f"Debt {debt:.1f}% >= {strict_debt_max}%")
                
                if is_growth:
                    cls = "🚀 Growth Value"
                    reason = "Price < Intrinsic Value"
                elif fail_reasons:
                    reason = f"Failed Strict: {', '.join(fail_reasons)}"
        
        return cls, reason

    for i, scen in enumerate(scenarios):
        with iv_cols[i]:
            st.caption(scen["label"])
            
            # Always display values, even if negative, to match the main table
            st.metric("Intrinsic Value", round(scen["val"], 2))
            
            if scen["val"] != 0:
                p_iv = company['close'] / scen["val"]
                st.metric("Price / Intrinsic", round(p_iv, 2))
            else:
                st.metric("Price / Intrinsic", "Inf")
            
            # Show Class
            scen_class, scen_reason = get_scenario_class(company, scen["val"])
            st.info(f"{scen_class}")
            if "Failed" in scen_reason or "Premium" in scen_class or "Declining" in scen_class:
                st.caption(f"Reason: {scen_reason}")

            # Always show the g used
            st.caption(f"Using g = {scen['g_disp']}")

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
        
    chart_bytes = None
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
        
        # Generate chart bytes for PDF
        try:
            # Use scale=2 for better resolution
            chart_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        except Exception as e:
            # Just log to console or ignore if image generation fails, don't break the app
            print(f"Error generating chart image: {e}")
            
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
        
        pdf_buffer = create_pdf(company.to_dict(), current_analysis, chart_bytes=chart_bytes)
        st.download_button(
            label="📄 Download Analysis PDF",
            data=pdf_buffer,
            file_name=f"{selected_code}_analysis.pdf",
            mime="application/pdf",
            use_container_width=True
        )

