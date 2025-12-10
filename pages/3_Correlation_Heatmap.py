import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.dca_backtest_engine import DCABacktestEngine

st.set_page_config(
    page_title="Correlation Heatmap | NL-to-Quant",
    page_icon="📈",
    layout="wide",
)

st.title("📈 股票/指数相关性热图")
st.caption("输入多只股票/ETF/指数代码，选择时间范围，计算区间内的收盘价收益率相关系数。")

# Sidebar inputs
st.sidebar.header("数据配置")
default_codes = "510300\n159915\n000300\n600519"
code_text = st.sidebar.text_area(
    "代码列表（每行一只）",
    value=default_codes,
    height=160,
    help="支持ETF/股票/指数代码，例如 510300, 159915, 000300, 600519",
)

# Date inputs
end_date = st.sidebar.date_input("结束日期", value=datetime.today())
start_date = st.sidebar.date_input(
    "开始日期", value=datetime.today() - timedelta(days=365 * 3)
)

if start_date >= end_date:
    st.sidebar.error("开始日期必须早于结束日期")
    st.stop()

codes = [c.strip() for c in code_text.splitlines() if c.strip()]
if not codes:
    st.warning("请至少输入一只代码")
    st.stop()

st.markdown("### 📅 区间与样本")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("代码数量", len(codes))
with col2:
    st.metric("开始日期", start_date.strftime("%Y-%m-%d"))
with col3:
    st.metric("结束日期", end_date.strftime("%Y-%m-%d"))

run = st.button("🚀 生成相关系数热图", type="primary", use_container_width=True)

if run:
    try:
        engine = DCABacktestEngine()
        price_df = engine.build_price_frame(
            codes=codes,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
        if price_df.empty:
            st.error("未获取到价格数据，请检查代码或时间范围。")
            st.stop()

        # Compute daily returns and correlation
        returns = price_df.pct_change().dropna(how="all")
        # Drop columns with all NaN returns
        returns = returns.dropna(axis=1, how="all")
        if returns.shape[1] < 2:
            st.warning("有效代码少于2个，无法计算相关性。")
            st.stop()

        corr = returns.corr()

        st.markdown("### 🔍 相关系数矩阵")
        st.dataframe(corr.round(3))

        st.markdown("### 🌡️ 热力图")
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(height=650, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

        st.caption("说明：使用对齐后的日度收益率计算皮尔逊相关系数，空值行已剔除；若有代码缺少完整数据，将被自动丢弃。")
    except Exception as e:
        st.error(f"生成热图失败：{e}")
