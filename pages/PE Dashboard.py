import os
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PE Dashboard", layout="wide")

st.title("📊 A股多维度市盈率面板 (PE Dashboard)")
st.caption("展示批量导出的原始数据，并支持分组与聚合分析。来源: scripts/batch_compute_pe.py 输出")

# ---------- Helpers ----------
DATA_DIR = Path("data")
DEFAULT_FILE = None

if DATA_DIR.exists():
    # Prefer most recent pe_ratios_YYYYMMDD.csv, else pe_sample.csv
    candidates: List[Path] = sorted(DATA_DIR.glob("pe_ratios_*.csv"), reverse=True)
    if candidates:
        DEFAULT_FILE = candidates[0]
    else:
        sample = DATA_DIR / "pe_sample.csv"
        DEFAULT_FILE = sample if sample.exists() else None

with st.sidebar:
    st.header("数据源")
    uploaded = st.file_uploader("上传CSV (可选)", type=["csv"])
    st.markdown("或选择本地文件：")

    file_options = []
    if DATA_DIR.exists():
        file_options = sorted([str(p) for p in DATA_DIR.glob("*.csv")])
    chosen_path = st.selectbox("选择CSV", options=["<未选择>"] + file_options, index=(0 if DEFAULT_FILE is None else (file_options.index(str(DEFAULT_FILE)) + 1) if str(DEFAULT_FILE) in file_options else 0))

    st.divider()
    st.markdown("过滤与分组")
    group_keys = st.multiselect(
        "分组字段 (可多选)",
        options=["industry", "area", "market", "is_hs", "scenario"],
        default=["industry"]
    )
    agg_funcs = {
        "static_pe": ["count", "mean", "median"],
        "ttm_pe": ["mean", "median"],
        "linear_pe": ["mean", "median"],
        "forecast_pe_mean": ["mean", "median"],
        "forecast_pe_median": ["mean", "median"],
        "linear_vs_ttm_pct": ["mean", "median"],
        "market_cap": ["mean", "median"],
    }

    st.divider()
    show_charts = st.checkbox("展示基础图表", value=True)

# ---------- Load Data ----------
@st.cache_data(show_spinner=False)
def load_df(uploaded_file, path_str: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    if path_str and path_str != "<未选择>" and os.path.exists(path_str):
        return pd.read_csv(path_str)
    if DEFAULT_FILE and DEFAULT_FILE.exists():
        return pd.read_csv(DEFAULT_FILE)
    return pd.DataFrame()


df = load_df(uploaded, chosen_path)

if df.empty:
    st.warning("未找到CSV数据。请先运行 scripts/batch_compute_pe.py 生成数据，或在侧边栏上传CSV。")
    st.stop()

# Ensure expected columns exist
expected_cols = {
    "ts_code","symbol","name","area","industry","market","is_hs","list_date",
    "trade_date","close","market_cap","static_pe","ttm_pe","linear_pe","latest_quarter",
    "forecast_pe_mean","forecast_pe_median","scenario","linear_vs_ttm_pct"
}
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    st.info(f"提示：以下列在当前文件中不存在，将忽略：{missing}")

# ---------- Raw Table ----------
st.subheader("原始数据")
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Grouping ----------
st.subheader("分组与聚合")
if not group_keys:
    st.info("请选择至少一个分组字段")
else:
    valid_groups = [g for g in group_keys if g in df.columns]
    if not valid_groups:
        st.info("所选分组字段在数据中不存在")
    else:
        agg_candidates = {k: v for k, v in agg_funcs.items() if k in df.columns}
        if not agg_candidates:
            st.info("没有可聚合的数值列")
        else:
            grouped = df.groupby(valid_groups).agg(agg_candidates)
            # flatten MultiIndex columns
            grouped.columns = [f"{c[0]}_{c[1]}" for c in grouped.columns.to_flat_index()]
            grouped = grouped.reset_index().sort_values(by=[valid_groups[0]])
            st.dataframe(grouped, use_container_width=True, hide_index=True)
            csv_bytes = grouped.to_csv(index=False).encode("utf-8")
            st.download_button("下载分组结果CSV", data=csv_bytes, file_name="pe_grouped.csv", mime="text/csv")

# ---------- Charts ----------
if show_charts:
    st.subheader("图表")
    # Scenario distribution
    if "scenario" in df.columns:
        scen = df["scenario"].fillna("(none)").value_counts().reset_index()
        scen.columns = ["scenario", "count"]
        st.bar_chart(scen.set_index("scenario"))
    # Industry median PE bars
    if "industry" in df.columns and "ttm_pe" in df.columns:
        med = df.groupby("industry")["ttm_pe"].median().sort_values(ascending=False).head(30)
        st.bar_chart(med)

st.caption("提示：如需更复杂的切片/筛选，我们可以进一步增加筛选器和交互。")
