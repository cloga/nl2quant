"""
Correlation Analysis Page
- Two assets (stock/fund/index) with explicit type dropdowns
- Optional time range (defaults to max overlap)
- Top: basic info; Below: full 6-dimension report per correlation_analyisi.md
- Analyst Agent button to summarize results
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import tushare as ts

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.correlation_analyzer import CorrelationAnalyzer
from app.config import Config
from app.agents.analyst import analyst_agent
from app.state import AgentState

st.set_page_config(page_title="Correlation Analyzer", layout="wide")

ASSET_OPTIONS = {
    "auto": "自动识别",
    "stock": "股票",
    "fund": "基金",
    "index": "指数",
}

PRESET_CHOICES = [
    {"label": "自定义/手动输入", "code": None, "type": "auto"},
    {"label": "黄金ETF 518880", "code": "518880.SH", "type": "fund"},
    {"label": "黄金股ETF 159880", "code": "159880.SZ", "type": "fund"},
    {"label": "消费ETF 159928", "code": "159928.SZ", "type": "fund"},
    {"label": "科创50ETF 588000", "code": "588000.SH", "type": "fund"},
    {"label": "沪深300ETF 510300", "code": "510300.SH", "type": "fund"},
    {"label": "30年国债ETF 511260", "code": "511260.SH", "type": "fund"},
]

# Initialize Tushare pro client
PRO = ts.pro_api(Config.TUSHARE_TOKEN)

def _parse_date(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None

def _enforce_range(date_value: datetime, low: datetime, high: datetime) -> datetime:
    if date_value < low:
        return low
    if date_value > high:
        return high
    return date_value

def _prep_market_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize columns for analyst_agent consumption
    out = df.copy()
    out = out.rename(columns={"trade_date": "Date", "close": "Close"})
    out = out.set_index("Date")
    out.sort_index(inplace=True)
    return out

def _render_basic_info(code: str, asset_type: str, df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("代码", code)
    with col2:
        st.metric("类型", ASSET_OPTIONS.get(asset_type, asset_type))
    with col3:
        st.metric("样本数", f"{len(df):,}")
    with col4:
        st.metric("区间", f"{df['trade_date'].min().date()} → {df['trade_date'].max().date()}")

def _render_fundamentals(fund: dict):
    if not fund:
        st.info("未获取到基本面信息。")
        return

    st.markdown(f"**{fund.get('code', '-')}: {fund.get('name', '-')}**")
    meta = []
    if fund.get("asset_type"):
        meta.append(ASSET_OPTIONS.get(fund.get("asset_type"), fund.get("asset_type")))
    if fund.get("range"):
        meta.append(fund.get("range"))
    if fund.get("samples"):
        meta.append(f"样本 {fund.get('samples'):,}")
    if meta:
        st.caption(" · ".join(meta))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("行业/类别", fund.get("industry") or fund.get("category") or "-")
        st.metric("PE", f"{fund.get('pe'):.2f}" if fund.get("pe") is not None else "-")
    with col2:
        st.metric("PB", f"{fund.get('pb'):.2f}" if fund.get("pb") is not None else "-")
        total_mv = fund.get("total_mv")
        st.metric("总市值(亿)", f"{total_mv/1e8:,.1f}" if total_mv else "-")
    with col3:
        circ_mv = fund.get("circ_mv")
        st.metric("流通市值(亿)", f"{circ_mv/1e8:,.1f}" if circ_mv else "-")
        rev = fund.get("revenue")
        rev_period = fund.get("revenue_period") or "-"
        st.metric("收入(期间)", f"{rev:,.0f}" if rev is not None else "-", help=f"{rev_period}")
    with col4:
        st.metric("ROE", f"{fund.get('roe'):.2f}%" if fund.get("roe") is not None else "-")
        st.metric("ROA", f"{fund.get('roa'):.2f}%" if fund.get("roa") is not None else "-")
    np_yoy = fund.get("netprofit_yoy")
    ind_period = fund.get("indicator_period") or "-"
    if np_yoy is not None:
        st.caption(f"净利润同比: {np_yoy:.2f}% (期间: {ind_period})")
    if fund.get("found_date"):
        st.caption(f"成立/上市日期: {fund.get('found_date')}")

def _build_performance_metrics(results: dict) -> dict:
    return {
        "pearson": results.get("linear", {}).get("pearson", {}).get("corr"),
        "spearman": results.get("linear", {}).get("spearman", {}).get("corr"),
        "beta": results.get("beta", {}).get("beta"),
        "rolling_vol": results.get("rolling", {}).get("volatility"),
        "tail_left": results.get("tail", {}).get("left_tail_dependence", {}).get("probability"),
        "tail_right": results.get("tail", {}).get("right_tail_dependence", {}).get("probability"),
    }

def _plot_normalized_prices(df1: pd.DataFrame, df2: pd.DataFrame, code1: str, code2: str):
    try:
        df = pd.DataFrame({code1: df1.set_index("trade_date")["close"], code2: df2.set_index("trade_date")["close"]}).dropna()
        norm = df / df.iloc[0]
        import plotly.express as px
        fig = px.line(norm, title="归一化价格走势对比", labels={"value": "归一化价格", "index": "日期", "variable": "标的"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"价格图生成失败: {e}")

def _plot_rolling_corr(results: dict):
    roll = results.get("rolling", {})
    series = roll.get("series")
    if series is None or len(series) == 0:
        return
    try:
        import plotly.express as px
        fig = px.line(series, x="date", y="rolling_corr", title="滚动相关系数", labels={"rolling_corr": "相关系数", "date": "日期"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.caption(f"滚动相关图生成失败: {e}")

def _fetch_latest_income(ts_code: str) -> tuple[float | None, str | None]:
    try:
        df = PRO.income(ts_code=ts_code, limit=4, fields="ts_code,end_date,revenue")
        if df is None or df.empty:
            return None, None
        df = df.sort_values("end_date", ascending=False)
        # Prefer annual report (ending with 1231) else latest
        annual = df[df["end_date"].str.endswith("1231")]
        row = annual.iloc[0] if not annual.empty else df.iloc[0]
        rev = float(row["revenue"]) if row.get("revenue") is not None else None
        return rev, row.get("end_date")
    except Exception:
        return None, None

def _fetch_latest_fina_indicator(ts_code: str):
    try:
        df = PRO.fina_indicator(ts_code=ts_code, limit=4, fields="ts_code,end_date,roe,roa,netprofit_yoy")
        if df is None or df.empty:
            return None, None, None, None
        df = df.sort_values("end_date", ascending=False)
        row = df.iloc[0]
        roe = float(row.get("roe")) if row.get("roe") is not None else None
        roa = float(row.get("roa")) if row.get("roa") is not None else None
        netprofit_yoy = float(row.get("netprofit_yoy")) if row.get("netprofit_yoy") is not None else None
        return roe, roa, netprofit_yoy, row.get("end_date")
    except Exception:
        return None, None, None, None

def _fetch_fundamentals(code: str, asset_type: str) -> dict:
    # Best-effort fundamentals fetch per asset type
    info = {"code": code, "asset_type": asset_type}
    if asset_type == "auto":
        prefix = code.split(".")[0]
        if prefix.startswith(("1", "5", "16")):
            asset_type = "fund"
        elif prefix.startswith(("3", "0", "6")):
            asset_type = "stock"
        else:
            asset_type = "index"
    try:
        if asset_type in ("stock", "auto"):
            base = PRO.stock_basic(ts_code=code, fields="ts_code,name,industry,market,list_date")
            if base is not None and not base.empty:
                row = base.iloc[0]
                info["name"] = row.get("name")
                info["industry"] = row.get("industry")
            db = PRO.daily_basic(ts_code=code, limit=1, fields="ts_code,pe_ttm,pb,total_mv,circ_mv")
            if db is not None and not db.empty:
                row = db.iloc[0]
                info["pe"] = float(row.get("pe_ttm")) if row.get("pe_ttm") is not None else None
                info["pb"] = float(row.get("pb")) if row.get("pb") is not None else None
                info["total_mv"] = float(row.get("total_mv")) if row.get("total_mv") is not None else None
                info["circ_mv"] = float(row.get("circ_mv")) if row.get("circ_mv") is not None else None
            rev, period = _fetch_latest_income(code)
            info["revenue"] = rev
            info["revenue_period"] = period
            roe, roa, np_yoy, ind_period = _fetch_latest_fina_indicator(code)
            info["roe"] = roe
            info["roa"] = roa
            info["netprofit_yoy"] = np_yoy
            info["indicator_period"] = ind_period
            return info

        if asset_type == "fund":
            base = PRO.fund_basic(ts_code=code, fields="ts_code,name,management,found_date")
            if base is not None and not base.empty:
                row = base.iloc[0]
                info["name"] = row.get("name")
                info["category"] = row.get("management")
                info["found_date"] = row.get("found_date")
            return info

        if asset_type == "index":
            base = PRO.index_basic(ts_code=code, fields="ts_code,name,market,publisher,category")
            if base is not None and not base.empty:
                row = base.iloc[0]
                info["name"] = row.get("name")
                info["category"] = row.get("category") or row.get("market")
            return info
    except Exception:
        return info
    return info

def _show_report_sections(results: dict):
    st.subheader("线性与方向")
    lin = results.get("linear", {})
    beta = results.get("beta", {})
    c_lin1, c_lin2 = st.columns(2)
    c_lin1.markdown(
        f"**Pearson** {lin.get('pearson', {}).get('corr', float('nan')):.4f}\n\n"
        f"p={lin.get('pearson', {}).get('p_value', float('nan')):.4f}\n\n"
        f"**Spearman** {lin.get('spearman', {}).get('corr', float('nan')):.4f}\n\n"
        f"p={lin.get('spearman', {}).get('p_value', float('nan')):.4f}"
    )
    c_lin2.markdown(
        f"**Kendall** {lin.get('kendall', {}).get('corr', float('nan')):.4f}\n\n"
        f"p={lin.get('kendall', {}).get('p_value', float('nan')):.4f}\n\n"
        f"**Beta** {beta.get('beta', float('nan')):.4f}\n\nR²={beta.get('r_squared', float('nan')):.4f}"
    )

    st.subheader("协整与价差")
    coint = results.get("cointegration", {})
    spread = results.get("spread", {})
    c_coint1, c_coint2 = st.columns(2)
    c_coint1.markdown(
        f"**Engle-Granger p值** {coint.get('engle_granger', {}).get('p_value', float('nan')):.6f}\n\n"
        f"**ADF p值** {coint.get('adf_spread', {}).get('p_value', float('nan')):.6f}"
    )
    c_coint2.markdown(
        f"**当前Z-Score** {spread.get('current_zscore', float('nan')):.4f}\n\n"
        f"**极端事件频率** {spread.get('extreme_events', {}).get('percentage', float('nan')):.2f}%"
    )

    st.subheader("时间领先与因果")
    lag = results.get("granger_lag", {})
    lead = results.get("granger_lead", {})
    cross = results.get("cross_corr", {})
    c_gr1, c_gr2 = st.columns(2)
    c_gr1.markdown(lead.get('interpretation', 'Granger(1→2)'))
    c_gr2.markdown(lag.get('interpretation', 'Granger(2→1)'))
    st.markdown(f"**互相关峰值** {cross.get('max_correlation', float('nan')):.4f} @ lag={cross.get('optimal_lag', 0)}")

    st.subheader("动态时变")
    roll = results.get("rolling", {})
    c_roll1, c_roll2 = st.columns(2)
    c_roll1.markdown(
        f"**滚动均值** {roll.get('mean_correlation', float('nan')):.4f}\n\n"
        f"**波动** {roll.get('volatility', float('nan')):.4f}"
    )
    c_roll2.markdown(
        f"**范围** [{roll.get('min_correlation', float('nan')):.4f}, {roll.get('max_correlation', float('nan')):.4f}]\n\n"
        f"**当前** {roll.get('current_correlation', float('nan')):.4f}"
    )

    st.subheader("尾部依赖")
    tail = results.get("tail", {})
    c_tail1, c_tail2 = st.columns(2)
    c_tail1.markdown(f"**左尾依赖** {tail.get('left_tail_dependence', {}).get('probability', float('nan')):.2%}")
    c_tail2.markdown(f"**右尾依赖** {tail.get('right_tail_dependence', {}).get('probability', float('nan')):.2%}")


def main():
    st.title("📊 相关性分析 (6 维度)")
    st.caption("依据 design/correlation_analyisi.md：线性/协整/因果/滚动/尾部/逻辑 全覆盖。")

    if "corr_state" not in st.session_state:
        st.session_state.corr_state = {}

    with st.form("corr_form"):
        # Row 1: types + preset dropdowns
        c1, c2, c3, c4 = st.columns([1, 2, 1, 2])
        
        preset_labels = [p["label"] for p in PRESET_CHOICES]
        
        type1_default = st.session_state.corr_state.get("type1", "auto")
        type1 = c1.selectbox("标的1类型", options=list(ASSET_OPTIONS.keys()), format_func=lambda x: ASSET_OPTIONS[x], index=list(ASSET_OPTIONS.keys()).index(type1_default), key="type1_select")
        
        code1_preset = c2.selectbox(
            "标的1代码",
            preset_labels,
            index=0,
            help="选择预设",
            key="code1_preset_select"
        )
        preset1 = next(p for p in PRESET_CHOICES if p["label"] == code1_preset)
        code1 = preset1["code"] if preset1["code"] else st.session_state.corr_state.get("code1", "600036.SH")
        type1_default = preset1["type"] if preset1["code"] else st.session_state.corr_state.get("type1", "auto")

        type2_default = st.session_state.corr_state.get("type2", "auto")
        type2 = c3.selectbox("标的2类型", options=list(ASSET_OPTIONS.keys()), format_func=lambda x: ASSET_OPTIONS[x], index=list(ASSET_OPTIONS.keys()).index(type2_default), key="type2_select")
        
        code2_preset = c4.selectbox(
            "标的2代码",
            preset_labels,
            index=0,
            help="选择预设",
            key="code2_preset_select"
        )
        preset2 = next(p for p in PRESET_CHOICES if p["label"] == code2_preset)
        code2 = preset2["code"] if preset2["code"] else st.session_state.corr_state.get("code2", "601166.SH")
        type2_default = preset2["type"] if preset2["code"] else st.session_state.corr_state.get("type2", "auto")

        # Row 2: dates + price/adj options
        c5, c6, c7, c8 = st.columns([1.5, 1.5, 1, 1])
        start_text = c5.text_input("开始日期 (YYYY-MM-DD，可留空自动)")
        end_text = c6.text_input("结束日期 (YYYY-MM-DD，可留空自动)")
        price_mode = c7.selectbox("价格处理方式", options=["log_return", "price"], format_func=lambda x: "对数收益率" if x == "log_return" else "价格水平", index=0)
        adj_type = c8.selectbox("复权方式", options=["hfq", "qfq", None], format_func=lambda x: "后复权" if x == "hfq" else "前复权" if x == "qfq" else "不复权", index=0)

        submitted = st.form_submit_button("运行相关分析", use_container_width=True)

    if submitted:
        analyzer = CorrelationAnalyzer(adj_type=adj_type, price_mode=price_mode)

        try:
            df1 = analyzer.fetch_data(code1, asset_type=type1, adj_type=adj_type)
            df2 = analyzer.fetch_data(code2, asset_type=type2, adj_type=adj_type)
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            return

        overlap_start = max(df1["trade_date"].min(), df2["trade_date"].min())
        overlap_end = min(df1["trade_date"].max(), df2["trade_date"].max())
        if overlap_start >= overlap_end:
            st.error("两个标的无重叠交易区间，无法分析。")
            return

        user_start = _parse_date(start_text)
        user_end = _parse_date(end_text)
        start_dt = _enforce_range(user_start, overlap_start, overlap_end) if user_start else overlap_start
        end_dt = _enforce_range(user_end, overlap_start, overlap_end) if user_end else overlap_end
        if start_dt >= end_dt:
            st.error("时间区间无效，请检查开始/结束日期。")
            return

        # Filter to requested window and push into analyzer cache so downstream methods use它
        df1_sel = df1[(df1["trade_date"] >= start_dt) & (df1["trade_date"] <= end_dt)].reset_index(drop=True)
        df2_sel = df2[(df2["trade_date"] >= start_dt) & (df2["trade_date"] <= end_dt)].reset_index(drop=True)

        if len(df1_sel) < 30 or len(df2_sel) < 30:
            st.warning("区间内数据不足 (<30 行)，结果可能不稳定。")

        analyzer.cache[code1] = df1_sel
        analyzer.cache[code2] = df2_sel

        f1 = _fetch_fundamentals(code1, type1)
        f2 = _fetch_fundamentals(code2, type2)

        with st.spinner("运行 6 维度相关分析..."):
            try:
                results = analyzer.comprehensive_analysis(code1, code2)
                report_text = analyzer._generate_report(code1, code2, results)
            except Exception as e:
                st.error(f"分析失败: {e}")
                return

        st.session_state.corr_state = {
            "code1": code1,
            "code2": code2,
            "type1": type1,
            "type2": type2,
            "start": start_dt,
            "end": end_dt,
            "price_mode": price_mode,
            "adj_type": adj_type,
            "df1": df1_sel,
            "df2": df2_sel,
            "f1": f1,
            "f2": f2,
            "results": results,
            "report": report_text,
            "market_data": {
                code1: _prep_market_df(df1_sel),
                code2: _prep_market_df(df2_sel),
            },
            "perf": _build_performance_metrics(results),
            "fundamentals": {code1: f1, code2: f2},
        }

    state_data = st.session_state.get("corr_state", {})
    if not state_data:
        st.info("填写标的并点击上方按钮运行 6 维度分析。")
        return

    code1 = state_data.get("code1")
    code2 = state_data.get("code2")
    type1 = state_data.get("type1")
    type2 = state_data.get("type2")
    price_mode = state_data.get("price_mode")
    adj_type = state_data.get("adj_type")
    start_dt = state_data.get("start")
    end_dt = state_data.get("end")
    df1_sel = state_data.get("df1")
    df2_sel = state_data.get("df2")
    f1 = state_data.get("f1")
    f2 = state_data.get("f2")
    results = state_data.get("results")
    report_text = state_data.get("report")

    st.markdown("### 🧾 基本面 + 概览")
    col_a, col_b = st.columns(2)
    with col_a:
        _render_fundamentals({**(f1 or {}), "code": code1, "samples": len(df1_sel), "range": f"{df1_sel['trade_date'].min().date()} → {df1_sel['trade_date'].max().date()}", "asset_type": type1})
    with col_b:
        _render_fundamentals({**(f2 or {}), "code": code2, "samples": len(df2_sel), "range": f"{df2_sel['trade_date'].min().date()} → {df2_sel['trade_date'].max().date()}", "asset_type": type2})

    if results:
        st.success("分析完成")
        _show_report_sections(results)

        st.markdown("### 📈 可视化")
        _plot_normalized_prices(df1_sel, df2_sel, code1, code2)
        _plot_rolling_corr(results)

        with st.expander("完整报告 (Markdown)", expanded=False):
            st.code(report_text, language="markdown")
            st.download_button("下载报告", data=report_text, file_name="correlation_report.md", mime="text/markdown")

    st.markdown("---")
    st.subheader("🧐 Analyst Agent 解读")
    st.caption("点击后，Analyst Agent 会基于上方报告给出要点总结与风险提示。")

    if st.button("生成解读", use_container_width=True):
        provider = os.getenv("LLM_PROVIDER", Config.LLM_PROVIDER)
        model_default = Config.PROVIDER_DEFAULT_MODELS.get(provider, None)
        state: AgentState = {
            "messages": [],
            "tickers": [code1, code2],
            "start_date": start_dt.strftime("%Y-%m-%d"),
            "end_date": end_dt.strftime("%Y-%m-%d"),
            "market_data": state_data.get("market_data"),
            "execution_output": report_text,
            "performance_metrics": state_data.get("perf"),
            "fundamentals": state_data.get("fundamentals"),
            "meta": {"price_mode": price_mode, "adj_type": adj_type},
            "llm_provider": provider,
            "llm_model": os.getenv(f"LLM_{provider.upper()}_MODEL_NAME", model_default),
            "analysis_runs": 0,
        }
        analyst_agent(state)


if __name__ == "__main__":
    main()
