"""
DCA (Dollar Cost Averaging) Backtest Page
==========================================
独立页面：定投回测分析工具
可直接在 main.py 所在的多页面应用中访问
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.dca_backtest_engine import DCABacktestEngine
from app.agents.analyst import analyst_agent
from app.state import AgentState
from app.vectorbt_runner import run_vectorbt_dca_backtest


def render_backtest_results(result, context):
    """Render full backtest outputs using cached context and precomputed benchmarks."""
    if result is None or context is None:
        return

    code = context.get("code", "")
    start_date_str = context.get("start_date_str", "")
    end_date_str = context.get("end_date_str", "")
    strategy_type = context.get("strategy_type", "plain")
    rebalance_freq = context.get("rebalance_freq", "M")
    freq_day = context.get("freq_day")
    monthly_investment = context.get("monthly_investment", 0.0)
    commission_rate = context.get("commission_rate", 0.0)
    min_commission = context.get("min_commission", 0.0)
    slippage = context.get("slippage", 0.0)
    initial_capital = context.get("initial_capital", 0.0)
    risk_free_rate = context.get("risk_free_rate", 0.0)
    enable_take_profit = context.get("enable_take_profit", False)
    trailing_params = context.get("trailing_params")
    enable_benchmark = context.get("enable_benchmark", False)
    benchmark_options = context.get("benchmark_options", []) or []
    benchmark_results_ctx = context.get("benchmark_results") or {}
    max_total_investment = context.get("max_total_investment", 0.0)
    elapsed_time = context.get("elapsed_time")

    strategy_display = {
        "plain": "普通定投",
        "smart_pe": "智能PE定投",
        "smart_pb": "智能PB定投",
    }
    freq_display = {"D": "每日", "W": "每周", "M": "每月"}
    freq_detail = ""
    if rebalance_freq == "W" and freq_day:
        freq_detail = f" ({freq_day})"
    elif rebalance_freq == "M" and freq_day:
        freq_detail = f" ({freq_day}号)"

    tp_info = "未启用"
    if enable_take_profit and trailing_params:
        if trailing_params.get("mode") == "target":
            target_return = trailing_params.get("target_return", 0.04)
            tp_info = f"目标 {target_return*100:.1f}%"
        elif trailing_params.get("mode") == "trailing":
            act_return = trailing_params.get("activation_return", 0.3)
            dd_threshold = trailing_params.get("drawdown_threshold", 0.08)
            tp_info = f"激活 {act_return*100:.0f}% 回撤 {dd_threshold*100:.0f}%"

    config_text = f"""
**标的与频率：**
• 标的代码: `{code}` | 复权: `{context.get('price_mode', '后复权')}`
• 策略: `{strategy_display.get(strategy_type, strategy_type)}`
• 频率: `{freq_display.get(rebalance_freq, rebalance_freq)}{freq_detail}` | 金额: `¥{monthly_investment:,.0f}`

**资金与风控：**
• 首期底仓: `¥{initial_capital:,.0f}` | 闲置收益: `{risk_free_rate*100:.1f}%`
• 投资上限: `¥{max_total_investment:,.0f}` | 止盈: `{tp_info}`

**成本参数：**
• 佣金: `{commission_rate*10000:.1f}‱` | 最低: `¥{min_commission:.0f}` | 滑点: `{slippage*100:.2f}%`
"""

    st.success("✅ 回测完成！")
    st.markdown(config_text)

    # Diagnostics and timing
    diag = result.get("diagnostics", {})
    if diag:
        ps = diag.get("price_start")
        pe = diag.get("price_end")
        ps_str = ps.strftime("%Y-%m-%d") if ps is not None else "-"
        pe_str = pe.strftime("%Y-%m-%d") if pe is not None else "-"
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("💾 加载价格行数", f"{diag.get('price_rows', 0):,}")
        with col2:
            st.metric("📅 价格覆盖范围", f"{ps_str} → {pe_str}")
        with col3:
            st.metric("⏱️ 执行耗时", f"{elapsed_time}s" if elapsed_time is not None else "-")

        if diag.get('valuation_rows', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("💹 加载估值行数", f"{diag.get('valuation_rows', 0):,}")
            with col2:
                st.metric("📊 投资执行日数", f"{diag.get('investment_dates', 0):,}")
            with col3:
                st.metric("💹 交易成功率", f"{len(result.get('transactions', [])):,} 笔")

    # Execution summary
    transactions_df = result.get("transactions")
    if transactions_df is not None and not transactions_df.empty:
        total_transactions = len(transactions_df)
        buy_transactions = len(transactions_df[transactions_df["action"] == "BUY"])
        sell_transactions = len(transactions_df[transactions_df["action"].str.contains("SELL", na=False)])

        ps = diag.get("price_start")
        pe = diag.get("price_end")
        actual_start = ps.strftime("%Y%m%d") if ps is not None else start_date_str
        actual_end = pe.strftime("%Y%m%d") if pe is not None else end_date_str

        st.info(f"""
        **交易执行摘要:**
        - 总交易次数: {total_transactions} 次
        - 买入次数: {buy_transactions} 次
        - 卖出次数: {sell_transactions} 次
        - 回测时长: {actual_start} - {actual_end}
        """)
    else:
        st.warning("⚠️ 未产生任何交易记录")

    # Benchmark results (precomputed during run; avoid reruns here)
    benchmark_results = benchmark_results_ctx if isinstance(benchmark_results_ctx, dict) else {}
    if enable_benchmark and benchmark_options and not benchmark_results:
        equity_curve = result["equity_curve"]
        total_invested = result["metrics"].get("total_invested", 0)
        price_series = result.get("price_series")

        if "lump_sum" in benchmark_options and total_invested > 0:
            try:
                if isinstance(price_series, pd.Series) and not price_series.empty:
                    start_price = price_series.iloc[0]
                    shares_lump = total_invested / start_price
                    lump_sum_equity = price_series * shares_lump

                    final_lump = lump_sum_equity.iloc[-1]
                    lump_return = (final_lump - total_invested) / total_invested * 100

                    benchmark_results["lump_sum"] = {
                        "label": "一次性买入",
                        "equity": lump_sum_equity,
                        "metrics": {
                            "total_invested": total_invested,
                            "final_value": final_lump,
                            "total_return_pct": lump_return,
                        },
                    }
                else:
                    st.caption("⚠️ 一次性买入基准缺少价格数据，已跳过")
            except Exception as e:
                st.caption(f"⚠️ 一次性买入基准计算失败: {str(e)}")

    metrics = result["metrics"]
    st.markdown("### 📊 核心指标（含基准对比）")

    def _format_value(value, kind="number"):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        if kind == "currency":
            return f"¥{value:,.0f}"
        if kind == "pct":
            return f"{value:.2f}%"
        if kind == "ratio":
            return f"{value:.2f}"
        if kind == "days":
            return f"{int(value)} 天"
        return f"{value:,.2f}"

    # Prepare benchmark metrics for per-indicator comparison
    comparison_targets = {
        strategy_display.get(strategy_type, strategy_type): metrics
    }

    total_days = metrics.get("total_days", 0) or 0

    for key, br in benchmark_results.items():
        br_metrics = br.get("metrics", {})
        # If benchmark lacks CAGR but has total_return_pct, derive a simple CAGR on total_days.
        if "cagr_pct" not in br_metrics and br_metrics.get("total_return_pct") is not None and total_days > 0:
            try:
                rtn = br_metrics["total_return_pct"] / 100
                br_metrics["cagr_pct"] = ((1 + rtn) ** (365 / total_days) - 1) * 100
            except Exception:
                br_metrics["cagr_pct"] = None
        comparison_targets[br.get("label", key)] = br_metrics

    indicator_plan = [
        ("总投资额", "total_invested", "currency"),
        ("期末资产", "final_value", "currency"),
        ("总收益率", "total_return_pct", "pct"),
        ("年化收益 (CAGR)", "cagr_pct", "pct"),
        ("Sharpe比率", "sharpe_ratio", "ratio"),
        ("Sortino比率", "sortino_ratio", "ratio"),
        ("Calmar比率", "calmar_ratio", "ratio"),
        ("最大回撤", "max_drawdown_pct", "pct"),
        ("年化波动率", "volatility_pct", "pct"),
        ("月度胜率", "win_rate_pct", "pct"),
        ("回测天数", "total_days", "days"),
    ]

    comparison_rows = []
    for label, key, kind in indicator_plan:
        row = {"指标": label}
        for name, values in comparison_targets.items():
            row[name] = _format_value(values.get(key), kind)
        comparison_rows.append(row)

    st.dataframe(pd.DataFrame(comparison_rows), width='stretch', hide_index=True)

    with st.expander("ℹ️ 核心指标含义", expanded=False):
        st.markdown(
            """
            - **总投资额**：回测期内投入的外部资金总和（不含回笼再投资的现金）。
            - **期末资产**：回测结束时的总资产（持仓市值 + 现金）。
            - **总收益率**：期末资产相对总投资额的累计收益百分比。
            - **年化收益 (CAGR)**：将总收益折算成年化的复合增长率。
            - **Sharpe 比率**：每单位总波动获得的超额收益，越高越好。
            - **Sortino 比率**：只考虑下行波动的风险调整收益，越高越好。
            - **Calmar 比率**：年化收益除以最大回撤，衡量收益相对回撤的性价比。
            - **最大回撤**：从最高点到最低点的最大跌幅，越小越稳健。
            - **年化波动率**：收益率的波动幅度年化后结果，越低越稳定。
            - **月度胜率**：月度收益为正的比例，体现收益稳定性。
            - **回测天数**：本次回测覆盖的自然日天数。
            """
        )

    st.markdown("### 📈 净值曲线与投资节点")
    equity_curve = result["equity_curve"]
    transactions = result["transactions"]

    fig_equity = go.Figure()
    fig_equity.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode="lines",
            name="组合净值",
            line=dict(color="royalblue", width=2),
            fill="tozeroy",
            hovertemplate="<b>组合净值</b><br>日期: %{x|%Y-%m-%d}<br>资产: ¥%{y:,.0f}<extra></extra>",
        )
    )

    if not transactions.empty:
        buy_txs = transactions[transactions["action"] == "BUY"]
        if not buy_txs.empty:
            buy_dates = pd.to_datetime(buy_txs["date"])
            buy_values = [
                equity_curve.loc[equity_curve.index >= d].iloc[0]
                if len(equity_curve.loc[equity_curve.index >= d]) > 0
                else equity_curve.iloc[-1]
                for d in buy_dates
            ]

            fig_equity.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_values,
                    mode="markers",
                    name="买入点",
                    marker=dict(color="green", size=8, symbol="triangle-up"),
                    hovertemplate="<b>买入</b><br>日期: %{x|%Y-%m-%d}<br>资产: ¥%{y:,.0f}<extra></extra>",
                )
            )

        sell_txs = transactions[transactions["action"].str.contains("SELL", na=False)]
        if not sell_txs.empty:
            sell_dates = pd.to_datetime(sell_txs["date"])
            sell_values = [
                equity_curve.loc[equity_curve.index >= d].iloc[0]
                if len(equity_curve.loc[equity_curve.index >= d]) > 0
                else equity_curve.iloc[-1]
                for d in sell_dates
            ]

            fig_equity.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_values,
                    mode="markers",
                    name="止盈卖出",
                    marker=dict(color="red", size=10, symbol="triangle-down"),
                    hovertemplate="<b>卖出</b><br>日期: %{x|%Y-%m-%d}<br>资产: ¥%{y:,.0f}<extra></extra>",
                )
            )

    if benchmark_results:
        if "lump_sum" in benchmark_results:
            lump_equity = benchmark_results["lump_sum"]["equity"]
            fig_equity.add_trace(
                go.Scatter(
                    x=lump_equity.index,
                    y=lump_equity.values,
                    mode="lines",
                    name="一次性买入",
                    line=dict(color="orange", width=1.5, dash="dash"),
                )
            )

        if "plain_dca" in benchmark_results:
            plain_equity = benchmark_results["plain_dca"]["equity"]
            fig_equity.add_trace(
                go.Scatter(
                    x=plain_equity.index,
                    y=plain_equity.values,
                    mode="lines",
                    name="普通定投",
                    line=dict(color="green", width=1.5, dash="dot"),
                )
            )

    fig_equity.update_layout(
        title="定投组合净值曲线（含交易标记）",
        xaxis_title="日期",
        yaxis_title="资产价值 (元)",
        hovermode="x unified",
        height=450,
        template="plotly_white",
    )
    fig_equity.update_xaxes(hoverformat="%Y-%m-%d")
    st.plotly_chart(fig_equity, width='stretch')

    st.markdown("### 💼 期末持仓")
    final_position = result["final_position"]
    if final_position:
        position_df = pd.DataFrame([{
            "代码": final_position["code"],
            "持仓数": f"{final_position['shares']:,.2f}",
            "当前价格": f"¥{final_position['price']:.2f}",
            "持仓市值": f"¥{final_position.get('holdings_value', 0):,.2f}",
            "现金余额": f"¥{final_position.get('cash', 0):,.2f}",
            "总资产": f"¥{final_position.get('total_value', 0):,.2f}",
            "总收益": f"¥{final_position['gain']:,.2f}",
            "收益率": f"{final_position['gain_pct']:.2f}%",
        }])
        st.dataframe(position_df, width='stretch')
    else:
        position_df = pd.DataFrame()

    if result.get("strategy_metrics") is not None and not result["strategy_metrics"].empty:
        st.markdown("### 📊 策略指标追踪")
        strategy_df = result["strategy_metrics"]

        metric_col = "pe" if strategy_type == "smart_pe" else "pb"
        if metric_col in strategy_df.columns:
            fig_metric = go.Figure()
            fig_metric.add_trace(
                go.Scatter(
                    x=strategy_df["date"],
                    y=strategy_df[metric_col],
                    mode="lines+markers",
                    name=metric_col.upper(),
                    line=dict(color="orange"),
                )
            )
            fig_metric.update_layout(
                title=f"投资时点的{metric_col.upper()}值变化",
                xaxis_title="日期",
                yaxis_title=metric_col.upper(),
                hovermode="x unified",
                height=350,
                template="plotly_white",
            )
            st.plotly_chart(fig_metric, width='stretch')

    st.markdown("### 📝 交易记录")
    if not transactions.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            total_commission = transactions["commission"].sum()
            st.metric("累计佣金", f"¥{total_commission:,.2f}")
        with col2:
            avg_price = transactions[transactions["action"] == "BUY"]["price"].mean()
            st.metric("平均买入价", f"¥{avg_price:.2f}")
        with col3:
            last_price = transactions.iloc[-1]["price"]
            st.metric("最后交易价", f"¥{last_price:.2f}")

        tx_display = transactions.copy()
        tx_display["date"] = tx_display["date"].dt.strftime("%Y-%m-%d")
        tx_display["cumulative_invested"] = tx_display[tx_display["action"] == "BUY"]["investment"].cumsum()

        display_columns = ["date", "action", "price", "execution_price", "shares", "investment", "commission"]
        if "cumulative_invested" in tx_display.columns:
            display_columns.append("cumulative_invested")

        show_all = st.checkbox("显示全部交易记录", value=False, key=f"show_all_transactions_{context.get('render_id', 'current')}")
        if show_all:
            st.dataframe(tx_display[display_columns], width='stretch')
        else:
            st.dataframe(tx_display[display_columns].tail(30), width='stretch')
            st.caption(f"显示最近30条交易，共{len(transactions)}条")
    else:
        st.warning("暂无交易记录")

    st.markdown("### 📥 导出结果")
    col1, col2, col3 = st.columns(3)

    render_id = context.get("render_id", "current")
    cache_key = f"downloads_{render_id}"
    cache = st.session_state.download_cache.get(cache_key, {})

    if not cache:
        equity_csv = equity_curve.reset_index()
        equity_csv.columns = ["date", "value"]
        cache["equity_csv"] = equity_csv.to_csv(index=False)
        cache["equity_name"] = f"dca_equity_{code}_{start_date_str}_{end_date_str}.csv"

        if not position_df.empty:
            cache["positions_csv"] = position_df.to_csv(index=False)
            cache["positions_name"] = f"dca_positions_{code}_{end_date_str}.csv"

        if not transactions.empty:
            cache["transactions_csv"] = transactions.to_csv(index=False)
            cache["transactions_name"] = f"dca_transactions_{code}_{start_date_str}_{end_date_str}.csv"

        st.session_state.download_cache[cache_key] = cache

    with col1:
        st.download_button(
            label="下载净值曲线",
            data=cache.get("equity_csv", ""),
            file_name=cache.get("equity_name", "dca_equity.csv"),
            mime="text/csv",
            key=f"download_equity_{render_id}",
        )

    with col2:
        if "positions_csv" in cache:
            st.download_button(
                label="下载持仓信息",
                data=cache["positions_csv"],
                file_name=cache.get("positions_name", "dca_positions.csv"),
                mime="text/csv",
                key=f"download_positions_{render_id}",
            )

    with col3:
        if "transactions_csv" in cache:
            st.download_button(
                label="下载交易记录",
                data=cache["transactions_csv"],
                file_name=cache.get("transactions_name", "dca_transactions.csv"),
                mime="text/csv",
                key=f"download_transactions_{render_id}",
            )

# ============================================================================
# Cache & Initialization
# ============================================================================
@st.cache_resource
def get_cached_demo_result():
    """获取中证红利指数的缓存回测结果（演示用）"""
    try:
        engine = DCABacktestEngine()
        result = engine.run_smart_dca_backtest(
            code="000922",
            monthly_investment=10000.0,
            start_date="20150101",
            end_date="20251209",
            strategy_type="plain",
            smart_params=None,
            rebalance_freq="M",
            freq_day=1,
            commission_rate=0.00025,
            min_commission=5.0,
            slippage=0.001,
            initial_capital=0.0,
            risk_free_rate=0.025,
            trailing_params={
                "mode": "target",
                "target_return": 0.04,
                "reentry_mode": "time",
                "reentry_days": 1
            },
            max_total_investment=0.0,  # 不限制总投入
        )
        return result
    except Exception as e:
        st.warning(f"⚠️ 无法加载演示数据: {str(e)}")
        return None

# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="DCA 定投回测 | NL-to-Quant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {font-size: 28px;}
    [data-testid="stMetricLabel"] {font-size: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# Session State Initialization
# ============================================================================
if "backtest_result" not in st.session_state:
    st.session_state.backtest_result = None
if "show_all_transactions" not in st.session_state:
    st.session_state.show_all_transactions = False
if "last_run_context" not in st.session_state:
    st.session_state.last_run_context = None
if "download_cache" not in st.session_state:
    st.session_state.download_cache = {}

# ============================================================================
# Page Title
# ============================================================================
st.title("📊 DCA 定投回测平台")
st.markdown("""
支持中国 A 股 **基金 / 指数 / 股票** 的定投回测，覆盖普通定投与基于估值的智能定投（PE / PB）。
绩效指标包含：总收益、年化收益率、波动率、最大回撤、Sharpe、Sortino 等常用风控与风险调整指标。
""")

# ============================================================================
# Sidebar: Configuration
# ============================================================================
st.sidebar.title("📊 定投回测配置")

# Asset type selection
st.sidebar.markdown("### 标的选择")
asset_type = st.sidebar.selectbox(
    "选择标的类型",
    ["基金", "指数", "股票"],
    index=1,  # Default to "指数"
    help="选择要回测的资产类型"
)

# Pre-defined examples by asset type
asset_examples = {
    "基金": {
        "name": "沪深300ETF",
        "code": "510300",
        "description": "沪深300指数基金",
        "help": "基金代码示例：510300(沪深300)、159915(创业板ETF)、512100(中证1000ETF)"
    },
    "指数": {
        "name": "中证红利指数",
        "code": "000922",
        "description": "中证红利指数",
        "help": "指数代码示例：000922(中证红利)、000300(沪深300指数)、399006(创业板指数)"
    },
    "股票": {
        "name": "长江电力",
        "code": "600900",
        "description": "长江电力股票",
        "help": "股票代码示例：600900(长江电力)、600519(贵州茅台)、000858(五粮液)"
    }
}

example = asset_examples[asset_type]
asset_code = st.sidebar.text_input(
    f"输入{asset_type}代码或名称",
    value=example["code"],
    help=example["help"]
)

price_mode = st.sidebar.selectbox(
    "复权类型",
    ["后复权", "前复权", "不复权"],
    index=0,
    help="建议选择后复权，包含分红再投资的收益，否则 ETF 的长期收益会被严重低估",
)

codes = [asset_code.strip()] if asset_code.strip() else [example["code"]]
weights = {codes[0]: 1.0}

# Strategy Selection
st.sidebar.markdown("### 策略配置")
strategy_type = st.sidebar.selectbox(
    "选择定投策略",
    ["plain", "smart_pe", "smart_pb"],
    format_func=lambda x: {
        "plain": "普通定投 (固定金额)",
        "smart_pe": "智能变额 (PE估值)",
        "smart_pb": "智能变额 (PB估值)",
    }[x],
    help="普通定投每次固定金额；智能变额根据估值动态调整",
)

# Engine selection for cross-validation
engine_choice = st.sidebar.selectbox(
    "回测引擎",
    ["builtin", "vectorbt_plain"],
    format_func=lambda x: {
        "builtin": "内置引擎 (全功能)",
        "vectorbt_plain": "vectorbt (仅plain, 无止盈/智能)",
    }[x],
    help="可选用 vectorbt 进行对照回测，目前仅支持普通定投且不含止盈/智能估值。",
)

# Investment parameters
st.sidebar.markdown("### 投资参数")

# Capital management
with st.sidebar.expander("💰 资金管理", expanded=True):
    initial_capital = st.number_input(
        "首期底仓 (元)",
        min_value=0.0,
        max_value=10000000.0,
        value=0.0,
        step=10000.0,
        help="首期建仓资金（也用于一次性买入基准）；设为 0 则仅做定投",
    )
    
    risk_free_rate = st.slider(
        "闲置资金年化收益率 (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        help="账户中未投资的现金享受的理财收益率（极其重要！）",
    ) / 100
    
    max_total_investment = st.number_input(
        "初始投入资金上限 (元)",
        min_value=0.0,
        max_value=100000000.0,
        value=1000000.0,
        step=100000.0,
        help="外部资金（初始+每期定投）的上限，达到即停止追加；止盈回笼现金不占用此额度（0 表示不限制）",
    )

monthly_investment = st.sidebar.number_input(
    "每次投资金额 (元)",
    min_value=100.0,
    max_value=100000.0,
    value=10000.0,
    step=1000.0,
    help="定期投资金额",
)

rebalance_freq = st.sidebar.selectbox(
    "投资频率",
    ["D", "W", "M"],
    format_func=lambda x: {"D": "每日", "W": "每周", "M": "每月"}[x],
    help="定投的时间间隔",
)

# Frequency details
freq_day = None
if rebalance_freq == "W":
    freq_day = st.sidebar.selectbox(
        "每周哪天投资",
        ["周一", "周二", "周三", "周四", "周五"],
        help="选择每周的哪一天执行定投",
    )
elif rebalance_freq == "M":
    freq_day = st.sidebar.number_input(
        "每月哪天投资",
        min_value=1,
        max_value=31,
        value=1,
        step=1,
        help="每月的第几天执行定投（如遇非交易日顺延）",
    )

# Smart strategy parameters
smart_params = None
if strategy_type in ["smart_pe", "smart_pb"]:
    with st.sidebar.expander("🧠 智能变额参数", expanded=False):
        low_multiplier = st.slider(
            "低估倍数 (便宜时买多少倍)",
            min_value=0.5,
            max_value=3.0,
            value=2.0,
            step=0.25,
            help="当估值极度低估时的投资倍数",
        )
        high_multiplier = st.slider(
            "高估倍数 (贵时买多少倍)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="当估值极度高估时的投资倍数 (0=暂停投资)",
        )
        lookback_days = st.slider(
            "回看周期 (天)",
            min_value=252,
            max_value=252 * 10,
            value=252 * 5,
            step=252,
            help="计算估值分位数的历史天数",
        )
        smart_params = {
            "low_multiple": low_multiplier,
            "high_multiple": high_multiplier,
            "lookback_days": lookback_days,
        }

# Cost and friction parameters
with st.sidebar.expander("💰 成本与摩擦参数", expanded=False):
    commission_rate = st.slider(
        "佣金费率 (万分之几)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
    ) / 10000
    
    min_commission = st.number_input(
        "最低佣金 (元)",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=1.0,
    )
    
    slippage = st.slider(
        "滑点 (%)",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
    ) / 100

# Exit and risk control
with st.sidebar.expander("🛡️ 止盈与风控", expanded=False):
    enable_take_profit = st.checkbox(
        "启用止盈",
        value=True,
        help="是否启用止盈机制",
    )
    
    take_profit_mode = None
    trailing_params = None
    
    if enable_take_profit:
        take_profit_mode = st.selectbox(
            "止盈模式",
            ["target", "trailing"],
            format_func=lambda x: {"target": "目标收益止盈", "trailing": "移动回撤止盈"}[x],
            help="目标收益：达到固定收益率清仓；移动回撤：从高点回撤一定幅度清仓",
        )
        
        if take_profit_mode == "target":
            # Add option for return calculation method
            return_calc_method = st.radio(
                "收益计算方式",
                ["holdings_only", "total_portfolio"],
                format_func=lambda x: {"holdings_only": "持仓收益百分比", "total_portfolio": "总仓位收益百分比"}[x],
                index=0,  # Default to holdings_only
                help="持仓收益：仅计算持仓部分的收益率；总仓位收益：包含现金在内的总资产收益率",
            )
            
            target_return = st.number_input(
                "目标收益率 (%)",
                min_value=0.0,
                max_value=500.0,
                value=4.0,
                step=0.5,
                help="达到此收益率后清仓",
            ) / 100
            trailing_params = {
                "mode": "target", 
                "target_return": target_return,
                "return_calc_method": return_calc_method,
            }
        
        elif take_profit_mode == "trailing":
            activation_return = st.slider(
                "激活线 - 收益率达到多少开始监控 (%)",
                min_value=10.0,
                max_value=100.0,
                value=30.0,
                step=5.0,
            ) / 100
            
            drawdown_threshold = st.slider(
                "回撤线 - 从最高点回吐多少触发清仓 (%)",
                min_value=5.0,
                max_value=30.0,
                value=8.0,
                step=1.0,
            ) / 100
            
            trailing_params = {
                "mode": "trailing",
                "activation_return": activation_return,
                "drawdown_threshold": drawdown_threshold
            }
        
        # Re-entry logic
        st.markdown("**再入场机制**")
        reentry_mode = st.selectbox(
            "清仓后何时重启定投",
            ["time", "price"],
            format_func=lambda x: {"time": "时间触发", "price": "价格触发"}[x],
        )
        
        if reentry_mode == "time":
            reentry_days = st.number_input(
                "空仓等待天数",
                min_value=1,
                max_value=365,
                value=1,
                step=1,
            )
            trailing_params["reentry_mode"] = "time"
            trailing_params["reentry_days"] = reentry_days
        else:
            reentry_drop = st.slider(
                "从卖出价下跌多少后重启 (%)",
                min_value=5.0,
                max_value=50.0,
                value=15.0,
                step=5.0,
            ) / 100
            trailing_params["reentry_mode"] = "price"
            trailing_params["reentry_drop"] = reentry_drop

# Date range selection
st.sidebar.markdown("### 时间范围")
today = datetime.now()

start_date = st.sidebar.date_input(
    "开始日期",
    value=None,
    help="回测的开始日期（留空则从数据最早日期开始）",
)

end_date = st.sidebar.date_input(
    "结束日期",
    value=today,
    help="回测的结束日期",
)

# Benchmark comparison
st.sidebar.markdown("### 📊 基准对比")
enable_benchmark = st.sidebar.checkbox(
    "启用基准对比",
    value=True,
    help="对比一次性买入和普通定投的收益",
)

benchmark_options = []
if enable_benchmark:
    if st.sidebar.checkbox("对比一次性买入", value=True, help="第一天全部买入"):
        benchmark_options.append("lump_sum")
    # 只有当策略不是普通定投时，才显示"对比普通定投"选项
    if strategy_type != "plain":
        if st.sidebar.checkbox("对比普通定投", value=True, help="固定金额定投"):
            benchmark_options.append("plain_dca")
    else:
        st.sidebar.caption("💡 当前已选择普通定投策略")

# ============================================================================
# Main Content Area
# ============================================================================

# Display configuration summary
st.markdown("### 📋 回测配置")

# Build configuration display
strategy_display = {
    "plain": "普通定投",
    "smart_pe": "智能PE定投",
    "smart_pb": "智能PB定投",
}
freq_display = {"D": "每日", "W": "每周", "M": "每月"}
freq_detail = ""
if rebalance_freq == "W" and freq_day:
    freq_detail = f" ({freq_day})"
elif rebalance_freq == "M" and freq_day:
    freq_detail = f" ({freq_day}号)"

# Format take-profit info
tp_info = "未启用"
if enable_take_profit and trailing_params:
    if take_profit_mode == "target":
        target_return = trailing_params.get("target_return", 0.04)
        tp_info = f"目标 {target_return*100:.1f}%"
    elif take_profit_mode == "trailing":
        act_return = trailing_params.get("activation_return", 0.3)
        dd_threshold = trailing_params.get("drawdown_threshold", 0.08)
        tp_info = f"激活 {act_return*100:.0f}% 回撤 {dd_threshold*100:.0f}%"

config_text = f"""
**标的与频率：**
• 标的代码: `{codes[0]}` | 复权: `{price_mode}`
• 策略: `{strategy_display[strategy_type]}`
• 频率: `{freq_display[rebalance_freq]}{freq_detail}` | 金额: `¥{monthly_investment:,.0f}`

**资金与风控：**
• 初始资金: `¥{initial_capital:,.0f}` | 闲置收益: `{risk_free_rate*100:.1f}%`
• 初始投入上限: `¥{max_total_investment:,.0f}` | 止盈: `{tp_info}`

**成本参数：**
• 佣金: `{commission_rate*10000:.1f}‱` | 最低: `¥{min_commission:.0f}` | 滑点: `{slippage*100:.2f}%`
"""

st.markdown(config_text)

# Run backtest
st.markdown("### 🚀 开始回测")
run_backtest_btn = st.button(
    "开始回测",
    type="primary",
    use_container_width=True,
    key="run_backtest",
)

if run_backtest_btn:
    if not codes or not weights:
        st.error("❌ 请先配置组合")
    elif start_date is not None and end_date is not None and start_date >= end_date:
        st.error("❌ 开始日期必须早于结束日期")
    else:
        # Convert dates
        if start_date is None:
            # Use a very early date if start_date is None
            start_date_str = "19900101"
        else:
            start_date_str = start_date.strftime("%Y%m%d")
        
        end_date_str = end_date.strftime("%Y%m%d")

        # Create progress containers
        progress_container = st.empty()
        log_container = st.empty()
        timer_container = st.empty()
        
        import time
        import io
        import sys
        
        start_time = time.time()
        
        # Create a custom log buffer to capture backend logs
        log_buffer = io.StringIO()
        
        try:
            # Initialize engine
            with progress_container:
                st.info("🔧 初始化回测引擎...")
            
            # Display log container header
            with log_container:
                st.markdown("### 📋 执行日志")
                log_display = st.empty()
                log_text = "🟢 开始初始化...\n"
                log_display.code(log_text, language="text")
            
            init_start = time.time()
            engine = DCABacktestEngine()
            init_elapsed = time.time() - init_start
            log_text += f"✓ 回测引擎初始化完成（耗时 {init_elapsed:.3f}s）\n"
            log_text += f"   - 当前价格缓存: {len(engine.PRICE_CACHE)} 条记录\n"
            log_text += f"   - 当前估值缓存: {len(engine.VALUATION_CACHE)} 条记录\n"
            with log_container:
                log_display.code(log_text, language="text")

            # For single asset or smart strategies
            code = codes[0] if len(codes) > 0 else "510300"
            strategy_label = {"plain": "普通定投", "smart_pe": "智能PE", "smart_pb": "智能PB"}.get(strategy_type, strategy_type)
            freq_label = {"D": "每日", "W": "每周", "M": "每月"}.get(rebalance_freq, rebalance_freq)
            
            with progress_container:
                st.info("⚙️ 正在执行回测模拟...")
            
            log_text += f"🟢 步骤1: 正在获取 {code} 的行情数据...\n"
            log_text += f"   时间范围: {start_date_str} - {end_date_str}\n"
            with log_container:
                log_display.code(log_text, language="text")

            # Show initial timer
            interim_elapsed = time.time() - start_time
            with timer_container:
                st.metric("⏱️ 当前耗时", f"{interim_elapsed:.1f}s", delta="执行中...")

            fetch_start = time.time()
            log_text += f"🟢 步骤2: 开始执行回测计算...\n"
            with log_container:
                log_display.code(log_text, language="text")

            # Run backtest in a background thread so we can tick the timer while it runs
            from concurrent.futures import ThreadPoolExecutor

            def _run_backtest():
                # vectorbt cross-validation only supports plain without take-profit/智能
                if engine_choice == "vectorbt_plain":
                    if strategy_type != "plain":
                        raise ValueError("vectorbt 模式仅支持普通定投，请切换回 builtin 或选择 plain")
                    if enable_take_profit:
                        raise ValueError("vectorbt 模式暂不支持止盈/再入场，请关闭止盈或切换 builtin")
                    return run_vectorbt_dca_backtest(
                        code=code,
                        monthly_investment=monthly_investment,
                        start_date=start_date_str,
                        end_date=end_date_str,
                        rebalance_freq=rebalance_freq,
                        freq_day=freq_day,
                        commission_rate=commission_rate,
                        min_commission=min_commission,
                        slippage=slippage,
                        initial_capital=initial_capital,
                        max_total_investment=max_total_investment,
                    )

                return engine.run_smart_dca_backtest(
                    code=code,
                    monthly_investment=monthly_investment,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    strategy_type=strategy_type,
                    smart_params=smart_params,
                    rebalance_freq=rebalance_freq,
                    freq_day=freq_day,
                    commission_rate=commission_rate,
                    min_commission=min_commission,
                    slippage=slippage,
                    initial_capital=initial_capital,
                    risk_free_rate=risk_free_rate,
                    trailing_params=trailing_params if enable_take_profit else None,
                    max_total_investment=max_total_investment,
                )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_run_backtest)
                while not future.done():
                    interim_elapsed = time.time() - start_time
                    with timer_container:
                        st.metric("⏱️ 当前耗时", f"{interim_elapsed:.1f}s", delta="执行中...")
                    time.sleep(0.2)
                result = future.result()

            fetch_elapsed = time.time() - fetch_start

            log_text += f"✓ 数据获取与回测完成（耗时 {fetch_elapsed:.2f}s）\n"
            log_text += (
                f"   - 缓存: 行情{'命中' if engine.last_price_cache_hit else '未命中'}"
                f" / 估值{'命中' if engine.last_valuation_cache_hit else '未命中'}\n"
            )
            # Show cache debug info when cache miss
            if not engine.last_price_cache_hit or not engine.last_valuation_cache_hit:
                ts_codes = engine.candidate_ts_codes(code)
                normalized_code = ts_codes[0] if ts_codes else code
                if not engine.last_price_cache_hit:
                    log_text += f"   - 行情查询键: ({normalized_code}, {start_date_str}, {end_date_str})\n"
                    log_text += f"   - 行情缓存现有: {len(engine.PRICE_CACHE)} 条\n"
                if not engine.last_valuation_cache_hit:
                    log_text += f"   - 估值查询键: ({normalized_code}, {start_date_str}, {end_date_str})\n"
                    log_text += f"   - 估值缓存现有: {len(engine.VALUATION_CACHE)} 条\n"
            log_text += f"🟢 回测参数:\n"
            log_text += f"   - 标的: {code} ({strategy_label})\n"
            log_text += f"   - 频率: {freq_label}\n"
            log_text += f"   - 金额: ¥{monthly_investment:,.0f}\n"
            with log_container:
                log_display.code(log_text, language="text")
            
            # Update timer after backtest completes
            interim_elapsed = time.time() - start_time
            with timer_container:
                st.metric("⏱️ 当前耗时", f"{interim_elapsed:.1f}s", delta="回测完成")

            benchmark_results = {}
            if enable_benchmark and benchmark_options:
                log_text += "🟢 正在计算基准对比...\n"
                with log_container:
                    log_display.code(log_text, language="text")
                
                # Update timer before benchmark
                interim_elapsed = time.time() - start_time
                with timer_container:
                    st.metric("⏱️ 当前耗时", f"{interim_elapsed:.1f}s", delta="计算基准...")
                
                benchmark_start = time.time()
                
                equity_curve = result["equity_curve"]
                total_invested = result["metrics"].get("total_invested", 0)
                price_series = result.get("price_series")

                if "lump_sum" in benchmark_options and total_invested > 0:
                    try:
                        if isinstance(price_series, pd.Series) and not price_series.empty:
                            start_price = price_series.iloc[0]
                            shares_lump = total_invested / start_price
                            lump_sum_equity = price_series * shares_lump

                            benchmark_results["lump_sum"] = {
                                "label": "一次性买入",
                                "equity": lump_sum_equity,
                                "metrics": DCABacktestEngine.compute_metrics_from_equity(
                                    lump_sum_equity,
                                    total_invested,
                                ),
                            }
                    except Exception as e:
                        st.caption(f"⚠️ 一次性买入基准计算失败: {str(e)}")

                if "plain_dca" in benchmark_options and strategy_type != "plain":
                    try:
                        plain_result = engine.run_smart_dca_backtest(
                            code=code,
                            monthly_investment=monthly_investment,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            strategy_type="plain",
                            smart_params=None,
                            rebalance_freq=rebalance_freq,
                            freq_day=freq_day,
                            commission_rate=commission_rate,
                            min_commission=min_commission,
                            slippage=slippage,
                            initial_capital=initial_capital,
                            risk_free_rate=risk_free_rate,
                            trailing_params=trailing_params if enable_take_profit else None,
                            max_total_investment=max_total_investment,
                        )
                        benchmark_results["plain_dca"] = {
                            "label": "普通定投",
                            "equity": plain_result["equity_curve"],
                            "metrics": plain_result.get("metrics", {}),
                        }
                    except Exception as e:
                        st.caption(f"⚠️ 普通定投基准计算失败: {str(e)}")
                
                benchmark_elapsed = time.time() - benchmark_start
                log_text += f"✓ 基准对比计算完成（耗时 {benchmark_elapsed:.2f}s）\n"
                with log_container:
                    log_display.code(log_text, language="text")
                
                # Update timer after benchmark
                interim_elapsed = time.time() - start_time
                with timer_container:
                    st.metric("⏱️ 当前耗时", f"{interim_elapsed:.1f}s")

            # Store result in session state to preserve state when checkbox changes
            st.session_state.backtest_result = result
            
            elapsed_time = time.time() - start_time
            
            log_text += f"\n✅ 全部完成！总耗时 {elapsed_time:.1f}s\n"
            log_text += f"📊 正在生成结果...\n"
            with log_container:
                log_display.code(log_text, language="text")
            
            # Display elapsed time with timer
            with timer_container:
                st.metric("⏱️ 总耗时", f"{elapsed_time:.1f}s")
            
            # Clear progress indicators
            progress_container.empty()

            # Persist context for re-rendering without rerunning backtest
            st.session_state.last_run_context = {
                "code": code,
                "start_date_str": start_date_str,
                "end_date_str": end_date_str,
                "strategy_type": strategy_type,
                "rebalance_freq": rebalance_freq,
                "freq_day": freq_day,
                "monthly_investment": monthly_investment,
                "commission_rate": commission_rate,
                "min_commission": min_commission,
                "slippage": slippage,
                "initial_capital": initial_capital,
                "risk_free_rate": risk_free_rate,
                "enable_take_profit": enable_take_profit,
                "trailing_params": trailing_params if enable_take_profit else None,
                "enable_benchmark": enable_benchmark,
                "benchmark_options": benchmark_options,
                "benchmark_results": benchmark_results,
                "max_total_investment": max_total_investment,
                "elapsed_time": elapsed_time,
                "price_mode": price_mode,
                "render_id": f"run_{int(time.time())}",
            }

        except Exception as e:
            st.error(f"❌ 回测失败：{str(e)}")
            st.info("💡 可能原因：\n- 数据源连接失败\n- ETF代码不存在\n- 时间范围内无数据\n- 估值数据不可用(智能定投需要)")

# ================================================================
# Render results (fresh run or cached) using shared renderer
# ================================================================
if st.session_state.backtest_result is not None and st.session_state.last_run_context is not None:
    render_backtest_results(
        st.session_state.backtest_result,
        st.session_state.last_run_context,
    )

    # Analyst agent powered by LLM
    result_for_insight = st.session_state.backtest_result
    metrics_for_insight = result_for_insight.get("metrics", {}) if isinstance(result_for_insight, dict) else {}
    transactions_df = result_for_insight.get("transactions") if isinstance(result_for_insight, dict) else None
    equity_curve = result_for_insight.get("equity_curve") if isinstance(result_for_insight, dict) else None

    portfolio_data = {}
    trades_json = "[]"
    if equity_curve is not None and len(equity_curve) > 1:
        drawdown_series = equity_curve / equity_curve.cummax() - 1
        portfolio_data = {
            "value": equity_curve.to_json(date_format="iso", orient="split"),
            "drawdown": drawdown_series.to_json(date_format="iso", orient="split"),
        }

    if transactions_df is not None and not transactions_df.empty:
        trades_json = transactions_df.to_json(orient="records", date_format="iso")

    benchmark_metrics = {}
    benchmark_results_ctx = st.session_state.last_run_context.get("benchmark_results") if st.session_state.last_run_context else {}
    if isinstance(benchmark_results_ctx, dict):
        # Use the first available benchmark metrics if present
        for _, br in benchmark_results_ctx.items():
            bm = br.get("metrics") if isinstance(br, dict) else None
            if bm:
                benchmark_metrics = bm
                break

    agent_state: AgentState = {
        "messages": [],
        "tickers": [st.session_state.last_run_context.get("code")],
        "benchmark_ticker": None,
        "start_date": st.session_state.last_run_context.get("start_date_str"),
        "end_date": st.session_state.last_run_context.get("end_date_str"),
        "market_data": {},
        "benchmark_data": {},
        "strategy_code": st.session_state.last_run_context.get("strategy_type"),
        "user_edited_code": None,
        "code_confirmed": True,
        "optimization_mode": False,
        "optimization_params": None,
        "optimization_results": None,
        "execution_output": "",
        "performance_metrics": metrics_for_insight,
        "portfolio_data": portfolio_data,
        "trades_data": trades_json,
        "figure_json": None,
        "benchmark_metrics": benchmark_metrics,
        "analyst_figures": None,
        "analyst_data": None,
        "analysis_completed": None,
        "analysis_runs": 0,
        "valuation": None,
        "data_failed": None,
        "need_full_history": None,
        "needs_benchmark": None,
        "llm_provider": None,
        "llm_model": None,
        "force_agent": None,
        "next_step": None,
        "sender": None,
        "feedback": None,
        "retry_count": None,
        "reasoning": None,
    }

    try:
        analyst_agent(agent_state)
    except Exception as e:
        st.warning(f"⚠️ Analyst Agent 运行失败：{e}")

# ============================================================================
# Footer & Help
# ============================================================================
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    ### 📚 关键指标说明
    - **CAGR**: 年化复合收益率
    - **Sharpe**: 总风险调整后的收益率
    - **Sortino**: 下行风险调整收益率 (仅看负收益)
    - **Calmar**: 年收益 / 最大回撤
    """)

with col2:
    st.markdown("""
    ### 🎯 策略类型
    - **普通定投**: 每期固定金额投资
    - **智能PE**: 根据PE百分位调整金额
    - **智能PB**: 根据PB百分位调整金额
    """)

with col3:
    st.markdown("""
    ### 💡 使用建议
    - 从 3-5 年数据开始回测
    - 对比多个策略找到最适合的方式
    - 关注最大回撤而非单期收益
    - 定期调整参数适应市场变化
    """)
