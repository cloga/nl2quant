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

# Investment parameters
st.sidebar.markdown("### 投资参数")

# Capital management
with st.sidebar.expander("💰 资金管理", expanded=True):
    initial_capital = st.number_input(
        "初始本金 (元)",
        min_value=0.0,
        max_value=10000000.0,
        value=0.0,
        step=10000.0,
        help="账户初始资金，用于一次性买入对比。设为0则只做定投",
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
        "总投入资金量上限 (元)",
        min_value=0.0,
        max_value=100000000.0,
        value=1000000.0,
        step=100000.0,
        help="累计投入达到此金额后停止定投（0表示不限制）",
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
            target_return = st.number_input(
                "目标收益率 (%)",
                min_value=0.0,
                max_value=500.0,
                value=4.0,
                step=0.5,
                help="达到此收益率后清仓",
            ) / 100
            trailing_params = {"mode": "target", "target_return": target_return}
        
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
    if st.sidebar.checkbox("对比普通定投", value=True, help="固定金额定投"):
        benchmark_options.append("plain_dca")

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
• 投资上限: `¥{max_total_investment:,.0f}` | 止盈: `{tp_info}`

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
        status_container = st.empty()
        timer_container = st.empty()
        
        import time
        
        start_time = time.time()
        
        try:
            # Initialize engine
            with progress_container:
                st.info("🔧 初始化回测引擎...")
            st.write("")  # 添加间距
            engine = DCABacktestEngine()

            # Fetch data
            with progress_container:
                st.info(f"📡 正在获取 {codes[0]} 的价格数据...")
                st.caption(f"   时间范围: {start_date_str} - {end_date_str}")
            st.write("")  # 添加间距

            # For single asset or smart strategies
            code = codes[0] if len(codes) > 0 else "510300"
            strategy_label = {"plain": "普通定投", "smart_pe": "智能PE", "smart_pb": "智能PB"}.get(strategy_type, strategy_type)
            freq_label = {"D": "每日", "W": "每周", "M": "每月"}.get(rebalance_freq, rebalance_freq)
            
            # Show configuration summary
            with status_container:
                st.markdown(f"""
                **回测配置:**
                - 标的代码: `{code}`
                - 策略类型: `{strategy_label}`
                - 投资频率: `{freq_label}`
                - 每次金额: `¥{monthly_investment:,.0f}`
                - 初始资金: `¥{initial_capital:,.0f}`
                - 佣金费率: `{commission_rate*10000:.1f}‱`
                """)
            
            with progress_container:
                st.info("⚙️ 正在执行回测模拟...")
            
            result = engine.run_smart_dca_backtest(
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
            
            elapsed_time = int(time.time() - start_time)
            
            # Display elapsed time with timer
            with timer_container:
                st.caption(f"⏱️ 已耗时: {elapsed_time}s")
            
            # Clear progress indicators
            progress_container.empty()
            status_container.empty()

            # ================================================================
            # Display Results
            # ================================================================
            st.success(f"✅ 回测完成！(耗时 {elapsed_time}s)")
            
            # Data diagnostics - show FIRST with detailed timing
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
                    st.metric("⏱️ 执行耗时", f"{elapsed_time}s")
                
                if diag.get('valuation_rows', 0) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("💹 加载估值行数", f"{diag.get('valuation_rows', 0):,}")
                    with col2:
                        st.metric("📊 投资执行日数", f"{diag.get('investment_dates', 0):,}")
                    with col3:
                        st.metric("💹 交易成功率", f"{len(result.get('transactions', [])):,} 笔")
            
            # Show execution summary
            transactions_df = result.get("transactions")
            if transactions_df is not None and not transactions_df.empty:
                total_transactions = len(transactions_df)
                buy_transactions = len(transactions_df[transactions_df["action"] == "BUY"])
                sell_transactions = len(transactions_df[transactions_df["action"].str.contains("SELL", na=False)])
                
                st.info(f"""
                **交易执行摘要:**
                - 总交易次数: {total_transactions} 次
                - 买入次数: {buy_transactions} 次
                - 卖出次数: {sell_transactions} 次
                - 回测时长: {start_date_str} - {end_date_str}
                """)
            else:
                st.warning("⚠️ 未产生任何交易记录")

            # Key metrics
            metrics = result["metrics"]
            st.markdown("### 📊 核心指标")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric(
                    "总投资额",
                    f"¥{metrics.get('total_invested', 0):,.0f}",
                )
            with col2:
                st.metric(
                    "期末资产",
                    f"¥{metrics.get('final_value', 0):,.0f}",
                )
            with col3:
                st.metric(
                    "总收益率",
                    f"{metrics.get('total_return_pct', 0):.2f}%",
                )
            with col4:
                st.metric(
                    "年化收益 (CAGR)",
                    f"{metrics.get('cagr_pct', 0):.2f}%",
                )
            with col5:
                st.metric(
                    "Sharpe比率",
                    f"{metrics.get('sharpe_ratio', 0):.2f}",
                )

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Sortino比率",
                    f"{metrics.get('sortino_ratio', 0):.2f}",
                )
            with col2:
                st.metric(
                    "Calmar比率",
                    f"{metrics.get('calmar_ratio', 0):.2f}",
                )
            with col3:
                st.metric(
                    "最大回撤",
                    f"{metrics.get('max_drawdown_pct', 0):.2f}%",
                )
            with col4:
                st.metric(
                    "年化波动率",
                    f"{metrics.get('volatility_pct', 0):.2f}%",
                )

            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "月度胜率",
                    f"{metrics.get('win_rate_pct', 0):.2f}%",
                )
            with col2:
                st.metric(
                    "回测天数",
                    f"{metrics.get('total_days', 0)} 天",
                )

            # Equity curve chart with investment markers
            st.markdown("### 📈 净值曲线与投资节点")
            equity_curve = result["equity_curve"]
            transactions = result["transactions"]

            fig_equity = go.Figure()
            
            # Add equity curve
            fig_equity.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode="lines",
                    name="组合净值",
                    line=dict(color="royalblue", width=2),
                    fill="tozeroy",
                )
            )
            
            # Add buy markers
            if not transactions.empty:
                buy_txs = transactions[transactions["action"] == "BUY"]
                if not buy_txs.empty:
                    # Get equity values at buy dates
                    buy_dates = pd.to_datetime(buy_txs["date"])
                    buy_values = [equity_curve.loc[equity_curve.index >= d].iloc[0] if len(equity_curve.loc[equity_curve.index >= d]) > 0 else equity_curve.iloc[-1] for d in buy_dates]
                    
                    fig_equity.add_trace(
                        go.Scatter(
                            x=buy_dates,
                            y=buy_values,
                            mode="markers",
                            name="买入点",
                            marker=dict(color="green", size=8, symbol="triangle-up"),
                            hovertemplate="<b>买入</b><br>日期: %{x}<br>资产: ¥%{y:,.0f}<extra></extra>",
                        )
                    )
                
                # Add sell markers if any
                sell_txs = transactions[transactions["action"].str.contains("SELL", na=False)]
                if not sell_txs.empty:
                    sell_dates = pd.to_datetime(sell_txs["date"])
                    sell_values = [equity_curve.loc[equity_curve.index >= d].iloc[0] if len(equity_curve.loc[equity_curve.index >= d]) > 0 else equity_curve.iloc[-1] for d in sell_dates]
                    
                    fig_equity.add_trace(
                        go.Scatter(
                            x=sell_dates,
                            y=sell_values,
                            mode="markers",
                            name="止盈卖出",
                            marker=dict(color="red", size=10, symbol="triangle-down"),
                            hovertemplate="<b>卖出</b><br>日期: %{x}<br>资产: ¥%{y:,.0f}<extra></extra>",
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
            st.plotly_chart(fig_equity, use_container_width=True)

            # Positions breakdown
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
                st.dataframe(position_df, use_container_width=True)

            # Strategy metrics for smart strategies
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
                    st.plotly_chart(fig_metric, use_container_width=True)

            # Transaction history
            st.markdown("### 📝 交易记录")
            transactions = result["transactions"]
            if not transactions.empty:
                # Show summary stats
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
                
                # Show transaction table with more details
                tx_display = transactions.copy()
                tx_display["date"] = tx_display["date"].dt.strftime("%Y-%m-%d")
                
                # Add cumulative investment column
                tx_display["cumulative_invested"] = tx_display[tx_display["action"] == "BUY"]["investment"].cumsum()
                
                display_columns = ["date", "action", "price", "execution_price", "shares", "investment", "commission"]
                if "cumulative_invested" in tx_display.columns:
                    display_columns.append("cumulative_invested")
                
                # Show last 30 transactions by default
                show_all = st.checkbox("显示全部交易记录", value=False)
                if show_all:
                    st.dataframe(tx_display[display_columns], use_container_width=True)
                else:
                    st.dataframe(tx_display[display_columns].tail(30), use_container_width=True)
                    st.caption(f"显示最近30条交易，共{len(transactions)}条")

            # Download results
            st.markdown("### 📥 导出结果")
            col1, col2, col3 = st.columns(3)

            with col1:
                equity_csv = equity_curve.reset_index()
                equity_csv.columns = ["date", "value"]
                st.download_button(
                    label="下载净值曲线",
                    data=equity_csv.to_csv(index=False),
                    file_name=f"dca_equity_{code}_{start_date_str}_{end_date_str}.csv",
                    mime="text/csv",
                )

            with col2:
                st.download_button(
                    label="下载持仓信息",
                    data=position_df.to_csv(index=False),
                    file_name=f"dca_positions_{code}_{end_date_str}.csv",
                    mime="text/csv",
                )

            with col3:
                if not transactions.empty:
                    st.download_button(
                        label="下载交易记录",
                        data=transactions.to_csv(index=False),
                        file_name=f"dca_transactions_{code}_{start_date_str}_{end_date_str}.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"❌ 回测失败：{str(e)}")
            st.info("💡 可能原因：\n- 数据源连接失败\n- ETF代码不存在\n- 时间范围内无数据\n- 估值数据不可用(智能定投需要)")

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
