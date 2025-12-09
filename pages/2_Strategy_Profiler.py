"""
Strategy Profiler Page
======================
独立页面：跑多种参数组合，挑选潜在的最佳定投策略
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
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Strategy Profiler | NL-to-Quant",
    page_icon="🧭",
    layout="wide",
)

st.title("🧭 Strategy Profiler")
st.markdown("""
批量跑不同参数组合，对比核心指标，快速筛选潜在的最佳定投策略。
""")

# ============================================================================
# Configuration
# ============================================================================
st.sidebar.title("⚙️ Profiler 配置")

# Asset Selection
asset_code = st.sidebar.text_input(
    "标的代码",
    value="000922",
    help="支持股票(如 600519.SH)、ETF(如 510300.SH)、指数(如 000922.CSI)"
)

# Date range
st.sidebar.markdown("### 时间范围")
today = datetime.now()

start_date = st.sidebar.date_input(
    "开始日期",
    value=None,
    help="留空则从数据最早日期开始",
)

end_date = st.sidebar.date_input(
    "结束日期",
    value=today,
)

# Investment amount
st.sidebar.markdown("### 投资参数")

# Capital management
with st.sidebar.expander("💰 资金管理", expanded=False):
    initial_capital = st.number_input(
        "初始本金 (元)",
        min_value=0.0,
        max_value=10000000.0,
        value=0.0,
        step=10000.0,
        help="账户初始资金",
    )
    
    risk_free_rate = st.slider(
        "闲置资金年化收益率 (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
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
)

rebalance_freq = st.sidebar.selectbox(
    "投资频率",
    ["D", "W", "M"],
    format_func=lambda x: {"D": "每日", "W": "每周", "M": "每月"}[x],
)

# Frequency details
freq_day = None
if rebalance_freq == "W":
    freq_day = st.sidebar.selectbox(
        "每周哪天投资",
        ["周一", "周二", "周三", "周四", "周五"],
    )
elif rebalance_freq == "M":
    freq_day = st.sidebar.number_input(
        "每月哪天投资",
        min_value=1,
        max_value=31,
        value=1,
        step=1,
    )

# Smart parameters for comparison
st.sidebar.markdown("### 智能定投参数")
low_multiplier = st.sidebar.slider(
    "低估倍数",
    min_value=0.5,
    max_value=3.0,
    value=2.0,
    step=0.25,
)

high_multiplier = st.sidebar.slider(
    "高估倍数",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1,
)

lookback_days = st.sidebar.slider(
    "回看周期 (天)",
    min_value=252,
    max_value=252 * 10,
    value=252 * 5,
    step=252,
)

# Cost parameters
with st.sidebar.expander("💰 成本参数", expanded=False):
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

# ============================================================================
# Main Content
# ============================================================================

st.markdown(f"### 📋 分析范围")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("标的代码", asset_code)
with col2:
    if start_date is not None and end_date is not None:
        days_diff = (end_date - start_date).days
        st.metric("回测天数", f"{days_diff} 天")
    else:
        st.metric("回测天数", "-")
with col3:
    freq_text = {"D": "每日", "W": "每周", "M": "每月"}[rebalance_freq]
    if rebalance_freq == "W" and freq_day:
        freq_text += f" ({freq_day})"
    elif rebalance_freq == "M" and freq_day:
        freq_text += f" ({freq_day}号)"
    st.metric("投资频率", freq_text)

# Parameter grid for profiling
st.markdown("### 🧪 参数组合 (逐行一组)")
st.caption("格式示例：\n- plain\n- smart_pe: low=2.0, high=0.5, lookback=1260\n- smart_pb: low=1.5, high=0.7, lookback=756")
combination_text = st.text_area(
    "策略与参数组合",
    value="plain\nsmart_pe: low=2.0, high=0.5, lookback=1260\nsmart_pb: low=1.5, high=0.7, lookback=756",
    height=140,
)

rank_metric = st.selectbox(
    "排序指标",
    ["cagr_pct", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"],
    format_func=lambda x: {
        "cagr_pct": "年化收益 (CAGR)",
        "total_return_pct": "总收益率",
        "sharpe_ratio": "Sharpe比率",
        "max_drawdown_pct": "最大回撤(越低越好)",
    }[x],
)

# Run profiling
run_btn = st.button("🚀 运行 Strategy Profiler", type="primary", use_container_width=True)

if run_btn:
    if start_date is not None and end_date is not None and start_date >= end_date:
        st.error("❌ 开始日期必须早于结束日期")
    else:
        if start_date is None:
            start_date_str = "19900101"
        else:
            start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        # Parse parameter combinations
        combinations = []
        parse_errors = []
        for line in combination_text.splitlines():
            raw = line.strip()
            if not raw:
                continue
            if ":" not in raw:
                # plain line like "plain"
                combinations.append({"label": raw, "strategy": raw, "params": None})
                continue
            try:
                strategy_part, params_part = raw.split(":", 1)
                strategy = strategy_part.strip()
                param_tokens = [p.strip() for p in params_part.split(",") if p.strip()]
                params = {}
                for tok in param_tokens:
                    if "=" not in tok:
                        continue
                    k, v = tok.split("=", 1)
                    k = k.strip()
                    v = v.strip()
                    try:
                        if "." in v:
                            params[k] = float(v)
                        else:
                            params[k] = int(v)
                    except ValueError:
                        params[k] = float(v) if v.replace(".", "", 1).isdigit() else v
                combinations.append({"label": raw, "strategy": strategy, "params": params})
            except Exception as e:
                parse_errors.append(f"无法解析行: {raw} ({e})")

        if parse_errors:
            st.error("参数解析错误:\n" + "\n".join(parse_errors))
        elif not combinations:
            st.warning("未提供任何参数组合")
        else:
            results_dict = {}
            with st.spinner("⏳ 正在执行策略组合回测..."):
                try:
                    engine = DCABacktestEngine()
                    for combo in combinations:
                        strategy_code = combo["strategy"]
                        label = combo["label"]
                        smart_params = None

                        if strategy_code in ["smart_pe", "smart_pb"]:
                            params = combo.get("params", {}) or {}
                            smart_params = {
                                "low_multiple": params.get("low", low_multiplier),
                                "high_multiple": params.get("high", high_multiplier),
                                "lookback_days": params.get("lookback", lookback_days),
                            }

                        result = engine.run_smart_dca_backtest(
                            code=asset_code,
                            monthly_investment=monthly_investment,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            strategy_type=strategy_code,
                            smart_params=smart_params,
                            rebalance_freq=rebalance_freq,
                            freq_day=freq_day,
                            commission_rate=commission_rate,
                            min_commission=min_commission,
                            slippage=slippage,
                            initial_capital=initial_capital,
                            risk_free_rate=risk_free_rate,
                            max_total_investment=max_total_investment,
                        )

                        results_dict[label] = {
                            "code": strategy_code,
                            "result": result,
                            "metrics": result["metrics"],
                            "diagnostics": result.get("diagnostics", {}),
                        }

                    st.success("✅ Profiler 完成！")

                except Exception as e:
                    st.error(f"❌ 分析失败：{str(e)}")
                    st.info("💡 可能原因：\n- ETF代码不存在\n- 数据源连接失败\n- 时间范围内无数据")
                    results_dict = {}
        
        if not results_dict:
            st.stop()

        # Show diagnostics for the first strategy as data reference
        first_diag = next((v.get("diagnostics", {}) for v in results_dict.values() if v.get("diagnostics")), None)
        if first_diag:
            ps = first_diag.get("price_start")
            pe = first_diag.get("price_end")
            ps_str = ps.strftime("%Y-%m-%d") if ps is not None else "-"
            pe_str = pe.strftime("%Y-%m-%d") if pe is not None else "-"
            st.info(
                f"数据加载: 价格 {first_diag.get('price_rows', 0)} 条 ({ps_str} 至 {pe_str}), "
                f"估值 {first_diag.get('valuation_rows', 0)} 条, 投资执行日 {first_diag.get('investment_dates', 0)} 个"
            )

        # ================================================================
        # 对比表格
        # ================================================================
        st.markdown("### 📊 性能对比表 (按排序指标排列)")

        comparison_data = []
        for strategy_name, data in results_dict.items():
            metrics = data["metrics"]
            comparison_data.append({
                "策略": strategy_name,
                "策略类型": data["code"],
                "总投资额 (¥)": metrics.get('total_invested', 0),
                "期末资产 (¥)": metrics.get('final_value', 0),
                "总收益率": metrics.get('total_return_pct', 0),
                "CAGR": metrics.get('cagr_pct', 0),
                "年化波动": metrics.get('volatility_pct', 0),
                "Sharpe": metrics.get('sharpe_ratio', 0),
                "Sortino": metrics.get('sortino_ratio', 0),
                "最大回撤": metrics.get('max_drawdown_pct', 0),
                "Calmar": metrics.get('calmar_ratio', 0),
            })

        comparison_df = pd.DataFrame(comparison_data)

        # Sort by chosen metric (max_drawdown_pct ascending, others descending)
        ascending = rank_metric == "max_drawdown_pct"
        comparison_df = comparison_df.sort_values(by={
            "cagr_pct": "CAGR",
            "total_return_pct": "总收益率",
            "sharpe_ratio": "Sharpe",
            "max_drawdown_pct": "最大回撤",
        }[rank_metric], ascending=ascending)

        # Display formatted
        display_df = comparison_df.copy()
        display_df["总投资额 (¥)"] = display_df["总投资额 (¥)"].map(lambda x: f"{x:,.0f}")
        display_df["期末资产 (¥)"] = display_df["期末资产 (¥)"].map(lambda x: f"{x:,.0f}")
        for col in ["总收益率", "CAGR", "年化波动", "最大回撤"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}%")
        for col in ["Sharpe", "Sortino", "Calmar"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.2f}")

        st.dataframe(display_df, width='stretch')

        # ================================================================
        # 净值曲线对比
        # ================================================================
        st.markdown("### 📈 净值曲线对比")
        
        fig_equity = go.Figure()
        colors = ["royalblue", "darkorange", "darkgreen"]
        
        for idx, (strategy_name, data) in enumerate(results_dict.items()):
            equity_curve = data["result"]["equity_curve"]
            fig_equity.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode="lines",
                    name=strategy_name,
                    line=dict(color=colors[idx], width=2),
                )
            )
        
        fig_equity.update_layout(
            title="策略净值曲线对比",
            xaxis_title="日期",
            yaxis_title="资产价值 (元)",
            hovermode="x unified",
            height=450,
            template="plotly_white",
        )
        st.plotly_chart(fig_equity, width='stretch')

        # ================================================================
        # 关键指标对比柱状图
        # ================================================================
        st.markdown("### 📊 核心指标对比")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CAGR对比
            cagr_data = {
                strategy_name: data["metrics"].get("cagr_pct", 0)
                for strategy_name, data in results_dict.items()
            }
            fig_cagr = go.Figure(
                data=[go.Bar(x=list(cagr_data.keys()), y=list(cagr_data.values()))]
            )
            fig_cagr.update_layout(
                title="年化收益率 (CAGR) 对比",
                yaxis_title="CAGR (%)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_cagr, width='stretch')
        
        with col2:
            # Sharpe对比
            sharpe_data = {
                strategy_name: data["metrics"].get("sharpe_ratio", 0)
                for strategy_name, data in results_dict.items()
            }
            fig_sharpe = go.Figure(
                data=[go.Bar(x=list(sharpe_data.keys()), y=list(sharpe_data.values()))]
            )
            fig_sharpe.update_layout(
                title="Sharpe 比率对比",
                yaxis_title="Sharpe Ratio",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_sharpe, width='stretch')

        col1, col2 = st.columns(2)
        
        with col1:
            # 最大回撤对比
            drawdown_data = {
                strategy_name: data["metrics"].get("max_drawdown_pct", 0)
                for strategy_name, data in results_dict.items()
            }
            fig_dd = go.Figure(
                data=[go.Bar(x=list(drawdown_data.keys()), y=list(drawdown_data.values()))]
            )
            fig_dd.update_layout(
                title="最大回撤对比",
                yaxis_title="最大回撤 (%)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_dd, width='stretch')
        
        with col2:
            # 波动率对比
            vol_data = {
                strategy_name: data["metrics"].get("volatility_pct", 0)
                for strategy_name, data in results_dict.items()
            }
            fig_vol = go.Figure(
                data=[go.Bar(x=list(vol_data.keys()), y=list(vol_data.values()))]
            )
            fig_vol.update_layout(
                title="年化波动率对比",
                yaxis_title="波动率 (%)",
                showlegend=False,
                height=350,
            )
            st.plotly_chart(fig_vol, width='stretch')

        # ================================================================
        # 详细统计表
        # ================================================================
        st.markdown("### 📋 详细统计")
        
        for strategy_name, data in results_dict.items():
            with st.expander(f"📌 {strategy_name} 详细信息", expanded=False):
                metrics = data["metrics"]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总投资额", f"¥{metrics.get('total_invested', 0):,.0f}")
                    st.metric("期末资产", f"¥{metrics.get('final_value', 0):,.0f}")
                    st.metric("总收益额", f"¥{metrics.get('final_value', 0) - metrics.get('total_invested', 0):,.0f}")
                
                with col2:
                    st.metric("总收益率", f"{metrics.get('total_return_pct', 0):.2f}%")
                    st.metric("CAGR", f"{metrics.get('cagr_pct', 0):.2f}%")
                    st.metric("年化波动", f"{metrics.get('volatility_pct', 0):.2f}%")
                
                with col3:
                    st.metric("最大回撤", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
                    st.metric("Sharpe比率", f"{metrics.get('sharpe_ratio', 0):.2f}")
                    st.metric("Sortino比率", f"{metrics.get('sortino_ratio', 0):.2f}")

# ============================================================================
# Footer & Help
# ============================================================================
st.markdown("---")
st.markdown("""
### 💡 Profiler 使用说明

**支持的策略：**
- **plain**: 普通定投，每期固定金额
- **smart_pe**: 根据PE分位动态调整，低估多买，高估少买
- **smart_pb**: 根据PB分位动态调整

**关键指标解读：**
- **CAGR**: 年化复合收益率越高越好
- **Sharpe**: 单位风险的回报，越高越好（衡量收益质量）
- **最大回撤**: 历史最大亏损幅度，越小越好
- **波动率**: 价格波动程度，越低越稳定

**选择策略的建议：**
1. 结合自身的风险承受能力
2. 关注 Sharpe 和 Sortino（风险调整后的收益）而非单纯的总收益
3. 在历史数据中表现好的策略不一定未来表现好
4. 定期回溯测试，随市场环境调整参数
5. 参数行示例：`smart_pe: low=2.0, high=0.5, lookback=1260`
""")
