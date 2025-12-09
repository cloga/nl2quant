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
import json
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.dca_backtest_engine import DCABacktestEngine
from app.llm import get_llm
from app.agents.analyst import analyst_agent
from app.state import AgentState

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
# Configuration (aligned with DCA Backtest)
# ============================================================================
st.sidebar.title("📊 定投回测配置")

# Asset Selection (same structure as DCA_Backtest)
st.sidebar.markdown("### 标的选择")
asset_type = st.sidebar.selectbox(
    "选择标的类型",
    ["基金", "指数", "股票"],
    index=1,
    help="选择要回测的资产类型",
)

asset_examples = {
    "基金": {
        "name": "沪深300ETF",
        "code": "510300",
        "description": "沪深300指数基金",
        "help": "基金代码示例：510300(沪深300)、159915(创业板ETF)、512100(中证1000ETF)",
    },
    "指数": {
        "name": "中证红利指数",
        "code": "000922",
        "description": "中证红利指数",
        "help": "指数代码示例：000922(中证红利)、000300(沪深300指数)、399006(创业板指数)",
    },
    "股票": {
        "name": "长江电力",
        "code": "600900",
        "description": "长江电力股票",
        "help": "股票代码示例：600900(长江电力)、600519(贵州茅台)、000858(五粮液)",
    },
}

example = asset_examples[asset_type]
asset_code = st.sidebar.text_input(
    f"输入{asset_type}代码或名称",
    value=example["code"],
    help=example["help"],
)

price_mode = st.sidebar.selectbox(
    "复权类型",
    ["后复权", "前复权", "不复权"],
    index=0,
    help="建议选择后复权，包含分红再投资的收益",
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
        "首期底仓 (元)",
        min_value=0.0,
        max_value=10000000.0,
        value=0.0,
        step=10000.0,
        help="首期建仓资金；设为 0 则仅做定投",
    )

    risk_free_rate = st.slider(
        "闲置资金年化收益率 (%)",
        min_value=0.0,
        max_value=10.0,
        value=2.5,
        step=0.1,
        help="账户中未投资的现金享受的理财收益率",
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
low_multiplier = 2.0
high_multiplier = 0.5
lookback_days = 252 * 5
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
            return_calc_method = st.radio(
                "收益计算方式",
                ["holdings_only", "total_portfolio"],
                format_func=lambda x: {"holdings_only": "持仓收益百分比", "total_portfolio": "总仓位收益百分比"}[x],
                index=0,
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
                "drawdown_threshold": drawdown_threshold,
            }

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
            if trailing_params is None:
                trailing_params = {}
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
            if trailing_params is None:
                trailing_params = {}
            trailing_params["reentry_mode"] = "price"
            trailing_params["reentry_drop"] = reentry_drop

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

# Profile config snapshot for combination guidance
profile_config = {
    "code": codes[0],
    "price_mode": price_mode,
    "strategy_type": strategy_type,
    "monthly_investment": monthly_investment,
    "rebalance_freq": rebalance_freq,
    "freq_day": freq_day,
    "initial_capital": initial_capital,
    "risk_free_rate": risk_free_rate,
    "max_total_investment": max_total_investment,
    "smart_params": smart_params,
    "commission_rate": commission_rate,
    "min_commission": min_commission,
    "slippage": slippage,
    "take_profit_enabled": enable_take_profit,
    "take_profit": trailing_params,
    "date_range": {
        "start": start_date.isoformat() if start_date else None,
        "end": end_date.isoformat() if end_date else None,
    },
}

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

with st.expander("当前 Profile 配置（用于参数组合/LLM 参考）", expanded=False):
    st.json(profile_config)
    st.caption("该快照基于左侧导航当前选择，可供参数组合与 LLM 生成时参考。")

# Parameter grid for profiling using JSON
st.markdown("### 🧪 参数组合（JSON 定义）")
st.caption(
    "使用 JSON 描述待遍历的策略组合。默认提供多档止盈的 plain 组合，可手动编辑、同步当前 Profile 或让 LLM 生成；"
    "可选字段 take_profit 用于单条组合独立的止盈/风控参数。"
)

# Default JSON seed: multiple take-profit variants for plain strategy
default_profiler_json = json.dumps(
    {
        "strategies": [
            {
                "label": "plain_tp4%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.04, "return_calc_method": "total_portfolio"},
            },
            {
                "label": "plain_tp6%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.06, "return_calc_method": "total_portfolio"},
            },
            {
                "label": "plain_tp8%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.08, "return_calc_method": "total_portfolio"},
            },
            {
                "label": "plain_tp10%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.10, "return_calc_method": "total_portfolio"},
            },
            {
                "label": "plain_tp12%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.12, "return_calc_method": "total_portfolio"},
            },
            {
                "label": "plain_tp15%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.15, "return_calc_method": "total_portfolio"},
            },
            {
                "label": "plain_tp20%",
                "strategy": "plain",
                "take_profit": {"mode": "target", "target_return": 0.20, "return_calc_method": "total_portfolio"},
            },
        ]
    },
    ensure_ascii=False,
    indent=2,
)

# Initialize editable JSON with default seed once
if "profiler_combinations_json" not in st.session_state:
    st.session_state.profiler_combinations_json = default_profiler_json

st.caption(
    "JSON 结构示例：{ 'strategies': [ { 'label': 'plain_tp6%', 'strategy': 'plain',"
    " 'take_profit': { 'mode':'target','target_return':0.06,'return_calc_method':'total_portfolio' } } ] }"
)

combination_json_text = st.text_area(
    "策略组合 JSON（可编辑）",
    key="profiler_combinations_json",
    height=220,
)

col_json_btn1, col_json_btn2, col_json_btn3 = st.columns([1, 1, 1])
with col_json_btn1:
    if st.button("格式化 JSON", use_container_width=True):
        try:
            obj = json.loads(st.session_state.get("profiler_combinations_json", ""))
            st.session_state.profiler_combinations_json = json.dumps(obj, ensure_ascii=False, indent=2)
            st.success("已格式化")
        except Exception as e:
            st.warning(f"无法格式化：{e}")

with col_json_btn2:
    if st.button("恢复默认止盈样例", use_container_width=True):
        st.session_state.profiler_combinations_json = default_profiler_json
        st.info("已恢复默认样例")

with col_json_btn3:
    if st.button("同步当前 Profile 到 JSON", use_container_width=True):
        # Build a single-entry profile-based JSON for convenience
        current_entry = {"label": f"{strategy_type}_current", "strategy": strategy_type}
        if strategy_type in ["smart_pe", "smart_pb"]:
            current_entry["params"] = {"low": low_multiplier, "high": high_multiplier, "lookback": lookback_days}
        st.session_state.profiler_combinations_json = json.dumps({"strategies": [current_entry]}, ensure_ascii=False, indent=2)
        st.info("已按当前配置生成 JSON 组合")

with st.expander("🤖 让 LLM 生成 JSON 参数组合", expanded=False):
    st.caption("基于当前配置，自动生成若干条可直接回测的 JSON 组合，可包含 per-strategy 的 take_profit。")
    llm_combo_count = st.slider("需要几条组合", 3, 10, 5, step=1, help="生成的策略条目数量")
    llm_temperature = st.slider("创意程度 (temperature)", 0.0, 1.0, 0.2, step=0.05)
    if st.button("生成 JSON 参数组合", use_container_width=True):
        prompt = f"""
你是一名量化定投策略助理，请围绕普通定投、智能PE定投、智能PB定投生成一个 JSON，字段结构如下：
{{
  "strategies": [
    {{ "label": "plain", "strategy": "plain" }},
        {{ "label": "smart_pe_low", "strategy": "smart_pe", "params": {{"low": 2.0, "high": 0.5, "lookback": 1260}},
             "take_profit": {{"mode": "target", "target_return": 0.05, "return_calc_method": "total_portfolio"}} }}
  ]
}}

要求：
1) 生成 {llm_combo_count} 条组合，写入 strategies 数组。
2) strategy 只能是 plain、smart_pe、smart_pb。
3) params 仅对 smart_pe/smart_pb 提供，包含 low、high、lookback；lookback 用交易日天数，建议是 252 的倍数。
4) 可选字段 take_profit：
     - target 模式示例: {"mode":"target","target_return":0.05,"return_calc_method":"total_portfolio"}
     - trailing 模式示例: {"mode":"trailing","activation_return":0.30,"drawdown_threshold":0.08,"reentry_mode":"time","reentry_days":3}
     - 若未提供 take_profit，则使用侧栏的默认止盈/风控设置。
5) 数字使用阿拉伯数字，不要中文单位；不要输出除 JSON 以外的任何解释或前后缀。

当前配置快照（仅供参考，可灵活给出不同组合）:
{json.dumps(profile_config, ensure_ascii=False)}
"""
        try:
            llm = get_llm(temperature=llm_temperature)
            response = llm.invoke(prompt)
            content = getattr(response, "content", None) or str(response)
            cleaned = content.strip()
            if cleaned:
                st.session_state.profiler_combinations_json = cleaned
                st.success("已生成并填入 JSON 组合，可直接运行。")
            else:
                st.warning("LLM 返回为空，请重试或手动填写。")
        except Exception as e:
            st.warning(f"无法生成参数组合：{e}")

# Live JSON validation feedback
try:
    preview_obj = json.loads(st.session_state.get("profiler_combinations_json", "{}"))
    strategies_preview = preview_obj.get("strategies", []) if isinstance(preview_obj, dict) else []
    st.success(f"✅ JSON 可用，策略条目: {len(strategies_preview)}")
except Exception as e:
    st.warning(f"⚠️ 当前 JSON 无法解析：{e}")

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
    results_dict = {}
    if start_date is not None and end_date is not None and start_date >= end_date:
        st.error("❌ 开始日期必须早于结束日期")
    else:
        if start_date is None:
            start_date_str = "19900101"
        else:
            start_date_str = start_date.strftime("%Y%m%d")
        end_date_str = end_date.strftime("%Y%m%d")

        # Parse parameter combinations from JSON
        combinations = []
        parse_errors = []
        try:
            combo_data = json.loads(st.session_state.get("profiler_combinations_json", default_profiler_json))
            strategies = combo_data.get("strategies", []) if isinstance(combo_data, dict) else []
            if not isinstance(strategies, list):
                raise ValueError("'strategies' 应为数组")

            for entry in strategies:
                if not isinstance(entry, dict):
                    parse_errors.append("每个组合需为对象：包含 strategy/label/params/take_profit")
                    continue
                strategy = entry.get("strategy")
                label = entry.get("label") or strategy
                params = entry.get("params")
                take_profit_override = entry.get("take_profit")
                if take_profit_override is not None and not isinstance(take_profit_override, dict):
                    parse_errors.append(f"take_profit 必须为对象: {entry}")
                    continue
                if not strategy:
                    parse_errors.append(f"组合缺少 strategy 字段: {entry}")
                    continue
                combinations.append({
                    "label": label or strategy,
                    "strategy": strategy,
                    "params": params,
                    "take_profit": take_profit_override,
                })
        except Exception as e:
            parse_errors.append(f"JSON 解析失败: {e}")

        if parse_errors:
            st.error("参数解析错误:\n" + "\n".join(parse_errors))
            st.stop()
        elif not combinations:
            st.warning("未提供任何参数组合")
            st.stop()
        else:
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

                        combo_take_profit = combo.get("take_profit")
                        if combo_take_profit is None and enable_take_profit:
                            combo_take_profit = trailing_params

                        result = engine.run_smart_dca_backtest(
                            code=codes[0],
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
                            trailing_params=combo_take_profit,
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

        # Buy & Hold benchmark using same total investment as first strategy (if possible)
        # We approximate by taking the first strategy's total_invested and applying it as lump-sum on first price.
        # This aligns with backtest context where lump_sum benchmark is used.
        bh_row = None
        try:
            first_result = next(iter(results_dict.values()))
            first_metrics = first_result.get("metrics", {})
            total_invested_first = first_metrics.get("total_invested", 0)
            price_series = first_result.get("result", {}).get("price_series") if isinstance(first_result.get("result"), dict) else None
            if total_invested_first and price_series is not None and hasattr(price_series, "empty") and not price_series.empty:
                start_price = price_series.iloc[0]
                shares = total_invested_first / start_price if start_price else 0
                bh_equity = price_series * shares
                final_bh = bh_equity.iloc[-1]
                total_return_bh = (final_bh - total_invested_first) / total_invested_first * 100 if total_invested_first else 0
                total_days = first_metrics.get("total_days", 0) or 0
                cagr_bh = None
                if total_days > 0:
                    try:
                        rtn = total_return_bh / 100
                        cagr_bh = ((1 + rtn) ** (365 / total_days) - 1) * 100
                    except Exception:
                        cagr_bh = None
                bh_row = {
                    "策略": "Buy&Hold",
                    "策略类型": "benchmark",
                    "总投资额 (¥)": total_invested_first,
                    "期末资产 (¥)": final_bh,
                    "总收益率": total_return_bh,
                    "CAGR": cagr_bh if cagr_bh is not None else 0,
                    "年化波动": None,
                    "Sharpe": None,
                    "Sortino": None,
                    "最大回撤": None,
                    "Calmar": None,
                }
        except Exception:
            bh_row = None

        if bh_row:
            comparison_data.append(bh_row)

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
        palette = px.colors.qualitative.Plotly + px.colors.qualitative.Safe + px.colors.qualitative.Pastel
        
        for idx, (strategy_name, data) in enumerate(results_dict.items()):
            equity_curve = data["result"]["equity_curve"]
            color = palette[idx % len(palette)]
            fig_equity.add_trace(
                go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode="lines",
                    name=strategy_name,
                    line=dict(color=color, width=2),
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

        # ================================================================
        # Analyst Agent (LLM-based insight across strategies)
        # ================================================================
        try:
            best_row = comparison_df.iloc[0]
            best_label = best_row["策略"]
            best_strategy_code = results_dict[best_label]["code"]
            best_metrics = results_dict[best_label]["metrics"]
            best_result = results_dict[best_label]["result"]

            equity_curve = best_result.get("equity_curve") if isinstance(best_result, dict) else None
            transactions_df = best_result.get("transactions") if isinstance(best_result, dict) else None

            portfolio_data = {}
            trades_json = "[]"
            if equity_curve is not None and len(equity_curve) > 1:
                drawdown_series = equity_curve / equity_curve.cummax() - 1
                portfolio_data = {
                    "value": equity_curve.to_json(date_format="iso", orient="split"),
                    "drawdown": drawdown_series.to_json(date_format="iso", orient="split"),
                }

            if transactions_df is not None and hasattr(transactions_df, "empty") and not transactions_df.empty:
                trades_json = transactions_df.to_json(orient="records", date_format="iso")

            opt_summary = {
                "best_params": best_label,
                "best_metrics": {k: v for k, v in best_metrics.items() if k in ["cagr_pct", "total_return_pct", "sharpe_ratio", "max_drawdown_pct"]},
                "param_sweep_summary": f"共 {len(results_dict)} 组，按 {rank_metric} 排序",
            }

            agent_state: AgentState = {
                "messages": [],
                "tickers": [codes[0]],
                "benchmark_ticker": None,
                "start_date": start_date_str,
                "end_date": end_date_str,
                "market_data": {},
                "benchmark_data": {},
                "strategy_code": best_strategy_code,
                "user_edited_code": None,
                "code_confirmed": True,
                "optimization_mode": True,
                "optimization_params": None,
                "optimization_results": opt_summary,
                "execution_output": "",
                "performance_metrics": best_metrics,
                "portfolio_data": portfolio_data,
                "trades_data": trades_json,
                "figure_json": None,
                "benchmark_metrics": {},
                "analyst_figures": None,
                "analyst_data": None,
                "analysis_completed": None,
                "analysis_runs": len(results_dict),
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

            st.markdown("### 🧐 Analyst Agent 专业意见")
            analyst_agent(agent_state)
        except Exception as e:
            st.warning(f"⚠️ Analyst Agent 运行失败：{e}")

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
