from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import uuid
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import streamlit as st
import io
import json
from app.llm import get_llm, invoke_llm_with_retry
from app.state import AgentState
from app.ui_utils import render_live_timer, display_token_usage

def analyst_agent(state: AgentState):
    """
    Agent responsible for summarizing the backtest results and generating visualizations.
    """
    print("--- ANALYST AGENT ---")
    st.write("🧐 **Analyst Agent:** Analyzing results...")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm = get_llm(provider=provider, model=model)
    
    metrics = state.get("performance_metrics", {})
    logs = state.get("execution_output", "")
    market_data = state.get("market_data", {})
    portfolio_data = state.get("portfolio_data", {})
    trades_data = state.get("trades_data", "[]")
    benchmark_data = state.get("benchmark_data", {})
    benchmark_metrics = state.get("benchmark_metrics", {})
    optimization_results = state.get("optimization_results", {})
    optimization_mode = state.get("optimization_mode", False)
    analysis_runs = state.get("analysis_runs", 0) or 0
    
    # --- Visualization Logic ---
    analyst_figures = []
    analyst_data = {}
    
    # 0. Parameter Optimization Results Visualization
    if optimization_mode and optimization_results:
        try:
            st.write("🔬 **Analyst Agent:** Visualizing parameter optimization results...")
            
            # Create optimization summary table
            opt_summary = {
                "Best Parameters": optimization_results.get("best_params", "N/A"),
                "Best Return": f"{optimization_results.get('best_return', 0):.2%}",
                "Best Sharpe": f"{optimization_results.get('best_sharpe', 0):.2f}",
                "Combinations Tested": optimization_results.get("total_combinations", "N/A")
            }
            analyst_data["Optimization Summary"] = pd.DataFrame([opt_summary])
            
        except Exception as e:
            print(f"Error processing optimization results: {e}")
    
    # 1. Professional Equity & Drawdown Analysis with Benchmark
    if portfolio_data:
        try:
            st.write("📊 **Analyst Agent:** Generating professional equity charts...")
            
            def _load_series(raw_json: str) -> pd.Series:
                try:
                    obj = json.loads(raw_json)
                    if isinstance(obj, dict) and "data" in obj and "index" in obj:
                        return pd.Series(obj["data"], index=pd.to_datetime(obj["index"]))
                except Exception:
                    pass
                try:
                    return pd.read_json(io.StringIO(raw_json), orient="split", typ="series")
                except Exception:
                    # Fallback: try DataFrame then first column
                    df = pd.read_json(io.StringIO(raw_json), orient="split")
                    if isinstance(df, pd.DataFrame):
                        return df.iloc[:, 0]
                    return pd.Series(dtype=float)

            val_series = _load_series(portfolio_data["value"])
            dd_series = _load_series(portfolio_data["drawdown"])

            # Create Subplots: Equity (Top) and Drawdown (Bottom)
            fig_equity = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                     vertical_spacing=0.05, 
                                     subplot_titles=("Strategy vs Benchmark (Cumulative Return)", "Drawdown (%)"),
                                     row_heights=[0.7, 0.3])
            
            # Normalize to cumulative returns for comparison
            strategy_cumret = (val_series / val_series.iloc[0] - 1) * 100
            fig_equity.add_trace(go.Scatter(x=val_series.index, y=strategy_cumret.values, 
                                          name="Strategy", line=dict(color='#00CC96', width=2)), row=1, col=1)
            
            # Add benchmark if available
            if benchmark_data:
                try:
                    bench_ticker = list(benchmark_data.keys())[0]
                    bench_df = benchmark_data[bench_ticker]
                    bench_close = bench_df['Close']
                    # Align with strategy dates
                    bench_close = bench_close.reindex(val_series.index, method='ffill')
                    bench_cumret = (bench_close / bench_close.iloc[0] - 1) * 100
                    fig_equity.add_trace(go.Scatter(x=bench_close.index, y=bench_cumret.values, 
                                                  name=f"Benchmark ({bench_ticker})", 
                                                  line=dict(color='#636EFA', width=2, dash='dash')), row=1, col=1)
                except Exception as e:
                    print(f"Error adding benchmark to chart: {e}")
            
            fig_equity.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100, 
                                          name="Drawdown", fill='tozeroy', line=dict(color='#EF553B', width=1)), row=2, col=1)
            
            fig_equity.update_layout(height=600, title_text="Strategy Performance Analysis", showlegend=True)
            fig_equity.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
            fig_equity.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
            analyst_figures.append(fig_equity)
            
        except Exception as e:
            print(f"Error generating equity chart: {e}")
    
    # 1.5 Benchmark Metrics Summary
    if benchmark_metrics:
        analyst_data["Benchmark Comparison Metrics"] = pd.DataFrame([benchmark_metrics])

    # 2. Trade Analysis
    if trades_data and trades_data != "[]":
        try:
            trades_df = pd.read_json(io.StringIO(trades_data), orient='records')
            if not trades_df.empty:
                analyst_data["Trade Records"] = trades_df
                
                # PnL Distribution
                if 'PnL' in trades_df.columns:
                    fig_pnl = px.histogram(trades_df, x="PnL", nbins=20, 
                                         title="Trade PnL Distribution", 
                                         marginal="box",
                                         color_discrete_sequence=['#636EFA'])
                    analyst_figures.append(fig_pnl)
        except Exception as e:
            print(f"Error processing trades: {e}")

    if market_data:
        # 3. K-Line Chart (Candlestick) for the first ticker
        # We only show K-Line if we have OHLC data
        first_ticker = list(market_data.keys())[0]
        df = market_data[first_ticker]
        
        if not df.empty and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            st.write(f"📊 **Analyst Agent:** Generating Technical Analysis chart for {first_ticker}...")
            
            # Calculate Moving Averages
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()
            
            # Create Subplots: Price (Top) and Volume (Bottom)
            fig_kline = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                                    vertical_spacing=0.03, 
                                    subplot_titles=(f"{first_ticker} Price & MA", "Volume"),
                                    row_heights=[0.7, 0.3])

            # Candlestick
            fig_kline.add_trace(go.Candlestick(x=df.index,
                            open=df['Open'], high=df['High'],
                            low=df['Low'], close=df['Close'],
                            name="OHLC"), row=1, col=1)
            
            # Moving Averages
            fig_kline.add_trace(go.Scatter(x=df.index, y=df['MA20'], 
                                         line=dict(color='orange', width=1), name="MA20"), row=1, col=1)
            fig_kline.add_trace(go.Scatter(x=df.index, y=df['MA60'], 
                                         line=dict(color='blue', width=1), name="MA60"), row=1, col=1)

            # Volume
            if 'Volume' in df.columns:
                colors = ['red' if row['Open'] - row['Close'] >= 0 else 'green' for index, row in df.iterrows()]
                fig_kline.add_trace(go.Bar(x=df.index, y=df['Volume'], 
                                         marker_color=colors, name="Volume"), row=2, col=1)

            fig_kline.update_layout(
                title=f"{first_ticker} Technical Analysis", 
                xaxis_rangeslider_visible=False,
                height=600,
                showlegend=True
            )
            analyst_figures.append(fig_kline)
            
            # Add data preview
            analyst_data[f"{first_ticker} Recent Data"] = df.tail(10)

        # 4. Correlation Heatmap (if multiple tickers)
        if len(market_data) > 1:
            # Combine Close prices
            close_prices = {}
            for ticker, data in market_data.items():
                if 'Close' in data.columns:
                    close_prices[ticker] = data['Close']
            
            if len(close_prices) > 1:
                combined_df = pd.DataFrame(close_prices)
                corr = combined_df.corr()
                
                fig_corr = px.imshow(corr, text_auto=True, title="Stock Correlation Matrix")
                analyst_figures.append(fig_corr)
                analyst_data["Correlation Matrix"] = corr

    # --- LLM Summary Logic ---
    # Determine which prompt to use based on mode
    if optimization_mode and optimization_results:
        # Optimization Mode Prompt
        system_prompt = """You are a Senior Quantitative Research Analyst specializing in strategy optimization.

## Task: Analyze Parameter Optimization Results

You are reviewing a parameter sweep optimization for a trading strategy.

### Optimization Results:
{optimization_info}

### Analysis Guidelines:
1. **最优参数分析**：
   - 识别最优参数组合及其指标表现
   - 分析参数敏感性（参数微调对收益的影响）
   
2. **参数稳健性**：
   - 评估参数是否过拟合（是否只在特定区间表现好）
   - 分析热力图中的"甜点区"分布
   
3. **风险警示**：
   - 指出可能的过拟合风险
   - 建议样本外测试验证
   
4. **实施建议**：
   - 给出推荐的参数范围（而非单一最优值）
   - 建议后续的验证步骤

### Base Metrics (if available):
{metrics}

### Market Data Info:
{data_info}

Output Requirements:
- 所有输出必须使用简体中文。
- 使用"参数优化报告"作为主标题。
"""
        
    elif benchmark_metrics:
        # Benchmark Comparison Mode Prompt  
        system_prompt = """You are a Senior Portfolio Analyst specializing in benchmark comparison and alpha analysis.

## Task: Analyze Strategy Performance vs Benchmark

### Benchmark Comparison Metrics:
{benchmark_info}

### Analysis Guidelines:
1. **相对表现评估**：
   - Alpha值解读（策略超额收益能力）
   - Beta分析（系统性风险暴露）
   - 信息比率评估（主动管理效率）
   
2. **风险特征对比**：
   - 跟踪误差分析
   - 相关性解读
   - 策略独立性评估
   
3. **归因分析**：
   - 超额收益来源分析
   - 择时vs选股贡献
   
4. **投资建议**：
   - 策略定位（替代基准/增强收益/绝对收益）
   - 风险预算建议

### Strategy Metrics:
{metrics}

### Market Data Info:
{data_info}

Output Requirements:
- 所有输出必须使用简体中文。
- 使用"策略对标分析报告"作为主标题。
"""
    else:
        # Standard Mode Prompt
        system_prompt = """You are a Senior Financial Analyst.

Scenario 1: Backtest Results Available
- If 'Metrics' contain values (not empty, not "{{}}", not "None"), interpret the backtest results and provide a comprehensive professional assessment.
- Focus on:
    * **风险调整收益**：Sharpe Ratio、Sortino Ratio、Calmar Ratio。
    * **回撤情况**：最大回撤的深度与持续时间。
    * **交易统计**：胜率、利润因子、期望收益、SQN 等。
- 报告结构：
    1. **执行摘要**（给出通过/需改进等结论）。
    2. **收益与风险分析**。
    3. **交易质量分析**。
    4. **风险评估**。

Scenario 2: Only Market Data or Metrics Missing
- 如果 'Metrics' 为空、为 "{{}}"、为 "None"，或总体缺失，则说明本次任务仅为行情/数据分析。
- 在这种情况下：
    * 禁止使用"策略"、"回测"等字眼；也不要编造不存在的执行结果。
    * 使用 **数据概览** 作为首个小节标题，聚焦以下要点：
        1. **趋势分析**：结合价格与均线判断多空趋势。
        2. **波动性**：描述价格波动幅度与稳定性。
        3. **关键价位**：指出重要支撑/阻力。
        4. **成交量分析**：若有成交量，描述量能变化。
        5. **短期观点**：给出偏多/偏空/中性判断及理由。

Output Requirements
- 所有输出必须使用简体中文。
- 根据所处场景合理设置段落标题，避免与事实不符的措辞。

Metrics: {metrics}
Logs: {logs}
Market Data Info: {data_info}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Please summarize the performance or data.")
    ])
    
    # Create a summary of data if available
    data_info = "No data available"
    if market_data:
        info = []
        for ticker, df in market_data.items():
            if not df.empty:
                start = df.index[0].strftime("%Y-%m-%d")
                end = df.index[-1].strftime("%Y-%m-%d")
                if 'Close' in df.columns:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    pct_change = ((end_price - start_price) / start_price) * 100
                    
                    # Basic Stats
                    high_price = df['High'].max() if 'High' in df.columns else 'N/A'
                    low_price = df['Low'].min() if 'Low' in df.columns else 'N/A'
                    volatility = df['Close'].pct_change().std() * (252**0.5) * 100 # Annualized Volatility
                    
                    info.append(f"""
                    Ticker: {ticker}
                    Range: {start} to {end}
                    Start: {start_price:.2f}, End: {end_price:.2f}
                    Change: {pct_change:.2f}%
                    High: {high_price}, Low: {low_price}
                    Annualized Volatility: {volatility:.2f}%
                    """)
        data_info = "\n".join(info)

    chain = prompt | llm
    
    # Build input variables based on the mode
    input_vars = {
        "metrics": str(metrics),
        "logs": logs,
        "data_info": data_info
    }
    
    # Add mode-specific variables
    if optimization_mode and optimization_results:
        # Format optimization results for the prompt
        opt_info_parts = []
        if isinstance(optimization_results, dict):
            if 'best_params' in optimization_results:
                opt_info_parts.append(f"最优参数: {optimization_results['best_params']}")
            if 'best_metrics' in optimization_results:
                opt_info_parts.append(f"最优指标: {optimization_results['best_metrics']}")
            if 'param_sweep_summary' in optimization_results:
                opt_info_parts.append(f"参数扫描摘要: {optimization_results['param_sweep_summary']}")
        input_vars["optimization_info"] = "\n".join(opt_info_parts) if opt_info_parts else str(optimization_results)
        
    elif benchmark_metrics:
        # Format benchmark metrics for the prompt
        benchmark_info_parts = []
        if isinstance(benchmark_metrics, dict):
            for key, value in benchmark_metrics.items():
                if isinstance(value, float):
                    benchmark_info_parts.append(f"{key}: {value:.4f}")
                else:
                    benchmark_info_parts.append(f"{key}: {value}")
        input_vars["benchmark_info"] = "\n".join(benchmark_info_parts) if benchmark_info_parts else str(benchmark_metrics)
    
    # Capture formatted messages
    formatted_messages = prompt.format_messages(**input_vars)
    formatted_prompt = "\n\n".join([f"**{m.type.upper()}**: {m.content}" for m in formatted_messages])
    
    with st.expander("🧐 Analyst Agent", expanded=True):
        timer = render_live_timer("⏳ Generating analysis summary...")
        try:
            response = invoke_llm_with_retry(chain, input_vars, max_retries=5, initial_delay=2.0)
        except Exception as e:
            timer.empty()
            st.error(f"❌ Failed to generate analysis after retries: {str(e)}")
            st.info("当前分析未能成功生成。请检查LLM服务状态或稍后重试。")
            raise
        timer.empty()

        # Show fallback/model info if available
        model_info = getattr(response, '_llm_provider', None)
        fallback_error = getattr(response, '_llm_fallback_error', None)
        if fallback_error:
            st.warning(f"⚠️ GitHub模型API报错/限流，已自动切换到DeepSeek。原始错误信息：{fallback_error}")
        if model_info:
            st.info(f"当前使用的Agent模型: {model_info}")

        display_token_usage(response)

        with st.popover("🧠 View Raw Prompt & Response"):
            st.markdown("**📝 Prompt:**")
            st.code(formatted_prompt, language="markdown")
            st.markdown("**💬 Response:**")
            st.code(response.content, language="markdown")

        st.markdown("#### 📝 Summary")
        st.markdown(response.content)

        if analyst_figures:
            st.markdown("#### 📊 Charts")
            unique_id = str(uuid.uuid4())[:8]
            for i, fig in enumerate(analyst_figures):
                st.plotly_chart(fig, use_container_width=True, key=f"analyst_chart_{unique_id}_{i}")

        if analyst_data:
            st.markdown("#### 📋 Data")
            for title, df in analyst_data.items():
                st.write(f"**{title}**")
                st.dataframe(df)

    return {
        "messages": [response],
        "sender": "analyst_agent",
        "analyst_figures": analyst_figures,
        "analyst_data": analyst_data,
        "analysis_completed": True,
        "analysis_runs": analysis_runs + 1,
        "llm_interaction": {
            "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
            "prompt": formatted_prompt,
            "response": response.content
        }
    }
