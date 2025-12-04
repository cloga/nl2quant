import pandas as pd
import vectorbt as vbt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import contextlib
import streamlit as st
from app.state import AgentState
from app.ui_utils import render_live_timer
from langchain_core.messages import AIMessage

def calculate_benchmark_metrics(strategy_returns, benchmark_returns):
    """
    Calculate Alpha, Beta, and other benchmark comparison metrics.
    """
    # Align the two series
    combined = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
    combined.columns = ['strategy', 'benchmark']
    combined = combined.dropna()
    
    if combined.empty or len(combined) < 30:
        return {}
    
    strat_ret = combined['strategy']
    bench_ret = combined['benchmark']
    
    # Beta = Cov(strategy, benchmark) / Var(benchmark)
    cov_matrix = np.cov(strat_ret, bench_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0
    
    # Alpha (annualized) = Strategy Return - Beta * Benchmark Return
    risk_free_rate = 0.02 / 252  # Assume 2% annual risk-free rate
    alpha_daily = strat_ret.mean() - risk_free_rate - beta * (bench_ret.mean() - risk_free_rate)
    alpha_annual = alpha_daily * 252
    
    # Tracking Error = Std(Strategy - Benchmark)
    tracking_diff = strat_ret - bench_ret
    tracking_error = tracking_diff.std() * np.sqrt(252)
    
    # Information Ratio = (Strategy Return - Benchmark Return) / Tracking Error
    excess_return = (strat_ret.mean() - bench_ret.mean()) * 252
    information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
    
    # Correlation
    correlation = strat_ret.corr(bench_ret)
    
    return {
        "Alpha (Annualized)": alpha_annual,
        "Beta": beta,
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio,
        "Correlation": correlation
    }

def exec_agent(state: AgentState):
    """
    Agent responsible for executing the generated code safely.
    Supports parameter optimization mode and benchmark comparison.
    """
    print("--- EXEC AGENT ---")
    
    # Check if code is confirmed (Human-in-the-Loop)
    if not state.get("code_confirmed", True):
        st.warning("⏸️ Code not confirmed yet. Please confirm the code in Quant Agent first.")
        return {
            "messages": [AIMessage(content="Waiting for code confirmation.")],
            "sender": "exec_agent",
            "next_step": "WAIT_FOR_CONFIRMATION"
        }
    
    optimization_mode = state.get("optimization_mode", False)
    
    with st.expander("⚙️ Exec Agent", expanded=True):
        timer = render_live_timer("⏳ Executing strategy...")
        
        st.write("⚙️ Preparing execution environment...")
        code = state["strategy_code"]
        market_data = state["market_data"]
        benchmark_data = state.get("benchmark_data", {})
        
        # Set default frequency for vectorbt to avoid "Index frequency is None" error
        vbt.settings.array_wrapper['freq'] = '1D'
        
        # Prepare the execution environment
        local_scope = {
            "vbt": vbt,
            "pd": pd,
            "np": np,
            "data_map": market_data,
            "plt": plt
        }
        
        output_buffer = io.StringIO()
        
        try:
            # Capture stdout
            with contextlib.redirect_stdout(output_buffer):
                st.write("🚀 Running strategy backtest...")
                exec(code, {}, local_scope)
                
            # Check if 'portfolio' was created
            if "portfolio" in local_scope:
                pf = local_scope["portfolio"]
                
                # --- Parameter Optimization Mode ---
                if optimization_mode and hasattr(pf, 'wrapper') and pf.wrapper.ndim > 1:
                    st.write("🔬 Processing parameter optimization results...")
                    
                    # Get all returns across parameter combinations
                    returns_df = pf.total_return()
                    sharpe_df = pf.sharpe_ratio()
                    
                    # Find best parameters
                    if isinstance(returns_df, pd.Series):
                        best_idx = returns_df.idxmax()
                        best_return = returns_df.max()
                        best_sharpe = sharpe_df.loc[best_idx] if isinstance(sharpe_df, pd.Series) else sharpe_df
                    else:
                        best_return = returns_df.max()
                        best_sharpe = sharpe_df.max()
                        best_idx = "N/A"
                    
                    # Extract param_names and param_values if defined
                    param_names = local_scope.get("param_names", [])
                    param_values = local_scope.get("param_values", {})
                    
                    optimization_results = {
                        "returns": returns_df.to_json() if hasattr(returns_df, 'to_json') else str(returns_df),
                        "sharpe": sharpe_df.to_json() if hasattr(sharpe_df, 'to_json') else str(sharpe_df),
                        "best_params": str(best_idx),
                        "best_return": float(best_return) if not pd.isna(best_return) else 0,
                        "best_sharpe": float(best_sharpe) if not pd.isna(best_sharpe) else 0,
                        "param_names": param_names,
                        "param_values": {k: list(v) for k, v in param_values.items()}
                    }
                    
                    metrics = {
                        "Best Parameters": str(best_idx),
                        "Best Total Return": best_return,
                        "Best Sharpe Ratio": best_sharpe,
                        "Total Combinations Tested": len(returns_df) if hasattr(returns_df, '__len__') else 1
                    }
                    
                    timer.empty()
                    
                    st.markdown("#### 🔬 Optimization Results")
                    st.json(metrics)
                    
                    # Generate heatmap if 2D parameter space
                    fig = None
                    if hasattr(returns_df, 'unstack') and returns_df.index.nlevels == 2:
                        try:
                            heatmap_data = returns_df.unstack()
                            fig = go.Figure(data=go.Heatmap(
                                z=heatmap_data.values,
                                x=[str(x) for x in heatmap_data.columns],
                                y=[str(y) for y in heatmap_data.index],
                                colorscale='RdYlGn',
                                text=np.round(heatmap_data.values * 100, 1),
                                texttemplate="%{text}%",
                                hovertemplate="Fast: %{y}<br>Slow: %{x}<br>Return: %{z:.2%}<extra></extra>"
                            ))
                            fig.update_layout(
                                title="Parameter Optimization Heatmap (Total Return)",
                                xaxis_title=param_names[1] if len(param_names) > 1 else "Param 2",
                                yaxis_title=param_names[0] if param_names else "Param 1"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.warning(f"Could not generate heatmap: {e}")
                    
                    return {
                        "execution_output": output_buffer.getvalue(),
                        "performance_metrics": metrics,
                        "optimization_results": optimization_results,
                        "figure_json": fig,
                        "messages": [AIMessage(content="Parameter optimization completed.")],
                        "sender": "exec_agent"
                    }
                
                # --- Standard Single Backtest Mode ---
                else:
                    st.write("🧮 Calculating extended performance metrics...")
                    
                    # Calculate extended metrics
                    metrics = {
                        "Total Return": pf.total_return().mean(),
                        "Annualized Return": pf.annualized_return().mean(),
                        "Sharpe Ratio": pf.sharpe_ratio().mean(),
                        "Sortino Ratio": pf.sortino_ratio().mean(),
                        "Calmar Ratio": pf.calmar_ratio().mean(),
                        "Max Drawdown": pf.max_drawdown().mean(),
                        "Omega Ratio": pf.omega_ratio().mean(),
                        "Win Rate": pf.trades.win_rate().mean(),
                        "Profit Factor": pf.trades.profit_factor().mean(),
                        "Expectancy": pf.trades.expectancy().mean(),
                        "SQN": pf.trades.sqn().mean(),
                        "Trade Count": int(pf.trades.count().mean()),
                        "Avg Trade Duration": str(pf.trades.duration.mean())
                    }
                    
                    # --- Benchmark Comparison ---
                    benchmark_metrics = {}
                    if benchmark_data:
                        st.write("📊 Calculating benchmark comparison metrics...")
                        try:
                            strategy_returns = pf.daily_returns()
                            if isinstance(strategy_returns, pd.DataFrame):
                                strategy_returns = strategy_returns.iloc[:, 0]
                            
                            bench_ticker = list(benchmark_data.keys())[0]
                            bench_df = benchmark_data[bench_ticker]
                            benchmark_returns = bench_df['Close'].pct_change().dropna()
                            
                            benchmark_metrics = calculate_benchmark_metrics(strategy_returns, benchmark_returns)
                            
                            if benchmark_metrics:
                                st.markdown("#### 📈 Benchmark Comparison")
                                st.json(benchmark_metrics)
                        except Exception as e:
                            st.warning(f"Could not calculate benchmark metrics: {e}")
                    
                    # Extract Time Series Data for Analyst Agent
                    portfolio_data = {
                        "value": pf.value().to_json(date_format='iso', orient='split'),
                        "drawdown": pf.drawdown().to_json(date_format='iso', orient='split'),
                        "daily_returns": pf.daily_returns().to_json(date_format='iso', orient='split')
                    }
                    
                    # Extract Trade Records
                    try:
                        trades_df = pf.trades.records_readable
                        trades_json = trades_df.to_json(orient='records', date_format='iso')
                    except Exception as e:
                        print(f"Error extracting trades: {e}")
                        trades_json = "[]"
                    
                    timer.empty()
                    
                    st.markdown("#### 📊 Performance Metrics")
                    st.json(metrics)
                    
                    # Generate Plotly Figure
                    fig = pf.plot()
                    
                    return {
                        "execution_output": output_buffer.getvalue(),
                        "performance_metrics": metrics,
                        "benchmark_metrics": benchmark_metrics,
                        "portfolio_data": portfolio_data,
                        "trades_data": trades_json,
                        "figure_json": fig,
                        "messages": [AIMessage(content="Strategy executed successfully.")],
                        "sender": "exec_agent"
                    }
            else:
                timer.empty()
                st.error("Error: The code did not define a 'portfolio' variable.")
                return {
                    "execution_output": output_buffer.getvalue() + "\nError: The code did not define a 'portfolio' variable.",
                    "messages": [AIMessage(content="Error: The code did not define a 'portfolio' variable.")],
                    "sender": "exec_agent"
                }
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            timer.empty()
            st.error(f"Execution failed: {str(e)}")
            st.code(tb)
            return {
                "execution_output": output_buffer.getvalue() + f"\nExecution Error:\n{tb}",
                "messages": [AIMessage(content=f"Execution failed: {str(e)}")],
                "sender": "exec_agent"
            }
