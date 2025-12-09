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
    Calculate a comprehensive set of benchmark and relative metrics so that
    benchmark comparison surfaces the same depth as strategy metrics.
    """
    # Align the two series
    combined = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner")
    combined.columns = ["strategy", "benchmark"]
    combined = combined.dropna()

    if combined.empty or len(combined) < 10:
        return {}

    strat_ret = combined["strategy"]
    bench_ret = combined["benchmark"]

    rf_daily = 0.02 / 252  # Assume 2% annual risk-free rate
    rf_annual = 0.02

    def perf_from_returns(ret: pd.Series):
        total_return = (1 + ret).prod() - 1
        years = len(ret) / 252 if len(ret) > 0 else 0
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        vol = ret.std() * np.sqrt(252) if len(ret) > 1 else 0
        downside = ret[ret < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0

        sharpe = (annual_return - rf_annual) / vol if vol > 0 else 0
        sortino = (annual_return - rf_annual) / downside_vol if downside_vol > 0 else 0

        equity_curve = (1 + ret).cumprod()
        rolling_max = equity_curve.cummax()
        drawdown = equity_curve / rolling_max - 1
        max_dd = drawdown.min() if not drawdown.empty else 0
        calmar = annual_return / abs(max_dd) if max_dd != 0 else 0

        win_rate = (ret > 0).mean() if len(ret) > 0 else 0
        pos_sum = ret[ret > 0].sum()
        neg_sum = ret[ret < 0].sum()
        loss_denom = abs(neg_sum)
        profit_factor = pos_sum / loss_denom if loss_denom > 0 else 0
        omega = pos_sum / loss_denom if loss_denom > 0 else 0

        expectancy = ret.mean() if len(ret) > 0 else 0
        sqn = (expectancy / ret.std()) * np.sqrt(len(ret)) if ret.std() > 0 else 0

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "omega_ratio": omega,
            "expectancy": expectancy,
            "sqn": sqn,
            "observations": len(ret),
        }

    strat_perf = perf_from_returns(strat_ret)
    bench_perf = perf_from_returns(bench_ret)

    # Relative metrics
    cov_matrix = np.cov(strat_ret, bench_ret)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else 0

    alpha_daily = strat_ret.mean() - rf_daily - beta * (bench_ret.mean() - rf_daily)
    alpha_annual = alpha_daily * 252

    tracking_diff = strat_ret - bench_ret
    tracking_error = tracking_diff.std() * np.sqrt(252)

    excess_return_annual = strat_perf["annual_return"] - bench_perf["annual_return"]
    active_return_annual = tracking_diff.mean() * 252
    information_ratio = (
        active_return_annual / tracking_error if tracking_error != 0 else 0
    )

    correlation = strat_ret.corr(bench_ret)
    r_squared = correlation**2 if correlation is not None else 0

    return {
        # Benchmark absolute metrics
        "Benchmark Total Return": bench_perf["total_return"],
        "Benchmark Annualized Return": bench_perf["annual_return"],
        "Benchmark Volatility": bench_perf["volatility"],
        "Benchmark Sharpe Ratio": bench_perf["sharpe"],
        "Benchmark Sortino Ratio": bench_perf["sortino"],
        "Benchmark Calmar Ratio": bench_perf["calmar"],
        "Benchmark Max Drawdown": bench_perf["max_drawdown"],
        "Benchmark Win Rate": bench_perf["win_rate"],
        "Benchmark Profit Factor": bench_perf["profit_factor"],
        "Benchmark Omega Ratio": bench_perf["omega_ratio"],
        "Benchmark Expectancy": bench_perf["expectancy"],
        "Benchmark SQN": bench_perf["sqn"],
        "Benchmark Observations": bench_perf["observations"],

        # Relative metrics
        "Excess Total Return": strat_perf["total_return"] - bench_perf["total_return"],
        "Excess Annualized Return": excess_return_annual,
        "Alpha (Annualized)": alpha_annual,
        "Beta": beta,
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio,
        "Correlation": correlation,
        "R-Squared": r_squared,
    }

def exec_agent(state: AgentState):
    """
    Agent responsible for executing the generated code safely.
    Supports parameter optimization mode and benchmark comparison.
    """
    print("--- EXEC AGENT ---")
    
    optimization_mode = state.get("optimization_mode", False)
    
    with st.expander("⚙️ Exec Agent", expanded=True):
        timer = render_live_timer("⏳ Executing strategy...")
        
        st.write("⚙️ Preparing execution environment...")
        code = state["strategy_code"]
        market_data = state["market_data"]
        benchmark_data = state.get("benchmark_data", {})

        # Display the exact code that will be executed for transparency/debugging
        with st.expander("💻 Strategy Code (to execute)", expanded=False):
            st.code(code, language="python")
        
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
