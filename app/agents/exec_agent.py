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

def exec_agent(state: AgentState):
    """
    Agent responsible for executing the generated code safely.
    """
    print("--- EXEC AGENT ---")
    
    with st.expander("⚙️ Exec Agent", expanded=True):
        timer = render_live_timer("⏳ Executing strategy...")
        
        st.write("⚙️ Preparing execution environment...")
        code = state["strategy_code"]
        market_data = state["market_data"]
        
        # Set default frequency for vectorbt to avoid "Index frequency is None" error
        # This is a fallback in case the generated code doesn't specify it.
        vbt.settings.array_wrapper['freq'] = '1D'
        
        # Prepare the execution environment
        # We inject 'vbt', 'pd', 'np' and the 'data_map'
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
                
                st.write("🧮 Calculating performance metrics (Sharpe, Drawdown)...")
                # Calculate metrics
                metrics = {
                    "Total Return": pf.total_return().mean(), # Mean in case of multiple columns
                    "Sharpe Ratio": pf.sharpe_ratio().mean(),
                    "Max Drawdown": pf.max_drawdown().mean(),
                    "Win Rate": pf.trades.win_rate().mean()
                }
                
                timer.empty() # Clear timer
                
                st.markdown("#### 📊 Performance Metrics")
                st.json(metrics)
                
                # Generate Plotly Figure
                # VectorBT plots are usually Plotly objects
                fig = pf.plot()
                
                # Convert to JSON for Streamlit to render (or pass object if within same process)
                # Since we are in the same process, we can just store the object reference or JSON
                # For state serialization, JSON is safer, but for MVP object is fine.
                # Let's store the object in a temporary way or just the JSON.
                # fig_json = fig.to_json() 
                
                return {
                    "execution_output": output_buffer.getvalue(),
                    "performance_metrics": metrics,
                    "figure_json": fig, # Storing the actual Figure object for now for simplicity in Streamlit
                    "messages": [AIMessage(content="Strategy executed successfully.")],
                    "sender": "exec_agent"
                }
            else:
                st.error("Error: The code did not define a 'portfolio' variable.")
                return {
                    "execution_output": output_buffer.getvalue() + "\nError: The code did not define a 'portfolio' variable.",
                    "messages": [AIMessage(content="Error: The code did not define a 'portfolio' variable.")],
                    "sender": "exec_agent"
                }
                
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            st.error(f"Execution failed: {str(e)}")
            st.code(tb)
            return {
                "execution_output": output_buffer.getvalue() + f"\nExecution Error:\n{tb}",
                "messages": [AIMessage(content=f"Execution failed: {str(e)}")],
                "sender": "exec_agent"
            }
