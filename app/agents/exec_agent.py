import pandas as pd
import vectorbt as vbt
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import io
import contextlib
from app.state import AgentState
from langchain_core.messages import AIMessage

def exec_agent(state: AgentState):
    """
    Agent responsible for executing the generated code safely.
    """
    print("--- EXEC AGENT ---")
    code = state["strategy_code"]
    market_data = state["market_data"]
    
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
            exec(code, {}, local_scope)
            
        # Check if 'portfolio' was created
        if "portfolio" in local_scope:
            pf = local_scope["portfolio"]
            
            # Calculate metrics
            metrics = {
                "Total Return": pf.total_return().mean(), # Mean in case of multiple columns
                "Sharpe Ratio": pf.sharpe_ratio().mean(),
                "Max Drawdown": pf.max_drawdown().mean(),
                "Win Rate": pf.win_rate().mean()
            }
            
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
                "messages": [AIMessage(content="Strategy executed successfully.")]
            }
        else:
            return {
                "execution_output": output_buffer.getvalue(),
                "messages": [AIMessage(content="Error: The code did not define a 'portfolio' variable.")]
            }
            
    except Exception as e:
        return {
            "execution_output": output_buffer.getvalue() + f"\nError: {str(e)}",
            "messages": [AIMessage(content=f"Execution failed: {str(e)}")]
        }
