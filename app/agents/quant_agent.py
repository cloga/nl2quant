from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import streamlit as st
from app.llm import get_llm
from app.state import AgentState
from app.ui_utils import render_live_timer, display_token_usage

def quant_agent(state: AgentState):
    """
    Agent responsible for generating VectorBT strategy code.
    Supports parameter optimization mode and human-in-the-loop editing.
    """
    print("--- QUANT AGENT ---")
    st.write("🧠 **Quant Agent:** Designing strategy logic...")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm = get_llm(provider=provider, model=model)
    
    messages = state["messages"]
    # Extract the user's latest request (or the accumulated context)
    first_msg = messages[0]
    if isinstance(first_msg, dict):
        user_request = first_msg.get("content", "")
    else:
        user_request = getattr(first_msg, "content", "")
    
    data_keys = list(state.get("market_data", {}).keys())
    
    # Check if user wants parameter optimization
    optimization_mode = state.get("optimization_mode", False)
    optimization_keywords = ["优化", "optimize", "参数扫描", "parameter sweep", "最优参数", "best parameter", "网格搜索", "grid search"]
    if any(kw in user_request.lower() for kw in optimization_keywords):
        optimization_mode = True
    
    # Check for feedback/errors from previous runs
    exec_output = state.get("execution_output", "")
    feedback = ""
    if exec_output and ("Error" in exec_output or "Traceback" in exec_output):
        feedback = f"\n\nPREVIOUS EXECUTION FAILED. FIX THE CODE BASED ON THIS ERROR:\n{exec_output[-1000:]}"
    
    # Choose prompt based on optimization mode
    if optimization_mode:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python Quant Developer using the `vectorbt` library.
            Your goal is to generate executable Python code that performs a PARAMETER OPTIMIZATION for a trading strategy.
            
            Context:
            - The data is already loaded into a dictionary named `data_map`.
            - `data_map` keys are tickers: {tickers}
            - Each value in `data_map` is a Pandas DataFrame with columns: Open, High, Low, Close, Volume.
            - The index is a DatetimeIndex.
            
            Requirements for Parameter Optimization:
            1. Use `vectorbt` (imported as `vbt`) for the backtest.
            2. Define parameter ranges using numpy arrays (e.g., `fast_windows = np.arange(5, 25, 5)`, `slow_windows = np.arange(20, 60, 10)`).
            3. Use vectorbt's broadcasting capability to test all parameter combinations at once.
            4. The code MUST define:
               - `portfolio`: The result of `vbt.Portfolio.from_signals(...)` with parameter combinations.
               - `param_names`: A list of parameter names, e.g., `['fast_window', 'slow_window']`
               - `param_values`: A dict mapping param names to their tested values, e.g., {{'fast_window': fast_windows, 'slow_window': slow_windows}}
            5. ALWAYS pass `freq='1D'` to avoid frequency inference errors.
            6. Return ONLY the Python code. No markdown formatting, no ```python blocks.
            
            Example for MA Crossover Optimization:
            price = data_map['600519.SH']['Close']
            fast_windows = np.arange(5, 25, 5)
            slow_windows = np.arange(20, 60, 10)
            fast_ma, slow_ma = vbt.MA.run_combs(price, window=fast_windows, r=2, short_names=['fast', 'slow'])
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
            param_names = ['fast_window', 'slow_window']
            param_values = {{'fast_window': fast_windows, 'slow_window': slow_windows}}
            """),
            ("user", "{input}{feedback}")
        ])
    else:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Python Quant Developer using the `vectorbt` library.
            Your goal is to generate executable Python code that backtests a trading strategy described by the user.
            
            Context:
            - The data is already loaded into a dictionary named `data_map`.
            - `data_map` keys are tickers: {tickers}
            - Each value in `data_map` is a Pandas DataFrame with columns: Open, High, Low, Close, Volume.
            - The index is a DatetimeIndex.
            
            Requirements:
            1. Use `vectorbt` (imported as `vbt`) for the backtest.
            2. The code MUST define a variable named `portfolio` which is the result of `vbt.Portfolio.from_signals(...)` or similar.
            3. Do NOT fetch data. Use `data_map['TICKER']` to access data.
            4. If multiple tickers are present, handle them appropriately (or just pick the first one if the strategy is single-asset).
            5. ALWAYS pass `freq='1D'` to `vbt.Portfolio.from_signals` or `from_orders` to avoid frequency inference errors with daily data.
            6. Return ONLY the Python code. No markdown formatting, no ```python blocks.
            7. Ensure the code is safe and does not use system calls.
            
            Example Snippet:
            price = data_map['600519.SH']['Close']
            fast_ma = vbt.MA.run(price, 10)
            slow_ma = vbt.MA.run(price, 50)
            entries = fast_ma.ma_crossed_above(slow_ma)
            exits = fast_ma.ma_crossed_below(slow_ma)
            portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
            """),
            ("user", "{input}{feedback}")
        ])
    
    chain = prompt | llm
    
    input_vars = {
        "input": user_request,
        "tickers": data_keys,
        "feedback": feedback
    }
    
    # Capture formatted messages
    formatted_messages = prompt.format_messages(**input_vars)
    formatted_prompt = "\n\n".join([f"**{m.type.upper()}**: {m.content}" for m in formatted_messages])
    
    with st.expander("📈 Quant Agent", expanded=True):
        timer = render_live_timer("⏳ Generating strategy code...")
        response = chain.invoke(input_vars)
        timer.empty()
        
        display_token_usage(response)
        
        with st.expander("🧠 View Raw Prompt & Response", expanded=False):
            st.markdown("**📝 Prompt:**")
            st.code(formatted_prompt, language="markdown")
            st.markdown("**💬 Response:**")
            st.code(response.content, language="python")

        code = response.content.strip()
        
        # Clean up markdown if the LLM ignored instructions
        if code.startswith("```python"):
            code = code.split("```python")[1]
        if code.startswith("```"):
            code = code.split("```")[1]
        if code.endswith("```"):
            code = code.rsplit("```", 1)[0]
        
        code = code.strip()
        
        # --- Human-in-the-Loop Code Editing ---
        st.markdown("#### 💻 Generated Code (Editable)")
        st.info("💡 **Tip:** You can edit the code below before execution. Click 'Confirm & Execute' when ready.")
        
        # Use session state to track code editing
        code_key = f"edited_code_{hash(code)}"
        if code_key not in st.session_state:
            st.session_state[code_key] = code
        
        edited_code = st.text_area(
            "Strategy Code",
            value=st.session_state[code_key],
            height=400,
            key=f"code_editor_{hash(code)}",
            label_visibility="collapsed"
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Reset to Original", key=f"reset_{hash(code)}"):
                st.session_state[code_key] = code
                st.rerun()
        
        with col2:
            confirm_button = st.button("✅ Confirm & Execute", key=f"confirm_{hash(code)}", type="primary")
        
        if confirm_button:
            final_code = edited_code.strip()
            st.session_state[code_key] = final_code
            st.success("Code confirmed! Proceeding to execution...")
            
            # Mask sensitive fields in input_vars before storing
            safe_input = dict(input_vars)
            for kk in list(safe_input.keys()):
                lk = kk.lower()
                if 'token' in lk or 'api_key' in lk or 'secret' in lk or 'password' in lk:
                    safe_input[kk] = '***MASKED***'

            return {
                "strategy_code": final_code,
                "user_edited_code": final_code if final_code != code else None,
                "code_confirmed": True,
                "optimization_mode": optimization_mode,
                "messages": [AIMessage(content="Generated strategy code (user confirmed).")],
                "sender": "quant_agent",
                "llm_interaction": {
                    "input": safe_input,
                    "prompt": formatted_prompt,
                    "response": response.content
                }
            }
        else:
            # If not confirmed, we still need to return something to allow the graph to proceed
            # We'll mark code_confirmed as False so planner knows to wait
            st.warning("⏸️ Waiting for code confirmation... Click 'Confirm & Execute' to proceed.")
            return {
                "strategy_code": code,
                "code_confirmed": False,
                "optimization_mode": optimization_mode,
                "messages": [AIMessage(content="Code generated. Waiting for user confirmation.")],
                "sender": "quant_agent",
                "next_step": "WAIT_FOR_CONFIRMATION",
                # Mask sensitive fields in input_vars before storing
                "llm_interaction": {
                    "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k,v in input_vars.items()},
                    "prompt": formatted_prompt,
                    "response": response.content
                }
            }
