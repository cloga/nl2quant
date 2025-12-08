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
            Avoid defaulting to MA crossover; choose parameters meaningful to the chosen template.
            
            Context:
            - The data is already loaded into a dictionary named `data_map`.
            - `data_map` keys are tickers: {tickers}
            - Each value in `data_map` is a Pandas DataFrame with columns: Open, High, Low, Close, Volume.
            - The index is a DatetimeIndex.
            
            Strategy menu (pick one that matches the request, otherwise vary):
            - RSI thresholds sweep (windows and bands).
            - Bollinger band width/alpha sweep.
            - Donchian breakout window sweep.
            - MACD fast/slow/signal sweep.
            - MA crossover only if explicitly asked.
            
            Requirements for Parameter Optimization:
            1. Use `vectorbt` (imported as `vbt`) for the backtest.
            2. Define parameter ranges using numpy arrays (e.g., windows = np.arange(10, 60, 5)).
            3. Use vectorbt's broadcasting to test all combinations.
            4. The code MUST define:
               - `portfolio`: result of `vbt.Portfolio.from_signals(...)` with parameter grids.
               - `param_names`: list of parameter names.
               - `param_values`: dict mapping param names to tested arrays.
            5. ALWAYS pass `freq='1D'`.
            6. Return ONLY the Python code. No markdown formatting, no ```python blocks.
            
            Example idea (MACD sweep, adjust to user need):
            price = data_map['600519.SH']['Close']
            fast = np.arange(8, 18, 2)
            slow = np.arange(20, 40, 4)
            signal = np.arange(6, 12, 2)
            macd = vbt.MACD.run(price, fast_window=fast, slow_window=slow, signal_window=signal)
            entries = macd.macd_above_signal()
            exits = macd.macd_below_signal()
            portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
            param_names = ['fast_window', 'slow_window', 'signal_window']
            param_values = {{'fast_window': fast, 'slow_window': slow, 'signal_window': signal}}
            """),
            ("user", "{input}{feedback}")
        ])
    else:
                prompt = ChatPromptTemplate.from_messages([
                        ("system", """You are an expert Python Quant Developer using the `vectorbt` library.
                        Your goal is to generate executable Python code that backtests a trading strategy described by the user.
                        Avoid always using moving-average crossover unless the user explicitly asks for it; select a strategy that matches intent or pick a diversified template.
            
                        Context:
                        - The data is already loaded into a dictionary named `data_map`.
                        - `data_map` keys are tickers: {tickers}
                        - Each value in `data_map` is a Pandas DataFrame with columns: Open, High, Low, Close, Volume.
                        - The index is a DatetimeIndex.
            
                        Strategy menu (choose the one best aligned to the request, otherwise vary from this list):
                        - RSI mean reversion: use `vbt.RSI.run`, enter when RSI < 30, exit when RSI > 70 (or user thresholds).
                        - Bollinger band fade: `vbt.BBANDS.run`, enter when Close < lower, exit when Close > upper.
                        - Donchian breakout / high-low channel: `vbt.DonchianChannel.run`, breakout entries/exits.
                        - MACD trend: `vbt.MACD.run`, signal/line cross.
                        - VWAP/volume filter: use rolling VWAP or volume spike filter.
                        - MA crossover only if user mentions it.
                        - Portfolio/hedge: if multiple tickers and user hints combination, allow simple equal-weight order sizing via `from_signals` per leg.
            
                        Requirements:
                        1. Use `vectorbt` (imported as `vbt`) for the backtest.
                        2. The code MUST define a variable named `portfolio` which is the result of `vbt.Portfolio.from_signals(...)` or similar.
                        3. Do NOT fetch data. Use `data_map['TICKER']` to access data.
                        4. If multiple tickers are present, handle them appropriately (or pick the first one for single-asset strategies when unspecified).
                        5. ALWAYS pass `freq='1D'` to `vbt.Portfolio.from_signals` or `from_orders`.
                        6. Return ONLY the Python code. No markdown formatting, no ```python blocks.
                        7. Ensure the code is safe and does not use system calls.
            
                        Example snippets (pick one pattern, do not output all):
                        - RSI mean reversion:
                            price = data_map['T']["Close"]
                            rsi = vbt.RSI.run(price, window=14)
                            entries = rsi.rsi < 30
                            exits = rsi.rsi > 70
                            portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
                        - Bollinger band fade:
                            bb = vbt.BBANDS.run(price, window=20, alpha=2)
                            entries = price < bb.lower
                            exits = price > bb.upper
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
        
        # --- Human-in-the-Loop Code Editing (auto-confirm) ---
        st.markdown("#### 💻 Generated Code (Editable)")
        st.info("💡 代码已自动确认，将直接进入执行。如需调整，可在下方编辑后自动生效。")
        
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
        
        # Optional: allow quick reset, but auto-confirm regardless
        if st.button("🔄 Reset to Original", key=f"reset_{hash(code)}"):
            st.session_state[code_key] = code
            st.rerun()

        # Auto-confirm branch
        final_code = edited_code.strip()
        st.session_state[code_key] = final_code
        st.success("Code auto-confirmed. Proceeding to execution...")
        
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
            "messages": [AIMessage(content="Generated strategy code (auto-confirmed).")],
            "sender": "quant_agent",
            "llm_interaction": {
                "input": safe_input,
                "prompt": formatted_prompt,
                "response": response.content
            }
        }
