from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.llm import get_llm
from app.state import AgentState

def quant_agent(state: AgentState):
    """
    Agent responsible for generating VectorBT strategy code.
    """
    print("--- QUANT AGENT ---")
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
        5. Return ONLY the Python code. No markdown formatting, no ```python blocks.
        6. Ensure the code is safe and does not use system calls.
        
        Example Snippet:
        price = data_map['600519.SH']['Close']
        fast_ma = vbt.MA.run(price, 10)
        slow_ma = vbt.MA.run(price, 50)
        entries = fast_ma.ma_crossed_above(slow_ma)
        exits = fast_ma.ma_crossed_below(slow_ma)
        portfolio = vbt.Portfolio.from_signals(price, entries, exits)
        """),
        ("user", "{input}")
    ])
    
    chain = prompt | llm
    
    response = chain.invoke({
        "input": user_request,
        "tickers": data_keys
    })
    
    code = response.content.strip()
    
    # Clean up markdown if the LLM ignored instructions
    if code.startswith("```python"):
        code = code.split("```python")[1]
    if code.startswith("```"):
        code = code.split("```")[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
        
    return {
        "strategy_code": code,
        "messages": [AIMessage(content="Generated strategy code.")]
    }
