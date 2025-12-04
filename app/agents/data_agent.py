import tushare as ts
import pandas as pd
from app.config import Config
from app.state import AgentState
from langchain_core.messages import AIMessage

def data_agent(state: AgentState):
    """
    Agent responsible for fetching market data using Tushare.
    """
    print("--- DATA AGENT ---")
    messages = state["messages"]
    last_message = messages[-1]
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    
    # In a real scenario, we would use an LLM to extract these parameters.
    # Let's use the LLM to extract the ticker if not present.
    
    tickers = state.get("tickers", [])
    start_date = state.get("start_date", "20230101")
    end_date = state.get("end_date", "20231231")
    
    if not tickers:
        print("Extracting ticker from user message...")
        from app.llm import get_llm
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = get_llm(provider=provider, model=model)
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a financial entity extractor. Extract the stock ticker (e.g., 600519.SH, AAPL) from the user input. Return ONLY the ticker symbol. If none found, return 'NONE'."),
            ("user", "{input}")
        ])
        chain = extract_prompt | llm
        result = chain.invoke({"input": last_message.content}).content.strip()
        
        if result and result != "NONE":
            tickers = [result]
            # Update state with extracted ticker for future reference
            # Note: We can't update state directly here easily without returning it, 
            # but we will return it in the dict below.
        else:
            # Fallback for MVP testing if extraction fails or no ticker mentioned
            # tickers = ["600519.SH"] 
            return {
                "messages": [AIMessage(content="I couldn't identify a stock ticker. Please specify one (e.g., 'Backtest on 600519.SH').")]
            }

    # Initialize Tushare
    if not Config.TUSHARE_TOKEN:
        return {"messages": [AIMessage(content="Error: Tushare token is missing in configuration.")]}
    
    ts.set_token(Config.TUSHARE_TOKEN)
    pro = ts.pro_api()
    
    data_map = {}
    
    try:
        for ticker in tickers:
            print(f"Fetching data for {ticker}...")
            # Fetch daily data
            df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
            
            if df.empty:
                print(f"No data found for {ticker}")
                continue
                
            # Tushare returns data in descending order usually, sort by date ascending
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('trade_date')
            df = df.set_index('trade_date')
            
            # Keep relevant columns for VectorBT (Open, High, Low, Close, Volume)
            # Tushare columns: open, high, low, close, vol
            df = df[['open', 'high', 'low', 'close', 'vol']]
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume'] # Standardize for vbt
            
            data_map[ticker] = df
            
        if not data_map:
             return {"messages": [AIMessage(content=f"Failed to fetch data for {tickers}. Please check the ticker symbol.")]}

        # Store data in state (In production, might want to store path to CSV/Parquet to avoid memory issues)
        # For MVP, we store the dict of DataFrames directly (or a serialized version if needed)
        
        return {
            "market_data": data_map,
            "messages": [AIMessage(content=f"Successfully fetched data for {', '.join(tickers)} from {start_date} to {end_date}.")]
        }

    except Exception as e:
        return {"messages": [AIMessage(content=f"Error fetching data: {str(e)}")]}
