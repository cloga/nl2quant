import tushare as ts
import pandas as pd
import streamlit as st
from datetime import datetime
from app.config import Config
from app.state import AgentState
from app.ui_utils import render_live_timer, display_token_usage
from langchain_core.messages import AIMessage

def data_agent(state: AgentState):
    """
    Agent responsible for fetching market data using Tushare.
    """
    print("--- DATA AGENT ---")
    st.write("🔍 **Data Agent:** Analyzing request to identify ticker...")
    messages = state["messages"]
    last_message = messages[-1]
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm_interaction = None
    
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
        current_date_str = datetime.now().strftime("%Y%m%d")
        
        extract_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial entity and parameter extractor. 
            Your task is to identify the stock ticker and the time range from the user's input.
            
            Current Date: {current_date}
            
            Rules for Ticker:
            1. If the user provides a ticker (e.g., 600519.SH, 000001.SZ, AAPL), use it directly.
            2. If the user provides a company name (e.g., 'Moutai', 'Ningde Times', 'Ping An'), convert it to the most likely ticker symbol.
               - Prefer Chinese A-shares (Shanghai .SH, Shenzhen .SZ) if the name is Chinese.
               - Examples: 
                 - 'Moutai' -> '600519.SH'
                 - 'Ningde Times' / 'CATL' -> '300750.SZ'
                 - 'Ping An' -> '601318.SH'
            3. If absolutely no ticker or company name is found, return 'NONE' for the ticker.
            
            Rules for Benchmark:
            - If the ticker is a Chinese A-share (ends in .SH or .SZ), also identify the benchmark '000300.SH' (CSI 300).
            - If the ticker is US stock, identify 'SPY' or '^GSPC'.
            
            Rules for Date Range:
            1. Extract the start and end dates based on the user's description (e.g., "last year", "recent half year", "2023-01-01 to 2023-05-01").
            2. Use the Current Date ({current_date}) as the reference for relative dates.
            3. Format dates as YYYYMMDD.
            4. If no date is specified, default to:
               - Start Date: 20230101
               - End Date: 20231231
            
            Return Format:
            TICKER,BENCHMARK,START_DATE,END_DATE
            
            Examples:
            - Input: "Backtest Moutai for last year" (Assuming current date is 20240501)
              Output: 600519.SH,000300.SH,20230101,20231231
            - Input: "Ningde Times recent half year" (Assuming current date is 20240601)
              Output: 300750.SZ,000300.SH,20231201,20240601
            """),
            ("user", "{input}")
        ])
        chain = extract_prompt | llm
        input_vars = {"input": last_message.content, "current_date": current_date_str}
        
        # Capture formatted messages
        formatted_messages = extract_prompt.format_messages(**input_vars)
        formatted_prompt = "\n\n".join([f"**{m.type.upper()}**: {m.content}" for m in formatted_messages])
        
        with st.expander("🗂️ Data Agent", expanded=True):
            timer = render_live_timer("⏳ Extracting ticker...")
            raw_response = chain.invoke(input_vars)
            timer.empty()
            
            display_token_usage(raw_response)
            
            with st.expander("🧠 View Raw Prompt & Response", expanded=False):
                st.markdown("**📝 Prompt:**")
                st.code(formatted_prompt, language="markdown")
                st.markdown("**💬 Response:**")
                st.code(raw_response.content, language="markdown")

            result = raw_response.content.strip()
            
            # Store interaction for debugging
            llm_interaction = {
                "input": input_vars,
                "prompt": formatted_prompt,
                "response": raw_response.content
            }
            
            if result and result != "NONE":
                parts = result.split(",")
                if len(parts) >= 4:
                    tickers = [parts[0].strip()]
                    benchmark = parts[1].strip()
                    start_date = parts[2].strip()
                    end_date = parts[3].strip()
                    st.success(f"**Extracted:** `{tickers[0]}` | **Benchmark:** `{benchmark}` | **Range:** `{start_date}` - `{end_date}`")
                elif len(parts) >= 2:
                    tickers = [parts[0].strip()]
                    benchmark = parts[1].strip()
                    st.success(f"**Extracted Ticker:** `{tickers[0]}` | **Benchmark:** `{benchmark}`")
                else:
                    tickers = [result]
                    benchmark = None
                    st.success(f"**Extracted Ticker:** `{result}`")
            else:
                st.warning("Could not identify ticker.")
                return {
                    "messages": [AIMessage(content="I couldn't identify a stock ticker. Please specify one (e.g., 'Backtest on 600519.SH').")],
                    "sender": "data_agent",
                    "llm_interaction": llm_interaction
                }

            # Initialize Tushare
            if not Config.TUSHARE_TOKEN:
                st.error("Tushare token missing.")
                return {
                    "messages": [AIMessage(content="Error: Tushare token is missing in configuration.")],
                    "sender": "data_agent"
                }
            
            st.write("⏳ Fetching data from Tushare...")
            ts.set_token(Config.TUSHARE_TOKEN)
            pro = ts.pro_api()
            
            data_map = {}
            
            try:
                for ticker in tickers:
                    print(f"Fetching data for {ticker}...")
                    st.write(f"⬇️ **Data Agent:** Calling Tushare `daily` API for `{ticker}` ({start_date} to {end_date})...")
                    # Fetch daily data
                    df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
                    
                    if df.empty:
                        print(f"No data found for {ticker}")
                        st.warning(f"No data found for {ticker}")
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
                    
                    # Show preview
                    with st.expander(f"📊 Data Preview: {ticker}", expanded=False):
                        st.dataframe(df.head())
                
                # Fetch benchmark data if available
                benchmark_data = None
                if benchmark and benchmark != "NONE":
                    try:
                        st.write(f"⬇️ **Data Agent:** Fetching benchmark `{benchmark}` for comparison...")
                        if benchmark.endswith('.SH') or benchmark.endswith('.SZ'):
                            # Chinese index - use index_daily API
                            bench_df = pro.index_daily(ts_code=benchmark, start_date=start_date, end_date=end_date)
                        else:
                            # Regular stock
                            bench_df = pro.daily(ts_code=benchmark, start_date=start_date, end_date=end_date)
                        
                        if not bench_df.empty:
                            bench_df['trade_date'] = pd.to_datetime(bench_df['trade_date'])
                            bench_df = bench_df.sort_values('trade_date')
                            bench_df = bench_df.set_index('trade_date')
                            bench_df = bench_df[['close']]
                            bench_df.columns = ['Close']
                            benchmark_data = {benchmark: bench_df}
                            st.success(f"Successfully fetched benchmark data: {benchmark}")
                    except Exception as e:
                        st.warning(f"Could not fetch benchmark data: {e}")
                
                if not data_map:
                     st.error("Failed to fetch data.")
                     return {
                         "messages": [AIMessage(content=f"Failed to fetch data for {tickers}. Please check the ticker symbol.")],
                         "sender": "data_agent"
                     }

                st.success(f"Successfully fetched data for {', '.join(tickers)}.")
                
                # Store data in state
                return {
                    "market_data": data_map,
                    "benchmark_data": benchmark_data,
                    "benchmark_ticker": benchmark if benchmark_data else None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "messages": [AIMessage(content=f"Successfully fetched data for {', '.join(tickers)} from {start_date} to {end_date}.")],
                    "sender": "data_agent",
                    "llm_interaction": llm_interaction
                }

            except Exception as e:
                st.error(f"Error fetching data: {str(e)}")
                return {
                    "messages": [AIMessage(content=f"Error fetching data: {str(e)}")],
                    "sender": "data_agent"
                }
