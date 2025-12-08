import tushare as ts
import akshare as ak
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
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
    # Prefer the latest human instruction; fallback to the last message if none
    last_message = messages[-1]
    for msg in reversed(messages):
        try:
            if msg.type == "human":
                last_message = msg
                break
        except AttributeError:
            # If message has no type, ignore and continue
            continue
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm_interaction = None
    
    # In a real scenario, we would use an LLM to extract these parameters.
    # Let's use the LLM to extract the ticker if not present.
    
    tickers = state.get("tickers", [])

    # Planner-provided intent flags
    need_full_history = bool(state.get("need_full_history"))
    needs_benchmark = bool(state.get("needs_benchmark"))

    # Default to trailing 12 months if user did not specify dates
    current_date = datetime.now()
    default_start_date = (current_date - timedelta(days=365)).strftime("%Y%m%d")
    default_end_date = current_date.strftime("%Y%m%d")

    start_date = state.get("start_date", default_start_date)
    end_date = state.get("end_date", default_end_date)

    # Clamp date range to today and ensure start <= end; reset to 12m if invalid
    today_str = default_end_date
    if end_date > today_str:
        end_date = today_str
    try:
        if start_date > end_date:
            start_date = default_start_date
            end_date = today_str
    except Exception:
        start_date = default_start_date
        end_date = today_str
    
    if not tickers:
        print("Extracting ticker from user message...")
        from langchain_core.prompts import ChatPromptTemplate
        
        # Use configured provider for extraction (defaults from Config)
        ds_settings = Config.get_llm_settings(provider)
        llm = ChatOpenAI(
            model=ds_settings.get("model"),
            api_key=ds_settings.get("api_key"),
            base_url=ds_settings.get("base_url"),
            temperature=0,
        )
        current_date_str = default_end_date

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
            3. End Date MUST NOT exceed the Current Date; if user asks for future, cap at Current Date.
            4. Format dates as YYYYMMDD.
            5. If no date is specified, default to the last 12 months:
                - Start Date: {default_start_date}
                - End Date: {default_end_date}

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
        input_vars = {
            "input": last_message.content,
            "current_date": current_date_str,
            "default_start_date": default_start_date,
            "default_end_date": default_end_date,
        }
        
        # Capture formatted messages
        formatted_messages = extract_prompt.format_messages(**input_vars)
        formatted_prompt = "\n\n".join([f"**{m.type.upper()}**: {m.content}" for m in formatted_messages])
        
        with st.expander("🗂️ Data Agent", expanded=True):
            timer = render_live_timer("⏳ Extracting ticker...")
            try:
                raw_response = chain.invoke(input_vars)
            except Exception as e:
                timer.empty()
                msg = f"提取代码失败：{e}. 请检查是否提供了正确的公司名或股票代码。"
                st.error(msg)
                return {
                    "messages": [AIMessage(content=msg)],
                    "sender": "data_agent"
                }
            timer.empty()
            
            display_token_usage(raw_response)
            
            with st.expander("🧠 View Raw Prompt & Response", expanded=False):
                st.markdown("**📝 Prompt:**")
                st.code(formatted_prompt, language="markdown")
                st.markdown("**💬 Response:**")
                st.code(raw_response.content, language="markdown")

            result = raw_response.content.strip()
            
            # Store interaction for debugging
            # Mask any sensitive fields in input_vars before storing
            safe_input_vars = dict(input_vars)
            for kk in list(safe_input_vars.keys()):
                lk = kk.lower()
                if 'token' in lk or 'api_key' in lk or 'secret' in lk or 'password' in lk:
                    safe_input_vars[kk] = '***MASKED***'

            llm_interaction = {
                "input": safe_input_vars,
                "prompt": formatted_prompt,
                "response": raw_response.content
            }
            
            def normalize_ts_code(code: str) -> str:
                """Add exchange suffix for 6-digit A-share tickers when missing."""
                if not code:
                    return code
                code = code.strip()
                if "." in code:
                    return code.upper()
                if code.isdigit() and len(code) == 6:
                    if code[0] in {"0", "3"}:
                        return f"{code}.SZ"
                    if code[0] in {"6", "5", "9"}:
                        return f"{code}.SH"
                return code.upper()

            if result and result != "NONE":
                parts = result.split(",")
                if len(parts) >= 4:
                    tickers = [normalize_ts_code(parts[0].strip())]
                    benchmark = normalize_ts_code(parts[1].strip()) if parts[1].strip() else None
                    start_date = parts[2].strip()
                    end_date = parts[3].strip()
                    st.success(f"**Extracted:** `{tickers[0]}` | **Benchmark:** `{benchmark}` | **Range:** `{start_date}` - `{end_date}`")
                elif len(parts) >= 2:
                    tickers = [normalize_ts_code(parts[0].strip())]
                    benchmark = normalize_ts_code(parts[1].strip()) if parts[1].strip() else None
                    st.success(f"**Extracted Ticker:** `{tickers[0]}` | **Benchmark:** `{benchmark}`")
                else:
                    tickers = [normalize_ts_code(result)]
                    benchmark = None
                    st.success(f"**Extracted Ticker:** `{result}`")
            else:
                st.warning("未能识别到股票代码，请检查公司名或股票代码是否正确。")
                return {
                    "messages": [AIMessage(content="未能识别到股票代码，请检查公司名或股票代码是否正确（例如 600519.SH）。")],
                    "sender": "data_agent",
                    "llm_interaction": llm_interaction
                }

            # Re-clamp extracted dates to today and valid order
            today_str = default_end_date
            if end_date > today_str:
                end_date = today_str
            try:
                if start_date > end_date:
                    start_date = default_start_date
                    end_date = today_str
            except Exception:
                start_date = default_start_date
                end_date = today_str

            # Expand date range if planner/user asked for all/full history
            if need_full_history:
                start_date = "19900101"  # earliest practical default
                end_date = today_str

            data_map = {}

            # -------------------------
            # Helper: AkShare fetch
            # -------------------------
            def fetch_with_akshare(ticker: str, sd: str, ed: str):
                """Fetch daily data via AkShare. Supports A-share stocks/ETFs and common indices."""
                # Normalize dates to YYYYMMDD strings (AkShare expects plain digits)
                start = sd.replace("-", "")
                end = ed.replace("-", "")

                def norm_stock(ts_code: str):
                    if "." in ts_code:
                        return ts_code.split(".")[0]
                    return ts_code

                def is_index(ts_code: str) -> bool:
                    return (
                        (ts_code.endswith('.SH') and ts_code.startswith('000')) or
                        (ts_code.endswith('.SZ') and ts_code.startswith('399'))
                    )

                try:
                    if is_index(ticker):
                        symbol = ticker.lower().replace(".sh", "sh").replace(".sz", "sz")
                        df_idx = ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")
                        if df_idx is None or df_idx.empty:
                            return pd.DataFrame()
                        df_idx = df_idx.rename(columns={"日期": "date", "收盘": "close"})
                        df_idx["date"] = pd.to_datetime(df_idx["date"])
                        df_idx = df_idx.sort_values("date").set_index("date")
                        df_idx = df_idx[["close"]]
                        df_idx.columns = ["Close"]
                        return df_idx

                    symbol = norm_stock(ticker)
                    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="qfq")
                    if df is None or df.empty:
                        return pd.DataFrame()
                    rename_map = {
                        "日期": "date",
                        "开盘": "Open",
                        "最高": "High",
                        "最低": "Low",
                        "收盘": "Close",
                        "成交量": "Volume",
                    }
                    df = df.rename(columns=rename_map)
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.sort_values("date").set_index("date")
                    df = df[["Open", "High", "Low", "Close", "Volume"]]
                    return df
                except Exception:
                    return pd.DataFrame()

            def fetch_with_chunks(api_fn, code: str, sd: str, ed: str):
                """Fetch data in yearly chunks to cover long history and avoid API limits."""
                frames = []
                sd_dt = datetime.strptime(sd, "%Y%m%d")
                ed_dt = datetime.strptime(ed, "%Y%m%d")
                cursor = sd_dt
                while cursor <= ed_dt:
                    chunk_end = min(cursor + timedelta(days=365), ed_dt)
                    chunk_sd = cursor.strftime("%Y%m%d")
                    chunk_ed = chunk_end.strftime("%Y%m%d")
                    df_chunk = api_fn(ts_code=code, start_date=chunk_sd, end_date=chunk_ed)
                    if not df_chunk.empty:
                        frames.append(df_chunk)
                    cursor = chunk_end + timedelta(days=1)
                if not frames:
                    return pd.DataFrame()
                return pd.concat(frames, ignore_index=True)
            
            try:
                # Initialize provider(s)
                use_provider = Config.DATA_PROVIDER if Config.DATA_PROVIDER in Config.SUPPORTED_DATA_PROVIDERS else "tushare"
                pro = None
                if use_provider == "tushare":
                    if not Config.TUSHARE_TOKEN:
                        st.error("Tushare token missing. 请在配置中设置 TUSHARE_TOKEN 或切换 DATA_PROVIDER=akshare。")
                        return {
                            "messages": [AIMessage(content="Error: Tushare token is missing in configuration. 请配置 TUSHARE_TOKEN 或使用 akshare 数据源。")],
                            "sender": "data_agent"
                        }
                    st.write("⏳ Fetching data from Tushare...")
                    ts.set_token(Config.TUSHARE_TOKEN)
                    pro = ts.pro_api()
                elif use_provider == "akshare":
                    st.write("⏳ Fetching data via AkShare...")

                for ticker in tickers:
                    print(f"Fetching data for {ticker} using {use_provider}...")

                    df = pd.DataFrame()

                    if use_provider == "akshare":
                        df = fetch_with_akshare(ticker, start_date, end_date)
                        if df.empty:
                            st.warning(f"AkShare 未取到数据，跳过 {ticker}。")
                            continue
                        st.success(f"AkShare 获取成功: {ticker}")

                    if use_provider == "tushare":
                        # Detect index vs stock
                        is_index = (
                            (ticker.endswith('.SH') and ticker.startswith('000')) or
                            (ticker.endswith('.SZ') and ticker.startswith('399'))
                        )

                        if is_index:
                            st.write(f"⬇️ **Data Agent:** Calling Tushare `index_daily` API for `{ticker}` ({start_date} to {end_date})...")
                            df = fetch_with_chunks(pro.index_daily, ticker, start_date, end_date) if need_full_history else pro.index_daily(ts_code=ticker, start_date=start_date, end_date=end_date)
                        else:
                            st.write(f"⬇️ **Data Agent:** Calling Tushare `daily` API for `{ticker}` ({start_date} to {end_date})...")
                            df = fetch_with_chunks(pro.daily, ticker, start_date, end_date) if need_full_history else pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)

                            # Fallback for ETFs/funds (A-share tickers often start with 5/1)
                            if df.empty and ticker[:1] in {"5", "1"}:
                                st.write(f"🔄 **Data Agent:** `daily` returned empty, retrying `fund_daily` for `{ticker}`...")
                                df = fetch_with_chunks(pro.fund_daily, ticker, start_date, end_date) if need_full_history else pro.fund_daily(ts_code=ticker, start_date=start_date, end_date=end_date)

                    if df.empty:
                        print(f"No data found for {ticker}")
                        st.warning(f"No data found for {ticker}")
                        continue
                        
                    # Standardize columns
                    if use_provider == "tushare":
                        date_col = 'trade_date' if 'trade_date' in df.columns else ('nav_date' if 'nav_date' in df.columns else None)
                        if date_col is None:
                            st.warning(f"Unexpected date column for {ticker}, skipping.")
                            continue
                        df[date_col] = pd.to_datetime(df[date_col])
                        df = df.sort_values(date_col)
                        df = df.set_index(date_col)
                        df = df[['open', 'high', 'low', 'close', 'vol']]
                        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

                    data_map[ticker] = df
                    
                    # Show preview
                    with st.expander(f"📊 Data Preview: {ticker}", expanded=False):
                        st.dataframe(df.head())
                
                # Conditionally fetch benchmark if planner signals need
                benchmark_data = None
                benchmark = state.get("benchmark_ticker")
                if needs_benchmark and tickers:
                    primary = tickers[0]
                    if not benchmark:
                        if primary.endswith('.SH') or primary.endswith('.SZ'):
                            benchmark = "000300.SH"
                        elif primary.isalpha():
                            benchmark = "SPY"
                    # Avoid self-benchmarking
                    if benchmark and benchmark != primary:
                        try:
                            st.write(f"⬇️ **Data Agent:** Fetching benchmark `{benchmark}` for comparison (planner signaled benchmark需求)...")
                            if use_provider == "akshare":
                                bench_df = fetch_with_akshare(benchmark, start_date, end_date)
                            else:
                                bench_df = pro.index_daily(ts_code=benchmark, start_date=start_date, end_date=end_date) if benchmark.endswith(('.SH', '.SZ')) else pro.daily(ts_code=benchmark, start_date=start_date, end_date=end_date)
                                if not bench_df.empty:
                                    bench_df['trade_date'] = pd.to_datetime(bench_df['trade_date'])
                                    bench_df = bench_df.sort_values('trade_date')
                                    bench_df = bench_df.set_index('trade_date')
                                    bench_df = bench_df[['close']]
                                    bench_df.columns = ['Close']
                            if bench_df is not None and not bench_df.empty:
                                benchmark_data = {benchmark: bench_df}
                                st.success(f"Successfully fetched benchmark data: {benchmark}")
                        except Exception as e:
                            st.warning(f"Could not fetch benchmark data: {e}")
                
                if not data_map:
                    st.error("Failed to fetch data.")
                    return {
                        "messages": [AIMessage(content="未能获取行情，请检查代码/公司名是否正确，日期范围是否有效，或该标的是否需使用指数代码(如 000300.SH)。如需重试，请更换代码后再次提交。")],
                        "sender": "data_agent",
                        "data_failed": True
                    }

                st.success(f"Successfully fetched data for {', '.join(tickers)}.")
                
                # Store data in state
                return {
                    "market_data": data_map,
                    "benchmark_data": benchmark_data,
                    "benchmark_ticker": benchmark if benchmark_data else None,
                    "start_date": start_date,
                    "end_date": end_date,
                    "tickers": tickers,
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
