from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from app.llm import get_llm
from app.state import AgentState

def analyst_agent(state: AgentState):
    """
    Agent responsible for summarizing the backtest results and generating visualizations.
    """
    print("--- ANALYST AGENT ---")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm = get_llm(provider=provider, model=model)
    
    metrics = state.get("performance_metrics", {})
    logs = state.get("execution_output", "")
    market_data = state.get("market_data", {})
    
    # --- Visualization Logic ---
    analyst_figures = []
    analyst_data = {}
    
    if market_data:
        # 1. K-Line Chart (Candlestick) for the first ticker
        # We only show K-Line if we have OHLC data
        first_ticker = list(market_data.keys())[0]
        df = market_data[first_ticker]
        
        if not df.empty and all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            fig_kline = go.Figure(data=[go.Candlestick(x=df.index,
                            open=df['Open'],
                            high=df['High'],
                            low=df['Low'],
                            close=df['Close'],
                            name=first_ticker)])
            fig_kline.update_layout(
                title=f"{first_ticker} K-Line Chart", 
                xaxis_rangeslider_visible=False,
                height=500
            )
            analyst_figures.append(fig_kline)
            
            # Add data preview
            analyst_data[f"{first_ticker} Recent Data"] = df.tail(10)

        # 2. Correlation Heatmap (if multiple tickers)
        if len(market_data) > 1:
            # Combine Close prices
            close_prices = {}
            for ticker, data in market_data.items():
                if 'Close' in data.columns:
                    close_prices[ticker] = data['Close']
            
            if len(close_prices) > 1:
                combined_df = pd.DataFrame(close_prices)
                corr = combined_df.corr()
                
                fig_corr = px.imshow(corr, text_auto=True, title="Stock Correlation Matrix")
                analyst_figures.append(fig_corr)
                analyst_data["Correlation Matrix"] = corr

    # --- LLM Summary Logic ---
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Financial Analyst. 
        
        Scenario 1: Backtest Results Available
        If 'Metrics' are provided, interpret the backtest results and provide a concise summary.
        Highlight the key metrics (Sharpe, Return, Drawdown).
        If there were execution errors, explain them.
        
        Scenario 2: Only Market Data Available
        If 'Metrics' are empty but 'Market Data' is available (check context), summarize the market data.
        - Mention the date range.
        - Mention the start and end price.
        - Mention the general trend (Up/Down).
        
        IMPORTANT: You MUST output your summary in Chinese (Simplified Chinese).
        
        Metrics: {metrics}
        Logs: {logs}
        Market Data Info: {data_info}
        """),
        ("user", "Please summarize the performance or data.")
    ])
    
    # Create a summary of data if available
    data_info = "No data available"
    if market_data:
        info = []
        for ticker, df in market_data.items():
            if not df.empty:
                start = df.index[0].strftime("%Y-%m-%d")
                end = df.index[-1].strftime("%Y-%m-%d")
                if 'Close' in df.columns:
                    start_price = df['Close'].iloc[0]
                    end_price = df['Close'].iloc[-1]
                    info.append(f"Ticker: {ticker}, Range: {start} to {end}, Start Price: {start_price}, End Price: {end_price}")
        data_info = "; ".join(info)

    chain = prompt | llm
    
    input_vars = {
        "metrics": str(metrics),
        "logs": logs,
        "data_info": data_info
    }
    
    response = chain.invoke(input_vars)
    
    return {
        "messages": [response],
        "sender": "analyst_agent",
        "analyst_figures": analyst_figures,
        "analyst_data": analyst_data,
        "llm_interaction": {
            "input": input_vars,
            "response": response.content
        }
    }
