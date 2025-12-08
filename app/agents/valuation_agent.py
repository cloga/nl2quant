import streamlit as st
import pandas as pd
from langchain_core.messages import AIMessage
from app.state import AgentState


def _percentile_series(series: pd.Series) -> float:
    if series.empty:
        return None
    latest = series.iloc[-1]
    pct = (series <= latest).mean()
    return round(pct * 100, 2)


def valuation_agent(state: AgentState):
    """Simple valuation/relative-position proxy using price percentiles (no fundamentals)."""
    print("--- VALUATION AGENT ---")
    st.write("💰 **Valuation Agent:** Assessing relative position (price percentile proxy)...")

    data_map = state.get("market_data") or {}
    if not data_map:
        msg = "暂无行情数据，无法估值定位，请先获取数据。"
        st.warning(msg)
        return {
            "messages": [AIMessage(content=msg)],
            "sender": "valuation_agent",
            "valuation": None,
            "data_failed": True
        }

    ticker = list(data_map.keys())[0]
    df = data_map[ticker]
    if "Close" not in df.columns:
        msg = f"数据缺少 Close 列，无法估值定位: {ticker}"
        st.warning(msg)
        return {"messages": [AIMessage(content=msg)], "sender": "valuation_agent", "valuation": None, "data_failed": True}

    price = df["Close"].dropna()
    pct = _percentile_series(price)
    latest = price.iloc[-1] if not price.empty else None

    if pct is None or latest is None:
        msg = "价格序列为空，无法估值定位。"
        st.warning(msg)
        return {"messages": [AIMessage(content=msg)], "sender": "valuation_agent", "valuation": None, "data_failed": True}

    label = "偏低" if pct < 35 else "中性" if pct < 70 else "偏高"

    st.success(f"{ticker} 当前收盘 {latest:.2f}，在样本期价格分位 {pct}%（{label}）。")
    with st.expander("📊 价格分位详情", expanded=False):
        st.line_chart(price)

    summary = {
        "ticker": ticker,
        "latest_price": float(latest),
        "price_percentile": pct,
        "label": label,
        "sample_size": int(len(price)),
    }

    text = f"{ticker} 在样本期价格分位约 {pct}%（{label}）。此为价格近似估值代理，未包含PE/PB等基本面信息。"

    return {
        "messages": [AIMessage(content=text)],
        "sender": "valuation_agent",
        "valuation": summary
    }
