import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.llm import get_llm
from app.state import AgentState
from app.ui_utils import render_live_timer, display_token_usage


def macro_agent(state: AgentState):
    """Agent for macroeconomic narrative analysis (LLM-based, no external data fetch)."""
    print("--- MACRO AGENT ---")
    st.write("🌐 **Macro Agent:** Running macro narrative analysis...")

    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm = get_llm(provider=provider, model=model)

    messages = state.get("messages", [])
    user_input = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None) if not isinstance(msg, dict) else msg.get("content")
        mtype = getattr(msg, "type", None) if not isinstance(msg, dict) else msg.get("type")
        if mtype == "human" or "HumanMessage" in str(type(msg)):
            user_input = content or ""
            break
    if not user_input and messages:
        last = messages[-1]
        user_input = last.get("content") if isinstance(last, dict) else getattr(last, "content", "")

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a macro strategy analyst.
Provide concise Chinese macro analysis for the user's request. Use a clear structure:
- 现状: summarize growth/inflation/liquidity/policy tone in generic terms.
- 逻辑: key drivers and potential paths.
- 风险: upside/downside risks (rates, credit, FX, commodities, geopolitics).
- 观察: 3-5 watch items (data prints, policy meetings, curves/spreads, credit/FX, commodities).
- 行动框架: suggest positioning tilt (pro-risk/defensive), duration bias, equity style tilt (value/growth, large/small), and hedge ideas.
Do NOT claim to have live data; keep statements generic unless user provided specifics.
"""),
        ("user", "{query}")
    ])

    chain = prompt | llm
    with st.expander("🌐 Macro Agent", expanded=True):
        timer = render_live_timer("⏳ Analyzing macro context...")
        response = chain.invoke({"query": user_input})
        timer.empty()
        display_token_usage(response)
        with st.expander("🧠 View Raw Prompt & Response", expanded=False):
            formatted = "\n\n".join([f"**{m.type.upper()}**: {m.content}" for m in prompt.format_messages(query=user_input)])
            st.markdown("**📝 Prompt:**")
            st.code(formatted, language="markdown")
            st.markdown("**💬 Response:**")
            st.code(response.content, language="markdown")

    return {
        "messages": [AIMessage(content=response.content)],
        "sender": "macro_agent",
        "llm_interaction": {
            "input": {"query": user_input},
            "prompt": "macro structured analysis",
            "response": response.content
        }
    }
