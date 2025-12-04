import streamlit as st
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from app.config import Config

# Load environment variables
load_dotenv()

st.set_page_config(page_title="NL-to-Quant Platform", layout="wide")

st.title("🤖 Natural Language to Quant Platform")
st.markdown("""
Welcome! I can help you with financial analysis and backtesting using natural language.
Supported by **Tushare** data and **VectorBT**.
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    tushare_token = st.text_input("Tushare Token", type="password", value=os.getenv("TUSHARE_TOKEN", ""))

    supported = Config.SUPPORTED_LLM_PROVIDERS
    current_provider = os.getenv("LLM_PROVIDER", Config.LLM_PROVIDER)
    try:
        default_index = supported.index(current_provider)
    except ValueError:
        default_index = 0

    provider = st.selectbox("LLM Provider", options=supported, index=default_index)
    prefix = f"LLM_{provider.upper()}"

    api_key = st.text_input(
        "LLM API Key",
        type="password",
        value=os.getenv(f"{prefix}_API_KEY", ""),
    )
    model_name = st.text_input(
        "LLM Model",
        value=os.getenv(f"{prefix}_MODEL_NAME", Config.PROVIDER_DEFAULT_MODELS.get(provider, "")),
    )
    base_url = st.text_input(
        "LLM Base URL",
        value=os.getenv(f"{prefix}_BASE_URL", Config.PROVIDER_DEFAULT_BASE_URL.get(provider, "")),
    )

    if tushare_token:
        os.environ["TUSHARE_TOKEN"] = tushare_token
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if api_key:
        os.environ[f"{prefix}_API_KEY"] = api_key
    if model_name:
        os.environ[f"{prefix}_MODEL_NAME"] = model_name
    if base_url:
        os.environ[f"{prefix}_BASE_URL"] = base_url

    st.divider()
    st.info(f"Provider: {provider} | Model: {model_name}")

# Chat Interface Placeholder
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Try: 'Backtest a generic crossover strategy on 600519.SH'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Processing...", expanded=True) as status:
            st.write("Initializing Agent Graph...")
            from app.graph import create_graph
            app = create_graph()
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                # "tickers": [], # Let the agent extract it
                "start_date": "20230101",
                "end_date": "20231231",
                "llm_provider": provider.lower() if provider else None,
                "llm_model": model_name.strip() if model_name else None,
            }
            
            # Run the graph
            st.write("Running Workflow...")
            final_state = None
            for output in app.stream(initial_state):
                for key, value in output.items():
                    st.write(f"Finished: {key}")
                    if "messages" in value and value["messages"]:
                        # Show intermediate messages if any
                        pass
                    final_state = value # Keep updating to get the last state
            
            status.update(label="Complete!", state="complete", expanded=False)

        # Display Final Result
        if final_state:
            # Show Analyst Summary
            # The final state might be nested depending on how stream returns. 
            # app.stream yields dictionaries keyed by node name.
            # We need to capture the accumulated state.
            # Actually, let's just run invoke for simplicity in the UI if stream is complex
            pass

        # Re-run invoke to get full final state easily (or manage state better above)
        # For MVP, let's just use the final output from the loop which is the partial state update
        # We need the FULL state.
        
        final_state = app.invoke(initial_state)
        
        # Display Analyst Response
        last_msg = final_state["messages"][-1]
        st.markdown(last_msg.content)
        
        # Display Plot
        if "figure_json" in final_state and final_state["figure_json"]:
            st.plotly_chart(final_state["figure_json"])
            
        # Append to session state
        st.session_state.messages.append({"role": "assistant", "content": last_msg.content})

