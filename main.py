import streamlit as st
import os
import time
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
# from streamlit_agraph import agraph, Node, Edge, Config
from app.config import Config as AppConfig
from app.history import save_session, load_session, list_sessions, create_new_session

# Load environment variables
load_dotenv()

# Agent Icons Mapping
AGENT_ICONS = {
    "planner_agent": "🧭",  # Compass for planning/navigation
    "data_agent": "🗂️",     # Card index for data
    "quant_agent": "📈",    # Chart for quantitative analysis
    "exec_agent": "⚙️",     # Gear for execution
    "analyst_agent": "🧐"   # Monocle face for deep analysis
}

st.set_page_config(page_title="NL-to-Quant Platform", layout="wide")

# Initialize session ID
if "session_id" not in st.session_state:
    st.session_state.session_id = create_new_session()

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None

def render_trace_ui(trace_data):
    """Renders the execution trace (Graph + Steps) from stored data."""
    if not trace_data:
        return

    # Render Graph - Removed
    # with st.expander("View Agent Graph", expanded=False):
    #     nodes = trace_data.get("nodes", [])
    #     edges = trace_data.get("edges", [])
    #     config = trace_data.get("config", None)
    #     if nodes and edges and config:
    #         agraph(nodes=nodes, edges=edges, config=config)
    
    # Render Steps
    steps = trace_data.get("steps", [])
    for i, step in enumerate(steps):
        key = step["key"]
        value = step["value"]
        full_state = step["full_state"]
        duration = step.get("duration", 0)
        
        icon = AGENT_ICONS.get(key, "🤖")
        with st.expander(f"{icon} Agent Execution: {key} ({duration:.2f}s)", expanded=True):
            # Display Reasoning & Inputs
            st.markdown("#### 🧠 Reasoning & Inputs")
            
            # Planner Reasoning
            if "reasoning" in value and value["reasoning"]:
                st.markdown(f"**Reasoning:** {value['reasoning']}")
            
            # Planner Decision
            if "next_step" in value and value["next_step"]:
                st.markdown(f"**Decision:** Next step is `{value['next_step']}`")

            # Inputs Context (inferred from agent type)
            if key == "data_agent":
                # Show the input used for extraction
                if "messages" in full_state and full_state['messages']:
                    input_msg = full_state['messages'][0].content
                    st.markdown(f"**Input:** User Request: '{input_msg}'")

            elif key == "quant_agent":
                if "feedback" in full_state and full_state["feedback"]:
                    st.warning(f"**Input Feedback:** {full_state['feedback']}")
                if "market_data" in full_state and full_state["market_data"]:
                    tickers = list(full_state["market_data"].keys())
                    st.markdown(f"**Input Data:** Available for {tickers}")
            
            elif key == "exec_agent":
                st.markdown("**Input:** Strategy Code (see below)")
            
            elif key == "analyst_agent":
                if "performance_metrics" in full_state:
                    st.markdown("**Input:** Performance Metrics & Execution Logs")
                
                # Display Analyst Visuals in Trace
                if "analyst_figures" in value and value["analyst_figures"]:
                    st.markdown("#### 📊 Analyst Charts")
                    for j, fig in enumerate(value["analyst_figures"]):
                        st.plotly_chart(fig, use_container_width=True, key=f"trace_analyst_chart_{i}_{j}")
                
                if "analyst_data" in value and value["analyst_data"]:
                    st.markdown("#### 📋 Analyst Data")
                    for title, df in value["analyst_data"].items():
                        st.write(f"**{title}**")
                        st.dataframe(df)

            # Display Messages
            if "messages" in value and value["messages"]:
                st.markdown("#### 💬 Messages")
                msgs = value["messages"]
                if not isinstance(msgs, list):
                    msgs = [msgs]
                for msg in msgs:
                    content = msg.content if hasattr(msg, "content") else str(msg)
                    st.info(content)

            # Display Strategy Code
            if "strategy_code" in value and value["strategy_code"]:
                st.markdown("#### 💻 Strategy Code")
                st.code(value["strategy_code"], language="python")

            # Display Execution Output
            if "execution_output" in value and value["execution_output"]:
                st.markdown("#### ⚙️ Execution Output")
                st.text(value["execution_output"])
            
            # Display Market Data Info
            if "market_data" in value and value["market_data"] is not None:
                st.markdown("#### 📈 Market Data")
                st.success("Market data fetched successfully.")
            
            # Display LLM Interaction
            if "llm_interaction" in value and value["llm_interaction"]:
                st.markdown("#### 🤖 LLM Interaction")
                with st.expander("Show Prompt & Response", expanded=False):
                    st.markdown("**Input Variables:**")
                    st.json(value["llm_interaction"]["input"])
                    st.markdown("**Raw Response:**")
                    st.text(value["llm_interaction"]["response"])

            # Display JSON for other details
            st.markdown("#### 🔍 State Update Details")
            st.json(value)

# def get_graph_data(current_node=None, completed_nodes=None, height=500, static=False):
#     if completed_nodes is None:
#         completed_nodes = []
#     
#     nodes = []
#     edges = []
    
#     # Interaction settings
#     kwargs = {}
#     if static:
#         kwargs = {
#             "interaction": {"zoomView": False, "dragView": False, "dragNodes": False}
#         }
# 
#     # --- MODE 1: Static Architecture (Initial View) ---
#     if not completed_nodes:
#         nodes.append(Node(id="Start", label="Start", size=20, shape="diamond", color="#EEEEEE"))
#         nodes.append(Node(id="planner_agent", label="Planner", size=30, shape="hexagon", color="#EEEEEE"))
#         
#         agents = ["data_agent", "quant_agent", "exec_agent", "analyst_agent"]
#         labels = ["Data Agent", "Quant Agent", "Exec Agent", "Analyst Agent"]
#         
#         for agent, label in zip(agents, labels):
#             nodes.append(Node(id=agent, label=label, size=25, shape="box", color="#EEEEEE"))
#             
#         nodes.append(Node(id="End", label="End", size=20, shape="dot", color="#EEEEEE"))
#         
#         edges.append(Edge(source="Start", target="planner_agent"))
#         edges.append(Edge(source="planner_agent", target="data_agent", label="Data"))
#         edges.append(Edge(source="data_agent", target="planner_agent"))
#         edges.append(Edge(source="planner_agent", target="quant_agent", label="Code"))
#         edges.append(Edge(source="quant_agent", target="planner_agent"))
#         edges.append(Edge(source="planner_agent", target="exec_agent", label="Exec"))
#         edges.append(Edge(source="exec_agent", target="planner_agent"))
#         edges.append(Edge(source="planner_agent", target="analyst_agent", label="Analysis"))
#         edges.append(Edge(source="analyst_agent", target="planner_agent"))
#         edges.append(Edge(source="planner_agent", target="End", label="Done"))
#         
#         config = Config(width=None, 
#                         height=height, 
#                         directed=True, 
#                         physics=True, 
#                         hierarchical=False,
#                         nodeHighlightBehavior=True, 
#                         highlightColor="#F7A7A6",
#                         collapsible=False,
#                         **kwargs)
#         return nodes, edges, config
# 
#     # --- MODE 2: Execution Trace (Running View) ---
#     else:
#         # Always start with Start node
#         nodes.append(Node(id="Start", label="Start", size=20, shape="diamond", color="#00CC66", font={"color": "white"}))
#         previous_id = "Start"
#         
#         for i, agent_name in enumerate(completed_nodes):
#             node_id = f"{i}_{agent_name}"
#             
#             # Determine Label & Shape
#             label = agent_name.replace("_agent", "").title()
#             if "planner" in agent_name:
#                 shape = "hexagon"
#                 size = 25
#                 label = "Planner"
#             else:
#                 shape = "box"
#                 size = 20
#             
#             # Determine Color
#             if i == len(completed_nodes) - 1:
#                 color = "#FF9900" # Active (Orange)
#             else:
#                 color = "#00CC66" # Completed (Green)
#                 
#             nodes.append(Node(id=node_id, label=label, size=size, shape=shape, color=color, font={"color": "white"}))
#             edges.append(Edge(source=previous_id, target=node_id))
#             previous_id = node_id
#             
#         # Use Hierarchical Layout for clear timeline
#         config = Config(width=None, 
#                         height=height, 
#                         directed=True, 
#                         physics=False, 
#                         hierarchical=True, # Tree layout
#                         # direction="UD", # Up-Down is default
#                         nodeHighlightBehavior=True, 
#                         highlightColor="#F7A7A6",
#                         collapsible=False,
#                         **kwargs)
#                         
#         return nodes, edges, config

st.title("🤖 Natural Language to Quant Platform")
st.markdown("""
Welcome! I can help you with financial analysis and backtesting using natural language.
Supported by **Tushare** data and **VectorBT**.
""")

# Sidebar for Configuration
with st.sidebar:
    st.header("Configuration")
    tushare_token = st.text_input("Tushare Token", type="password", value=os.getenv("TUSHARE_TOKEN", ""))

    supported = AppConfig.SUPPORTED_LLM_PROVIDERS
    current_provider = os.getenv("LLM_PROVIDER", AppConfig.LLM_PROVIDER)
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
        value=os.getenv(f"{prefix}_MODEL_NAME", AppConfig.PROVIDER_DEFAULT_MODELS.get(provider, "")),
    )
    base_url = st.text_input(
        "LLM Base URL",
        value=os.getenv(f"{prefix}_BASE_URL", AppConfig.PROVIDER_DEFAULT_BASE_URL.get(provider, "")),
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

    st.markdown("### Quick Start")
    example_prompts = [
        "对 600519.SH 进行双均线回测",
        "对 600519.SH 进行双均线策略回测：MA10 > MA50 买入，MA10 < MA50 卖出",
        "获取中国平安(601318.SH)的数据并展示收盘价"
    ]
    for prompt_text in example_prompts:
        if st.button(prompt_text):
            st.session_state["auto_prompt"] = prompt_text

    st.divider()
    st.header("History")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.session_id = create_new_session()
        st.session_state.messages = []
        st.session_state.last_trace = None
        st.rerun()

    st.markdown("---")
    sessions = list_sessions()
    for s in sessions:
        # Highlight current session
        label = f"📄 {s['title']}"
        if s['id'] == st.session_state.session_id:
            label = f"**{label}** (Current)"
            
        if st.button(label, key=s['id'], use_container_width=True):
            st.session_state.session_id = s['id']
            loaded = load_session(s['id'])
            if loaded:
                st.session_state.messages = loaded['messages']
            st.session_state.last_trace = None
            st.rerun()

# Chat Interface Placeholder
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Try: 'Backtest a generic crossover strategy on 600519.SH'")
if "auto_prompt" in st.session_state:
    prompt = st.session_state.pop("auto_prompt")

# Render the last execution trace if available (Persistent View)
# Only render if we are NOT starting a new execution (prompt is None)
if "last_trace" in st.session_state and st.session_state.last_trace and not prompt:
    render_trace_ui(st.session_state.last_trace)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Graph Area - Removed
        # with st.expander("View Agent Graph", expanded=False):
        #     graph_placeholder = st.empty()
        #     with graph_placeholder:
        #         nodes, edges, config = get_graph_data(height=500, static=False)
        #         agraph(nodes=nodes, edges=edges, config=config)
        
        with st.status("Processing...", expanded=True) as status:
            st.write("Initializing Agent Graph...")
            from app.graph import create_graph
            app = create_graph()
            
            # Initialize state
            initial_state = {
                "messages": [HumanMessage(content=prompt)],
                # "tickers": [], # Let the agent extract it
                # "start_date": "20230101",
                # "end_date": "20231231",
                "llm_provider": provider.lower() if provider else None,
                "llm_model": model_name.strip() if model_name else None,
            }
            
            # Run the graph
            st.write("Running Workflow...")
            full_state = initial_state.copy()
            completed_nodes = []
            execution_steps = []
            
            st.write(f"{AGENT_ICONS['planner_agent']} **Agent:** Planner is thinking...")
            
            step_start_time = time.time()
            
            for output in app.stream(initial_state):
                step_duration = time.time() - step_start_time
                
                for key, value in output.items():
                    # Update Graph - Removed
                    completed_nodes.append(key)
                    # with graph_placeholder:
                    #     nodes, edges, config = get_graph_data(current_node=key, completed_nodes=completed_nodes, height=500, static=False)
                    #     agraph(nodes=nodes, edges=edges, config=config)
                    
                    # Update the full state with the new information from this step
                    full_state.update(value)
                    
                    # Collect step data for persistence
                    execution_steps.append({
                        "key": key,
                        "value": value,
                        "full_state": full_state.copy(),
                        "duration": step_duration
                    })
                    
                    step_idx = len(execution_steps)
                    icon = AGENT_ICONS.get(key, "🤖")
                    
                    # Display duration for the completed step
                    st.caption(f"⏱️ Step completed in {step_duration:.2f}s")
                    
                    # --- RENDERING MOVED TO AGENTS FOR LIVE VIEW ---
                    # The agents now handle their own UI rendering inside st.expander blocks.
                    # We only track state here.

                    # Show status for next step
                    if key == "planner_agent":
                        next_step = value.get("next_step")
                        if next_step and next_step != "FINISH":
                             icon = AGENT_ICONS.get(next_step, "🤖")
                             # st.write(f"{icon} **Agent:** {next_step} is working...") 
                             # Commented out to avoid clutter, let the agent announce itself
                    else:
                        # All other agents go back to planner
                        # st.write(f"{AGENT_ICONS['planner_agent']} **Agent:** Planner is thinking...")
                        pass
                
                # Reset timer for the next step
                step_start_time = time.time()
            
            status.update(label="Complete!", state="complete", expanded=False)
            final_state = full_state
            
            # Save trace to session state for persistence
            st.session_state.last_trace = {
                "nodes": [],
                "edges": [],
                "config": None,
                "steps": execution_steps
            }
        
        # Display Analyst Response
        last_msg = final_state["messages"][-1]
        st.markdown(last_msg.content)
        
        # Display Plot
        if "figure_json" in final_state and final_state["figure_json"]:
            st.plotly_chart(final_state["figure_json"], key="final_figure_json")

        # Display Analyst Figures
        if "analyst_figures" in final_state and final_state["analyst_figures"]:
            for i, fig in enumerate(final_state["analyst_figures"]):
                st.plotly_chart(fig, key=f"final_analyst_chart_{i}")
        
        # Display Analyst Data
        if "analyst_data" in final_state and final_state["analyst_data"]:
            for title, df in final_state["analyst_data"].items():
                with st.expander(title, expanded=False):
                    st.dataframe(df)
            
        # Append to session state
        st.session_state.messages.append({"role": "assistant", "content": last_msg.content})
        
        # Save session
        save_session(st.session_state.session_id, st.session_state.messages)

