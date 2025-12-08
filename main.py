import streamlit as st
import os
import time
import json
import streamlit.components.v1 as components
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
# from streamlit_agraph import agraph, Node, Edge, Config
from app.config import Config as AppConfig
from app.history import save_session, load_session, list_sessions, create_new_session
from app.ui_utils import mask_secret

# Load environment variables
load_dotenv()

# Agent Icons Mapping
AGENT_ICONS = {
    "planner_agent": "🧭",  # Compass for planning/navigation
    "data_agent": "🗂️",     # Card index for data
    "quant_agent": "📈",    # Chart for quantitative analysis
    "exec_agent": "⚙️",     # Gear for execution
    "analyst_agent": "🧐",   # Monocle face for deep analysis
    "macro_agent": "🌐",     # Globe for macro analysis
    "valuation_agent": "💰"  # Money bag for valuation
}

st.set_page_config(page_title="NL-to-Quant Platform", layout="wide")

def load_quickstart_sections():
    """Load quickstart queries from config file; fall back to defaults if missing/invalid."""
    config_path = os.path.join("app", "quickstart_config.json")
    default_sections = [
        {
            "title": "回测示例",
            "samples": [
                "对 600519.SH 进行双均线回测",
                "对 600519.SH 进行双均线策略回测：MA10 > MA50 买入，MA10 < MA50 卖出"
            ],
        },
        {
            "title": "行情查询",
            "samples": [
                "获取中国平安(601318.SH)的数据并展示收盘价",
                "/data 获取 000300.SH 的行情"
            ],
        },
        {
            "title": "宏观与估值",
            "samples": [
                "/macro 简要点评当前宏观环境",
                "/valuation 评估 300750.SZ 的估值相对位置"
            ],
        },
    ]
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sections = data.get("sections", [])
            if isinstance(sections, list) and sections:
                return sections
        except Exception:
            pass
    return default_sections


if "session_id" not in st.session_state:
    st.session_state.session_id = create_new_session()

if "last_trace" not in st.session_state:
    st.session_state.last_trace = None



if "persist_state" not in st.session_state:
    st.session_state.persist_state = {}

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
        for idx, step in enumerate(steps):
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
                    # Mask any obvious secret keys in the input dict
                    try:
                        safe_input = dict(value["llm_interaction"]["input"]) if isinstance(value["llm_interaction"]["input"], dict) else {"input": str(value["llm_interaction"]["input"]) }
                        for k in list(safe_input.keys()):
                            lk = k.lower()
                            if 'token' in lk or 'api_key' in lk or 'secret' in lk or 'password' in lk:
                                safe_input[k] = mask_secret(safe_input[k])
                    except Exception:
                        safe_input = {"input": "<could not format>"}

                    st.json(safe_input)
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
    st.markdown("**Run Steps**")
    
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

# Load quickstart samples once for sidebar and inline chips
quickstart_sections = load_quickstart_sections()

# Sidebar (no config UI; provider/model are read from env only)
with st.sidebar:
    provider = os.getenv("LLM_PROVIDER", AppConfig.LLM_PROVIDER)
    prefix = f"LLM_{provider.upper()}"
    model_name = os.getenv(f"{prefix}_MODEL_NAME", AppConfig.PROVIDER_DEFAULT_MODELS.get(provider, ""))

    st.markdown("### Quick Start")
    for sec_idx, section in enumerate(quickstart_sections):
        title = section.get("title", f"Section {sec_idx+1}")
        samples = section.get("samples", [])
        if not samples:
            continue
        with st.expander(title, expanded=(sec_idx == 0)):
            for sample_idx, prompt_text in enumerate(samples):
                key = f"qs_{sec_idx}_{sample_idx}"
                if st.button(prompt_text, key=key):
                    st.session_state["auto_prompt"] = prompt_text

    st.divider()
    with st.expander("可用能力 / Slash 命令", expanded=False):
        st.markdown("""
        - `/data` 获取行情（需 Tushare token）
        - `/quant` 生成策略代码（VectorBT）
        - `/exec` 执行已生成策略
        - `/analyst` 解读回测结果
        - `/macro` 宏观解读（结构化总结）
        - `/valuation` 估值相对位置（需已获取行情）
        """)

    st.divider()
    with st.expander("History", expanded=False):
        if st.button("➕ New Chat", use_container_width=True):
            st.session_state.session_id = create_new_session()
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.session_state.persist_state = {}
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

# Inline quickstart chips above input only when disambiguation is needed (planner flagged)
show_chips = st.session_state.get("persist_state", {}).get("need_disambiguation")
if show_chips:
    chip_samples = []
    for section in quickstart_sections:
        chip_samples.extend(section.get("samples", []))
    chip_samples = chip_samples[:8]  # avoid clutter

    if chip_samples:
        st.markdown("**快捷示例：意图不清时可点击填充**")
        rows = [chip_samples[i:i+4] for i in range(0, len(chip_samples), 4)]
        for r_idx, row in enumerate(rows):
            cols = st.columns(len(row))
            for c_idx, sample in enumerate(row):
                if cols[c_idx].button(sample, key=f"chip_{r_idx}_{c_idx}"):
                    st.session_state["auto_prompt"] = sample

placeholder_text = " /macro 宏观解读 · /valuation 估值分位 · /data 拉行情 · /quant 回测策略 · 直接自然语言描述也可"
prompt = st.chat_input(placeholder_text)
if "auto_prompt" in st.session_state:
    prompt = st.session_state.pop("auto_prompt")

# If user starts typing a new prompt, clear disambiguation flag to avoid sticky chips
if prompt:
    if st.session_state.get("persist_state"):
        st.session_state.persist_state.pop("need_disambiguation", None)

# Allow direct agent call via slash command prefix
force_agent = None
prompt_for_agent = prompt
if prompt:
    lower = prompt.lower().strip()
    prefix_map = {
        "/data": "data_agent",
        "/quant": "quant_agent",
        "/exec": "exec_agent",
        "/analyst": "analyst_agent",
        "/macro": "macro_agent",
        "/valuation": "valuation_agent",
    }
    for k, v in prefix_map.items():
        if lower.startswith(k):
            force_agent = v
            prompt_for_agent = prompt[len(k):].strip() or f"(direct call to {v})"
            break

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
            # Carry over context from previous turn (market_data, code, metrics...)
            persisted = st.session_state.get("persist_state", {}) or {}

            # Build short conversation history to aid multi-turn context
            history_msgs = []
            max_history = 8  # last 8 turns (user/assistant)
            prior = st.session_state.messages[-max_history:] if st.session_state.messages else []
            for msg in prior:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user":
                    history_msgs.append(HumanMessage(content=content))
                elif role == "assistant":
                    history_msgs.append(AIMessage(content=content))
            history_msgs.append(HumanMessage(content=prompt_for_agent))

            initial_state = {
                "messages": history_msgs,
                "llm_provider": provider.lower() if provider else None,
                "llm_model": model_name.strip() if model_name else None,
                "force_agent": force_agent,
                **persisted,
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

                    # Display duration for the completed step (shown last for each step)
                # st.caption(f"⏱️ Step completed in {step_duration:.2f}s")
                
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

        # If planner asks for disambiguation, surface actionable buttons
        if final_state.get("need_disambiguation"):
            chip_samples = []
            for section in quickstart_sections:
                chip_samples.extend(section.get("samples", []))
            chip_samples = chip_samples[:8]
            if chip_samples:
                st.markdown("**请选择一个示例快速继续：**")
                rows = [chip_samples[i:i+4] for i in range(0, len(chip_samples), 4)]
                for r_idx, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c_idx, sample in enumerate(row):
                        if cols[c_idx].button(sample, key=f"need_disamb_chip_{r_idx}_{c_idx}"):
                            st.session_state["auto_prompt"] = sample
                            st.experimental_rerun()

        # Display Analyst Response
        last_msg = final_state["messages"][-1]
        st.markdown(last_msg.content)

        # If planner asked for disambiguation, surface clickable chips right below the reply
        if final_state.get("need_disambiguation"):
            chip_samples = []
            for section in quickstart_sections:
                chip_samples.extend(section.get("samples", []))
            chip_samples = chip_samples[:8]
            if chip_samples:
                st.markdown("**快捷示例：点击填充后可直接发送**")
                rows = [chip_samples[i:i+4] for i in range(0, len(chip_samples), 4)]
                for r_idx, row in enumerate(rows):
                    cols = st.columns(len(row))
                    for c_idx, sample in enumerate(row):
                        if cols[c_idx].button(sample, key=f"reply_chip_{r_idx}_{c_idx}"):
                            st.session_state["auto_prompt"] = sample
                            st.experimental_rerun()
        
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

        # Persist key context for follow-up turns (avoid re-fetching data when user asks to “interpret above data”)
        persist_keys = [
            "market_data", "benchmark_data", "benchmark_ticker", "start_date", "end_date", "tickers",
            "strategy_code", "user_edited_code", "code_confirmed", "execution_output", "performance_metrics",
            "portfolio_data", "trades_data", "figure_json", "analysis_completed", "analysis_runs", "valuation",
            "need_full_history", "needs_benchmark", "need_disambiguation"
        ]
        st.session_state.persist_state = {k: final_state.get(k) for k in persist_keys if final_state.get(k) is not None}
        
        # Save session
        save_session(st.session_state.session_id, st.session_state.messages)

