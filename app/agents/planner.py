from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
import streamlit as st
from app.llm import get_llm
from app.state import AgentState
from app.ui_utils import render_live_timer, display_token_usage

def planner_agent(state: AgentState):
    """
    Supervisor agent that plans the workflow and reflects on results.
    """
    print("--- PLANNER AGENT ---")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm = get_llm(provider=provider, model=model)
    
    # Gather Context
    messages = state.get("messages", [])
    user_input = ""
    if messages:
        # Prefer the most recent human instruction to reflect the latest request context
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_input = message.content
                break
        if not user_input:
            user_input = messages[-1].content
    
    has_data = "Yes" if state.get("market_data") else "No"
    has_code = "Yes" if state.get("strategy_code") else "No"
    
    exec_output = state.get("execution_output", "")
    metrics = state.get("performance_metrics")
    has_metrics = "Yes" if metrics else "No"
    analysis_completed_flag = bool(state.get("analysis_completed"))
    analysis_completed = "Yes" if analysis_completed_flag else "No"
    analysis_runs = state.get("analysis_runs", 0) or 0
    
    retry_count = state.get("retry_count", 0)
    
    # Heuristic/LLM Decision Logic
    # We use an LLM to make the decision dynamic and "reflective"
    
    system_prompt = """You are the Manager and Planner of a quantitative trading platform.
    Your goal is to fulfill the user's request by orchestrating the following agents:
    
    1. `data_agent`: Fetches market data. (Requires: Ticker symbol)
    2. `quant_agent`: Generates Python strategy code. (Requires: Market Data)
    3. `exec_agent`: Executes the strategy code. (Requires: Strategy Code)
    4. `analyst_agent`: Analyzes the backtest results. (Requires: Execution Metrics)
    
    Current Status:
    - Data Available: {has_data}
    - Code Available: {has_code}
    - Execution Metrics Available: {has_metrics}
    - Analysis Completed: {analysis_completed}
    - Analyst Runs: {analysis_runs}
    - Retry Count: {retry_count}
    
    Last Execution Output (if any):
    {exec_output}
    
    User Request:
    {user_input}
    
     Decision Logic:
     1. **Analyze User Intent**:
         - If the user ONLY wants to fetch/see data (e.g., "Show me the price of AAPL", "Get data for 600519"), you DO NOT need `quant_agent` or `exec_agent`.
         - If the user wants to BACKTEST a strategy (e.g., "Run a moving average strategy", "Backtest RSI"), you need the full pipeline.
    
     2. **Step Selection**:
         - If Data is missing, call `data_agent`.
         - If Data is present:
            - If User Intent is JUST DATA or a quick confirmation, write a concise Chinese reply yourself and choose `FINISH`.
            - If User Intent is BACKTEST:
              - If Code is missing, call `quant_agent`.
              - If Code is present but Metrics are missing, call `exec_agent`.
              - If Execution failed, call `quant_agent` (retry).
              - If Execution succeeded and the user expects professional interpretation, call `analyst_agent`.
    
     3. **Termination**:
         - Provide the final answer and return `FINISH` once the user's goal is satisfied, regardless of whether the analyst was used.
         - Only call `analyst_agent` when deeper or more specialized analysis adds value.
         - Only call `analyst_agent` again if you clearly explain in Reasoning why deeper analysis or revisions are required.
         - If Retry Count > 3, return `FINISH`.
    
     4. **Final Reply**:
         - When you choose `FINISH`, include a short, user-facing answer in Chinese after `FinalResponse:`.
         - When you choose another agent, set `FinalResponse` to `N/A`.
    
     Return your response in the following format:
     Reasoning: <Explanation of why you are choosing the next step, including any reflection on previous errors. IMPORTANT: The content of Reasoning MUST be in Chinese (Simplified Chinese).>
     Decision: <Next Agent Name or FINISH>
     FinalResponse: <Your final answer if Decision is FINISH, otherwise N/A>
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "What is the next step?")
    ])
    
    chain = prompt | llm
    
    input_vars = {
        "has_data": has_data,
        "has_code": has_code,
        "has_metrics": has_metrics,
        "analysis_completed": analysis_completed,
        "analysis_runs": analysis_runs,
        "retry_count": retry_count,
        "exec_output": exec_output[-500:] if exec_output else "None", # Truncate log
        "user_input": user_input
    }
    
    # Capture formatted messages for debugging/UI
    formatted_messages = prompt.format_messages(**input_vars)
    formatted_prompt = "\n\n".join([f"**{m.type.upper()}**: {m.content}" for m in formatted_messages])
    
    with st.expander("🧭 Planner Agent", expanded=True):
        timer = render_live_timer("⏳ Planner is thinking...")
        response = chain.invoke(input_vars)
        timer.empty() # Clear timer when done
        
        display_token_usage(response)
        
        with st.expander("🧠 View Raw Prompt & Response", expanded=False):
            st.markdown("**📝 Prompt:**")
            st.code(formatted_prompt, language="markdown")
            st.markdown("**💬 Response:**")
            st.code(response.content, language="text")
        
        content = response.content.strip()

        final_response = ""
        decision_text = content
        if "FinalResponse:" in content:
            decision_text, final_response_section = content.split("FinalResponse:", 1)
            final_response = final_response_section.strip()

        reasoning = ""
        decision = "FINISH"

        if "Decision:" in decision_text:
            parts = decision_text.split("Decision:")
            reasoning = parts[0].replace("Reasoning:", "").strip()
            decision = parts[1].strip().replace("'", "").replace('"', "")
        else:
            # Fallback if format is not followed
            decision = decision_text.replace("'", "").replace('"', "")
            reasoning = "No reasoning provided."
        
        # Basic validation of decision
        valid_agents = ["data_agent", "quant_agent", "exec_agent", "analyst_agent", "FINISH"]
        if decision not in valid_agents:
            # Fallback logic if LLM hallucinates
            if has_data == "No": decision = "data_agent"
            elif has_code == "No": decision = "quant_agent"
            elif has_metrics == "No": decision = "exec_agent"
            else: decision = "analyst_agent"

        if decision == "analyst_agent" and analysis_runs > 0:
            followup_keywords = ["进一步", "深入", "复盘", "补充", "不满意", "重新分析", "更详细", "follow", "再分析"]
            needs_followup = any(keyword in reasoning for keyword in followup_keywords)
            if needs_followup:
                analysis_completed_flag = False
            else:
                decision = "FINISH"
                final_response = final_response or reasoning

        normalized_final = final_response.strip().lower()
        if normalized_final in {"", "n/a", "na", "none", "null", "无", "暂无"}:
            final_response = ""
            
        print(f"Planner Decision: {decision}")
        
        st.markdown(f"**Reasoning:** {reasoning}")
        st.markdown(f"**Decision:** `{decision}`")
        if final_response:
            st.markdown(f"**Final Response:** {final_response}")
    
    # If retrying, increment counter
    if decision == "quant_agent" and has_code == "Yes":
        retry_count += 1
    
    update = {
        "next_step": decision,
        "retry_count": retry_count,
        "sender": "planner_agent",
        "reasoning": reasoning,
        "analysis_completed": analysis_completed_flag,
        "analysis_runs": analysis_runs,
        "llm_interaction": {
            "input": input_vars,
            "prompt": formatted_prompt,
            "response": content
        }
    }

    if final_response and decision == "FINISH":
        update["messages"] = [AIMessage(content=final_response)]

    return update
