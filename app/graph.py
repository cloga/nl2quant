from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.agents.data_agent import data_agent
from app.agents.quant_agent import quant_agent
from app.agents.exec_agent import exec_agent
from app.agents.analyst import analyst_agent
from app.agents.planner import planner_agent
from app.agents.macro_agent import macro_agent
from app.agents.valuation_agent import valuation_agent

def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("planner_agent", planner_agent)
    workflow.add_node("data_agent", data_agent)
    workflow.add_node("quant_agent", quant_agent)
    workflow.add_node("exec_agent", exec_agent)
    workflow.add_node("analyst_agent", analyst_agent)
    workflow.add_node("macro_agent", macro_agent)
    workflow.add_node("valuation_agent", valuation_agent)

    # Define Edges
    # The Planner decides the next step
    workflow.set_entry_point("planner_agent")
    
    # Conditional edge from planner
    workflow.add_conditional_edges(
        "planner_agent",
        lambda state: state["next_step"],
        {
            "data_agent": "data_agent",
            "quant_agent": "quant_agent",
            "exec_agent": "exec_agent",
            "analyst_agent": "analyst_agent",
            "macro_agent": "macro_agent",
            "valuation_agent": "valuation_agent",
            "FINISH": END
        }
    )
    
    # All agents report back to planner
    workflow.add_edge("data_agent", "planner_agent")
    workflow.add_edge("quant_agent", "planner_agent")
    workflow.add_edge("exec_agent", "planner_agent")
    workflow.add_edge("analyst_agent", "planner_agent")
    workflow.add_edge("macro_agent", "planner_agent")
    workflow.add_edge("valuation_agent", "planner_agent")

    return workflow.compile()
