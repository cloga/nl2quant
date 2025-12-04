from langgraph.graph import StateGraph, END
from app.state import AgentState
from app.agents.data_agent import data_agent
from app.agents.quant_agent import quant_agent
from app.agents.exec_agent import exec_agent
from app.agents.analyst import analyst_agent
from langchain_core.messages import HumanMessage

def create_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("data_agent", data_agent)
    workflow.add_node("quant_agent", quant_agent)
    workflow.add_node("exec_agent", exec_agent)
    workflow.add_node("analyst_agent", analyst_agent)

    # Define Edges
    # For this MVP, we enforce a linear flow: Data -> Quant -> Exec -> Analyst
    # In a real app, we would have a Router node.
    
    workflow.set_entry_point("data_agent")
    
    workflow.add_edge("data_agent", "quant_agent")
    workflow.add_edge("quant_agent", "exec_agent")
    workflow.add_edge("exec_agent", "analyst_agent")
    workflow.add_edge("analyst_agent", END)

    return workflow.compile()
