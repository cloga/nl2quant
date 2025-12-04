from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """
    The state of the agent workflow.
    """
    messages: Annotated[List[Any], add_messages]  # Chat history
    
    # Data Context
    tickers: Optional[List[str]]
    start_date: Optional[str]
    end_date: Optional[str]
    market_data: Optional[Dict[str, Any]] # Serialized DataFrame or reference
    
    # Strategy Context
    strategy_code: Optional[str]
    
    # Execution Results
    execution_output: Optional[str] # Stdout
    performance_metrics: Optional[Dict[str, float]]
    figure_json: Optional[str] # JSON representation of Plotly figure
    
    # Analyst Output
    analyst_figures: Optional[List[Any]] # List of Plotly figures
    analyst_data: Optional[Dict[str, Any]] # Dict of title -> DataFrame

    # LLM Overrides
    llm_provider: Optional[str]
    llm_model: Optional[str]

    # Flow Control
    next_step: Optional[str]
    sender: Optional[str]
    feedback: Optional[str]
    retry_count: Optional[int]
    reasoning: Optional[str]
