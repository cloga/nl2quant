from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    The state of the agent workflow.
    """
    messages: List[Dict[str, str]]  # Chat history
    
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
    
    # LLM Overrides
    llm_provider: Optional[str]
    llm_model: Optional[str]

    # Flow Control
    next_step: Optional[str]
