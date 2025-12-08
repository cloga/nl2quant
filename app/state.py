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
    benchmark_ticker: Optional[str]  # e.g., 000300.SH for CSI300
    start_date: Optional[str]
    end_date: Optional[str]
    market_data: Optional[Dict[str, Any]] # Serialized DataFrame or reference
    benchmark_data: Optional[Dict[str, Any]]  # Benchmark price data
    
    # Strategy Context
    strategy_code: Optional[str]
    user_edited_code: Optional[str]  # Human-in-the-loop edited code
    code_confirmed: Optional[bool]  # Whether user has confirmed the code
    
    # Parameter Optimization Context
    optimization_mode: Optional[bool]
    optimization_params: Optional[Dict[str, Any]]  # {param_name: [min, max, step]}
    optimization_results: Optional[Dict[str, Any]]  # Grid search results
    
    # Execution Results
    execution_output: Optional[str] # Stdout
    performance_metrics: Optional[Dict[str, float]]
    portfolio_data: Optional[Dict[str, Any]]  # Serialized portfolio time series
    trades_data: Optional[str]  # Serialized trade records
    figure_json: Optional[str] # JSON representation of Plotly figure
    
    # Benchmark Comparison
    benchmark_metrics: Optional[Dict[str, float]]  # Alpha, Beta, etc.
    
    # Analyst Output
    analyst_figures: Optional[List[Any]] # List of Plotly figures
    analyst_data: Optional[Dict[str, Any]] # Dict of title -> DataFrame
    analysis_completed: Optional[bool]
    analysis_runs: Optional[int]

    # Valuation Output
    valuation: Optional[Dict[str, Any]]

    # Data fetch status
    data_failed: Optional[bool]

    # Planner / intent flags
    need_full_history: Optional[bool]
    needs_benchmark: Optional[bool]

    # LLM Overrides
    llm_provider: Optional[str]
    llm_model: Optional[str]

    # Manual override for entry agent
    force_agent: Optional[str]

    # Flow Control
    next_step: Optional[str]
    sender: Optional[str]
    feedback: Optional[str]
    retry_count: Optional[int]
    reasoning: Optional[str]
