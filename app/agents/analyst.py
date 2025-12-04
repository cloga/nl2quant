from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.llm import get_llm
from app.state import AgentState

def analyst_agent(state: AgentState):
    """
    Agent responsible for summarizing the backtest results.
    """
    print("--- ANALYST AGENT ---")
    provider = state.get("llm_provider")
    model = state.get("llm_model")
    llm = get_llm(provider=provider, model=model)
    
    metrics = state.get("performance_metrics", {})
    logs = state.get("execution_output", "")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Senior Financial Analyst. 
        Interpret the following backtest results and provide a concise summary.
        Highlight the key metrics (Sharpe, Return, Drawdown).
        If there were execution errors, explain them.
        
        Metrics: {metrics}
        Logs: {logs}
        """),
        ("user", "Please summarize the strategy performance.")
    ])
    
    chain = prompt | llm
    response = chain.invoke({
        "metrics": str(metrics),
        "logs": logs
    })
    
    return {
        "messages": [response]
    }
