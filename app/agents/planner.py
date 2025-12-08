import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
import streamlit as st
from app.state import AgentState
from app.ui_utils import render_live_timer, display_token_usage
from app.config import Config

# Simple in-memory cache for planner decisions to speed up repeated states
_planner_cache = {}

# Cache for ambiguity judgments to avoid repeated LLM calls on the same text
_ambig_cache = {}

def planner_agent(state: AgentState):
    """
    Supervisor agent that plans the workflow and reflects on results.
    Three-tier flow:
    1) Hard bypass when user forces a specific agent (no LLM calls).
    2) deepseek-chat for intent + complexity check; simple paths use fast heuristics.
    3) deepseek-reasoner plans only when the task is classified as complex.
    """
    print("--- PLANNER AGENT ---")
    # LLM settings (provider can be set via env; default in Config)
    provider = state.get("llm_provider") or Config.LLM_PROVIDER
    settings = Config.get_llm_settings(provider)
    planner_api_key = settings.get("api_key")
    planner_base = settings.get("base_url")
    planner_model = settings.get("model")

    def make_llm(model_override=None, temperature: float = 0, max_tokens=None):
        return ChatOpenAI(
            model=model_override or planner_model,
            api_key=planner_api_key,
            base_url=planner_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    # Manual override: allow upstream to force a specific agent
    force_agent = state.get("force_agent")

    # Gather Context early for classification
    messages = state.get("messages", [])
    user_input = ""
    if messages:
        # Prefer the most recent human instruction to reflect the latest request context
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                user_input = message.content
                break
        if not user_input and messages:
            user_input = messages[-1].content

    # Ambiguity check: focus on missing action/intent (not ticker suffix). Skip when force_agent already set.
    def is_ambiguous(text: str) -> bool:
        if not text:
            return True
        stripped = text.strip()
        if len(stripped) == 0:
            return True

        # allow explicit slash agent commands to pass through
        if stripped.startswith("/"):
            return False

        cache_key = stripped.lower()
        if cache_key in _ambig_cache:
            return _ambig_cache[cache_key]

        # Heuristics: short, no verb/action, or conflicting multi-intent without priority
        action_keywords = [
            "行情", "价格", "收盘", "分时", "K线", "走势", "指标", "均线", "RSI", "MACD",
            "回测", "策略", "代码", "执行", "交易", "下单",
            "估值", "分位", "便宜", "贵", "性价比",
            "宏观", "点评",
            "财报", "年报", "季报", "基本面", "市值", "pe", "pb", "roe", "营收", "净利"
        ]

        lt = stripped.lower()
        has_action_kw = any(kw in lt for kw in action_keywords)
        too_short = len(stripped) < 8
        # conflicting intents: mentions both 回测/策略 and 估值; or only says 看看/分析 without object
        conflict = ("回测" in stripped and ("估值" in stripped or "分位" in stripped))
        no_action = not has_action_kw
        heuristic_ambig = too_short or no_action or conflict

        # LLM confirmation (short, deterministic)
        clf_prompt = ChatPromptTemplate.from_messages([
            ("system", "Decide if the user request is clear about WHAT ACTION to perform on the ticker(s). Reply with 'clear' or 'ambiguous' only."),
            ("user", "{text}")
        ])

        is_ambig = heuristic_ambig
        try:
            clf_llm = make_llm(max_tokens=2)
            resp = (clf_prompt | clf_llm).invoke({"text": stripped[:400]})
            result = (resp.content or "").strip().lower()
            is_ambig = heuristic_ambig or (result != "clear")
        except Exception:
            pass

        _ambig_cache[cache_key] = is_ambig
        return is_ambig

    # When caller already forced an agent (e.g., /data 000001.SZ), skip ambiguity asks.
    if not force_agent and is_ambiguous(user_input):
        clarification = "你想对这只标的做什么？示例：`/data 600519.SH 查看行情`，`回测 600519.SH 双均线策略`，`估值 600519.SH 当前分位`。如只想看价格可直接回复 `/data 600519.SH`。"
        reasoning_text = "缺少具体操作意图，需要确认是行情/回测/估值/宏观/财报等哪类请求。"
        try:
            ambig_prompt = ChatPromptTemplate.from_messages([
                ("system", """你需要说明为何要澄清，并给出简短澄清问题。严格输出两行：
Reasoning: <简洁中文，说明缺少具体操作/意图>
Clarification: <中文澄清提问，给出 2-3 个示例，覆盖行情/回测/估值 等，如 /data 600519.SH>
"""),
                ("user", "{text}")
            ])
            ambig_llm = make_llm(max_tokens=80)
            resp = (ambig_prompt | ambig_llm).invoke({"text": user_input[:400]})
            content = (resp.content or "").strip()
            if "Clarification:" in content:
                head, tail = content.split("Clarification:", 1)
                reasoning_text = head.replace("Reasoning:", "").strip() or reasoning_text
                clarification = tail.strip() or clarification
            else:
                reasoning_text = content or reasoning_text
        except Exception:
            pass
        return {
            "next_step": "FINISH",
            "retry_count": state.get("retry_count", 0),
            "sender": "planner_agent",
            "reasoning": reasoning_text,
            "analysis_completed": state.get("analysis_completed"),
            "analysis_runs": state.get("analysis_runs", 0),
            "force_agent": None,
            "need_disambiguation": True,
            "messages": [AIMessage(content=clarification)],
            "llm_interaction": {
                "input": {"ambiguous": True, "user_input": user_input},
                "prompt": "<ambiguous guard>",
                "response": clarification
            }
        }

    # Allow inline agent markers in the prompt (fallback in case frontend did not set force_agent)
    if not force_agent and user_input:
        marker_map = {
            "/data": "data_agent",
            "/quant": "quant_agent",
            "/exec": "exec_agent",
            "/analyst": "analyst_agent",
            "/macro": "macro_agent",
            "/valuation": "valuation_agent",
            "data_agent": "data_agent",
            "quant_agent": "quant_agent",
            "exec_agent": "exec_agent",
            "analyst_agent": "analyst_agent",
            "macro_agent": "macro_agent",
            "valuation_agent": "valuation_agent",
        }
        lower_text = user_input.strip().lower()
        for marker, agent in marker_map.items():
            if lower_text.startswith(marker):
                force_agent = agent
                break

    # Cheap, no-LLM flag detection for bypass scenarios
    def heuristic_flags(text: str):
        if not text:
            return False, False
        lt = text.lower()
        need_full = any(k in lt for k in ["all history", "full history", "entire history", "全历史", "全部历史", "全量" ])
        need_bench = any(k in lt for k in ["benchmark", "bench", "对比", "相对", "超额", "基准", "指数"])
        return need_full, need_bench

    # Tier 1: short-circuit on forced agent BEFORE any LLM calls
    if force_agent in {"data_agent", "quant_agent", "exec_agent", "analyst_agent", "macro_agent", "valuation_agent"}:
        need_full_history_flag, needs_benchmark_flag = heuristic_flags(user_input)
        flag_payload = {
            "need_full_history": need_full_history_flag,
            "needs_benchmark": needs_benchmark_flag,
        }
        reasoning = f"收到直接调用指令，转交 {force_agent}。"
        return {
            "next_step": force_agent,
            "retry_count": state.get("retry_count", 0),
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": state.get("analysis_completed"),
            "analysis_runs": state.get("analysis_runs", 0),
            "force_agent": None,  # clear override after one hop
            **flag_payload,
            "llm_interaction": {
                "input": {"force_agent": force_agent},
                "prompt": "<skipped - forced agent>",
                "response": "<forced>"
            }
        }

    # Lightweight LLM extraction for flags (full history / benchmark) to avoid keyword explosion
    def extract_flags(text: str):
        if not text:
            return False, False
        flag_prompt = ChatPromptTemplate.from_messages([
            ("system", "You extract booleans for data needs. Reply with two lines exactly: full_history: yes/no ; benchmark: yes/no. Decide if the user wants ALL history (full/entire/全部) and whether a benchmark/comparison is requested (对比/超额/相对/benchmark)."),
            ("user", "{text}")
        ])
        clf_llm = ChatOpenAI(
            model="deepseek-chat",
            api_key=planner_api_key,
            base_url=planner_base,
            temperature=0,
            max_tokens=10,
        )
        try:
            resp = (flag_prompt | clf_llm).invoke({"text": text[:400]})
            out = resp.content.lower()
            full = "full_history: yes" in out or "full_history: true" in out
            bench = "benchmark: yes" in out or "benchmark: true" in out
            return full, bench
        except Exception:
            return False, False

    need_full_history_flag, needs_benchmark_flag = extract_flags(user_input)
    flag_payload = {
        "need_full_history": need_full_history_flag,
        "needs_benchmark": needs_benchmark_flag,
    }

    # Keep legacy interpret hook (default off). Future: move to LLM signals.
    interpret_request = False

    # Direct short-circuit if force_agent provided
    if force_agent in {"data_agent", "quant_agent", "exec_agent", "analyst_agent", "macro_agent", "valuation_agent"}:
        reasoning = f"收到直接调用指令，转交 {force_agent}。"
        return {
            "next_step": force_agent,
            "retry_count": state.get("retry_count", 0),
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": state.get("analysis_completed"),
            "analysis_runs": state.get("analysis_runs", 0),
            "force_agent": None,  # clear override after one hop
            **flag_payload,
            "llm_interaction": {
                "input": {"force_agent": force_agent},
                "prompt": "<skipped - forced agent>",
                "response": "<forced>"
            }
        }

    # Helper: intent classification
    def classify_intent(user_text: str) -> str:
        """Return one of: data_query, quant_strategy, macro, valuation, news_info, other."""
        if not user_text:
            return "other"
        intent_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a fast, deterministic intent classifier for a quant assistant. Output exactly one label from this set (lowercase):
- data_query     # 单标的行情/指标/走势/简单分析/获取数据
 - data_query     # 单标的行情/指标/走势/简单分析/获取数据/基础面简析（市值/PE/PB/ROE/营收/净利/财报摘取）
- quant_strategy # 回测/策略/指标/组合/仓位/调仓/风险控制/回测代码/执行策略
- macro          # 宏观经济/利率/通胀/增长/流动性/信用/汇率/宏观事件解读
- valuation      # 估值/性价比/相对位置/分位/便宜贵/估值与收益对比/估值分位
- news_info      # 资讯/新闻/事件/公告/宏观消息类查询
- other          # 以上均不匹配

Few-shot guidance:
Q: 获取中国平安601318.SH的收盘价
A: data_query
Q: 对 600519.SH 做双均线回测，MA5>MA20 进场
A: quant_strategy
Q: 简要点评当前宏观流动性
A: macro
Q: 评估 300750.SZ 当前估值分位
A: valuation
Q: 今天有无美联储相关新闻
A: news_info
Q: 看看 300750.SZ 的市值和PE
A: data_query
Q: 摘要 300750.SZ 最近一季财报，给关键指标
A: data_query
Q: 588000 估值算贵吗，分位在哪
A: valuation
Q: 588000 做双均线回测
A: quant_strategy
Q: 帮我写一首诗
A: other

Reply with the label only.
"""
            ),
            ("user", "{text}")
        ])
        clf_llm = make_llm(max_tokens=6)
        try:
            resp = (intent_prompt | clf_llm).invoke({"text": user_text[:400]})
            label = resp.content.strip().lower()
            if "data" in label:
                return "data_query"
            if "strategy" in label or "quant" in label or "组合" in label or "策略" in label:
                return "quant_strategy"
            if "macro" in label or "宏观" in label:
                return "macro"
            if "valuation" in label or "估值" in label or "性价比" in label or "分位" in label:
                return "valuation"
            if "news" in label or "info" in label or "资讯" in label or "新闻" in label:
                return "news_info"
            return "other"
        except Exception:
            return "other"

    intent_label = classify_intent(user_input)

    # Helper: quick classification to decide if task is complex (Tier 2 => choose chat vs reasoner)
    def is_complex_task(user_text: str) -> bool:
        if not user_text:
            return False
        clf_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a fast task classifier. Reply with 'simple' or 'complex' only."),
            ("user", "Classify the task: {task}")
        ])
        clf_llm = make_llm(max_tokens=4)
        try:
            resp = (clf_prompt | clf_llm).invoke({"task": user_text[:400]})
            text = resp.content.strip().lower()
            return any(k in text for k in ["complex", "复杂"])
        except Exception:
            return False

    complex_flag = is_complex_task(user_input)
    allow_fastpath = not complex_flag  # simple tasks take deterministic shortcuts; complex tasks defer to planner LLM
    # Use a heavier model only if provider is deepseek and marked complex; otherwise use provider default.
    planner_model_name = "deepseek-reasoner" if (settings.get("provider") == "deepseek" and complex_flag) else planner_model
    llm = make_llm(model_override=planner_model_name, temperature=0)
    
    has_data = "Yes" if state.get("market_data") else "No"
    data_failed = bool(state.get("data_failed"))
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

    Detected Intent: {intent_label} (data_query | quant_strategy | macro | valuation | news_info | other)
    Flags from intent extraction:
    - need_full_history: {need_full_history}
    - needs_benchmark: {needs_benchmark}
    
     Decision Logic:
     1. **Analyze User Intent**:
         - If the user ONLY wants to fetch/see data (e.g., "Show me the price of AAPL", "Get data for 600519"), you DO NOT need `quant_agent` or `exec_agent`.
         - If the user wants to BACKTEST a strategy (e.g., "Run a moving average strategy", "Backtest RSI"), you need the full pipeline.
    
     2. **Step Selection (intent-aware)**:
         - If intent is `news_info`: politely reply in Chinese that the system专注行情/回测，不提供资讯抓取，并提示用户给出具体数据/回测需求；then `FINISH`.
         - If intent is `other`: reply briefly in Chinese that当前能力聚焦行情、回测、策略，请提供相关请求；then `FINISH`.
         - If intent is `macro`: call `macro_agent` for宏观分析；如需补充信息可在 Reasoning 说明。
         - If intent is `valuation`: 如果缺少数据先调用 `data_agent` 获取行情；若已有数据，调用 `valuation_agent` 做估值相对位置分析。
         - If intent is `data_query`:
             - If Data is missing, call `data_agent`.
             - If Data is present and no backtest is asked, give a concise Chinese reply (可提醒可视化/指标需求) and `FINISH`.
         - If intent is `quant_strategy` (含组合/指标/策略/回测):
             - If Data is missing, call `data_agent`.
             - If Data is present and Code is missing, call `quant_agent`.
             - If Code is present but Metrics are missing, call `exec_agent`.
             - If Execution failed, call `quant_agent` (retry).
             - If Execution succeeded and the user expects interpretation, call `analyst_agent`.
    
     3. **Termination**:
         - Provide the final answer and return `FINISH` once the user's goal is satisfied, regardless of whether the analyst was used.
         - Only call `analyst_agent` when deeper or more specialized analysis adds value.
         - Only call `analyst_agent` again if you clearly explain in Reasoning why deeper analysis or revisions are required.
         - If Retry Count > 3, return `FINISH`.
    
     4. **Final Reply**:
         - When you choose `FINISH`, include a short, user-facing answer in Chinese after `FinalResponse:`.
         - When you choose another agent, set `FinalResponse` to `N/A`.
    
    Return your response in the following format:
    Reasoning: <用中文，简洁说明决策，必须包含：用户需求概要、判定的 intent、need_full_history/needs_benchmark 标志、当前已有/缺少的数据或代码、为何选择该下一步>
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
        "exec_output": exec_output[-200:] if exec_output else "None", # shorter for speed
        "user_input": user_input,
        "intent_label": intent_label,
        "need_full_history": need_full_history_flag,
        "needs_benchmark": needs_benchmark_flag,
    }

    # --- Intent/early handling ---
    if data_failed:
        reasoning = "数据获取失败，终止并提示用户检查代码/公司名。"
        return {
            "next_step": "FINISH",
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            "force_agent": None,
            **flag_payload,
            "messages": [AIMessage(content="行情获取失败，请检查股票/指数代码或公司名是否正确。")],
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - data failed>",
                "response": "<data_failed>"
            }
        }

    if intent_label == "news_info":
        reasoning = "意图为资讯/新闻查询，当前系统专注行情、回测和策略，不抓取资讯。"
        return {
            "next_step": "FINISH",
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "messages": [AIMessage(content="当前平台专注行情、回测和策略，不提供资讯抓取。请提供具体的行情或回测需求，我会继续帮助。")],
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - intent guard>",
                "response": "<intent=news_info>"
            }
        }

    if intent_label == "other":
        reasoning = "意图不在支持范围，提示用户提供行情/回测相关需求。"
        return {
            "next_step": "FINISH",
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "messages": [AIMessage(content="当前支持行情获取、量化策略与回测。如果需要，请说明标的和需求。")],
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - intent guard>",
                "response": "<intent=other>"
            }
        }

    if intent_label == "macro":
        reasoning = "意图为宏观分析，转交 macro_agent。"
        return {
            "next_step": "macro_agent",
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - intent macro>",
                "response": "<intent=macro>"
            }
        }

    if intent_label == "valuation":
        if has_data == "No" and allow_fastpath:
            reasoning = "意图为估值/性价比，需先获取行情数据。"
            return {
                "next_step": "data_agent",
                "retry_count": retry_count,
                "sender": "planner_agent",
                "reasoning": reasoning,
                "analysis_completed": analysis_completed_flag,
                "analysis_runs": analysis_runs,
                **flag_payload,
                "llm_interaction": {
                    "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                    "prompt": "<skipped - valuation needs data>",
                    "response": "<intent=valuation>"
                }
            }
        reasoning = "意图为估值/性价比，已有数据，调用 valuation_agent。"
        return {
            "next_step": "valuation_agent",
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - intent valuation>",
                "response": "<intent=valuation>"
            }
        }

    # --- Early-stop heuristic (skip LLM for obvious routes) ---
    if allow_fastpath and has_data == "No":
        decision = "data_agent"
        reasoning = "缺少行情数据，先调用 data_agent 获取数据。"
        update = {
            "next_step": decision,
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - early rule>",
                "response": "<skipped - early rule>"
            }
        }
        return update

    # If user only wants data/quick analysis and data is ready, finish with a short reply
    if allow_fastpath and intent_label == "data_query" and has_data == "Yes":
        if interpret_request:
            reasoning = "用户要求解读/分析已有行情数据，转交 analyst_agent 做数据概览与解读。"
            return {
                "next_step": "analyst_agent",
                "retry_count": retry_count,
                "sender": "planner_agent",
                "reasoning": reasoning,
                "analysis_completed": analysis_completed_flag,
                "analysis_runs": analysis_runs,
                **flag_payload,
                "llm_interaction": {
                    "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                    "prompt": "<skipped - data interpret>",
                    "response": "<intent=data_query, route=analyst>"
                }
            }
        reasoning = "用户意图为行情/单标的查询，数据已具备，直接回复即可。"
        return {
            "next_step": "FINISH",
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            "need_full_history": need_full_history_flag,
            "needs_benchmark": needs_benchmark_flag,
            "messages": [AIMessage(content="行情数据已获取。如需指标/图表/回测，请告知具体需求。")],
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - data intent finish>",
                "response": "<intent=data_query>"
            }
        }

    if allow_fastpath and has_code == "No":
        decision = "quant_agent"
        reasoning = "已有数据但缺少策略代码，调用 quant_agent 生成代码。"
        update = {
            "next_step": decision,
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - early rule>",
                "response": "<skipped - early rule>"
            }
        }
        return update

    if allow_fastpath and has_metrics == "No":
        decision = "exec_agent"
        reasoning = "已有数据和代码，但缺少回测指标，调用 exec_agent 执行回测。"
        update = {
            "next_step": decision,
            "retry_count": retry_count,
            "sender": "planner_agent",
            "reasoning": reasoning,
            "analysis_completed": analysis_completed_flag,
            "analysis_runs": analysis_runs,
            **flag_payload,
            "llm_interaction": {
                "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
                "prompt": "<skipped - early rule>",
                "response": "<skipped - early rule>"
            }
        }
        return update

    # --- Simple cache key to avoid repeated LLM calls on identical state ---
    cache_key = (
        has_data,
        has_code,
        has_metrics,
        analysis_completed,
        analysis_runs,
        retry_count,
        user_input.strip()[:200],
        complex_flag,
        intent_label,
    )
    if cache_key in _planner_cache:
        return _planner_cache[cache_key]
    
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
        valid_agents = ["data_agent", "quant_agent", "exec_agent", "analyst_agent", "macro_agent", "valuation_agent", "FINISH"]
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
        **flag_payload,
        "llm_interaction": {
            "input": {k: ('***MASKED***' if any(x in k.lower() for x in ['token','api_key','secret','password']) else v) for k, v in input_vars.items()},
            "prompt": formatted_prompt,
            "response": content
        }
    }

    if final_response and decision == "FINISH":
        update["messages"] = [AIMessage(content=final_response)]

    return update
