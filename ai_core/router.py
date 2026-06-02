from langgraph.graph import StateGraph, END
from ai_core.state import AgentState
from ai_core.nodes import *

workflow = StateGraph(AgentState)

# ==========================================
# 노드 등록
# ==========================================
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("clarify", clarify_node)

# 일반 대화 경로
workflow.add_node("conversation_memory", conversation_memory_node)
workflow.add_node("general_agent", general_agent_node)

# 코딩 경로
workflow.add_node("code_search", code_search_node)
workflow.add_node("coding_math_agent", coding_math_agent_node)

# 전문가 경로 — Phase 1 검색 인텔리전스
workflow.add_node("query_rewriter",  query_rewriter_node)
workflow.add_node("selective_search", selective_search_node)
workflow.add_node("search_grader",   search_grader_node)

# 전문가 경로 — Phase 1 ReAct 루프
workflow.add_node("expert_agent_react", expert_agent_react_node)
workflow.add_node("expert_tools",       expert_tools_node)   # ToolNode

# 공통 후처리
workflow.add_node("custom_agent_gate", custom_agent_gate_node)
workflow.add_node("dashboard_select",  dashboard_select_node)
workflow.add_node("summary",           summary_node)
workflow.add_node("persona_agent",     persona_agent_node)
workflow.add_node("critic",            critic_node)
workflow.add_node("revision_agent",    revision_agent_node)

# ==========================================
# 진입점 및 Supervisor 분기
# ==========================================
workflow.set_entry_point("supervisor")

def route_from_supervisor(state):
    return state["target_agent_name"]

workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "general_agent":     "conversation_memory",
        "expert_agent":      "query_rewriter",        # Phase 1: 검색 전 쿼리 재작성
        "coding_math_agent": "code_search",
        "clarify":           "clarify",
    }
)

# ==========================================
# Clarify 종착
# ==========================================
workflow.add_edge("clarify", END)

# ==========================================
# 일반 대화 경로
# ==========================================
workflow.add_edge("conversation_memory", "general_agent")

# ==========================================
# 코딩 경로
# ==========================================
workflow.add_edge("code_search", "coding_math_agent")

# ==========================================
# 전문가 경로 — 검색 인텔리전스 루프
#
#  query_rewriter → selective_search → search_grader
#                         ↑                  │ grade=rewrite & attempts<2
#                         └──────────────────┘
#                                            │ grade=sufficient
#                                            ▼
#                                   expert_agent_react
# ==========================================
workflow.add_edge("query_rewriter",  "selective_search")
workflow.add_edge("selective_search", "search_grader")

workflow.add_conditional_edges(
    "search_grader",
    route_from_search_grader,
    {
        "selective_search":  "selective_search",   # 재검색 루프
        "expert_agent":      "expert_agent_react", # 충분 → ReAct 진입
    }
)

# ==========================================
# 전문가 경로 — ReAct 루프
#
#  expert_agent_react ──tool_calls?──▶ expert_tools ──▶ expert_agent_react
#          │ no tool_calls
#          ▼
#   custom_agent_gate
# ==========================================
workflow.add_conditional_edges(
    "expert_agent_react",
    should_use_tools,
    {
        "tools": "expert_tools",        # 도구 실행
        "end":   "custom_agent_gate",   # 최종 답변 완성
    }
)
workflow.add_edge("expert_tools", "expert_agent_react")  # 도구 결과 → 에이전트 재실행

# ==========================================
# 공통 후처리 경로
# ==========================================
workflow.add_edge("general_agent",     "custom_agent_gate")
workflow.add_edge("coding_math_agent", "custom_agent_gate")
workflow.add_edge("custom_agent_gate", "dashboard_select")
workflow.add_edge("dashboard_select",  "summary")
workflow.add_edge("summary",           "persona_agent")
workflow.add_edge("persona_agent",     "critic")

# ==========================================
# Critic 루프 — Phase 1: 최대 2회 revision
# revision_agent → persona_agent (dashboard 재생성 생략)
# ==========================================
workflow.add_conditional_edges(
    "critic",
    check_critic_approval,
    {"end": END, "revision": "revision_agent"}
)
workflow.add_edge("revision_agent", "persona_agent")

# recursion_limit: invoke 시 config로 전달 (LangGraph 1.x 방식)
langgraph_app = workflow.compile()
RECURSION_LIMIT = 50
