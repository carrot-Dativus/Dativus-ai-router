# ai_core/router.py

from langgraph.graph import StateGraph, END
from ai_core.state import AgentState
from ai_core.nodes import (
    initialize_persona_node, greeting_node, search_node, summary_node,
    commander_node, external_llm_node, custom_agent_node, critic_node,
    web_search_node ,external_llm, graph_memory_node
)


# 💡 [신규] 진짜 대빵 관리자(Supervisor) 노드
def supervisor_node(state: AgentState):
    query = state["query"]
    print("\n👔 [Supervisor] 작전 상황을 분석하고 다음 작업 요원을 지정합니다...")

    # 관리자는 현재까지 수집된 서류철 상태를 확인합니다.
    has_team_data = bool(state.get("team_context"))
    has_web_data = bool(state.get("web_context"))

    prompt = f"""당신은 Dativus 시스템의 최고 관리자(Supervisor)입니다.
    사용자의 질문과 현재까지 수집된 데이터 상태를 보고, 다음으로 출동할 요원을 지정하세요.

    [사용자 질문]: {query}
    [사내 데이터 확보 여부]: {has_team_data}
    [외부 데이터 확보 여부]: {has_web_data}

    <요원 목록 및 조건>
    1. "search": 사내 기밀문서 검색이 필요한데 아직 안 했을 때
    2. "web_search": 최신 인터넷 뉴스가 필요한데 아직 안 했을 때
    3. "BOTH": 사내/외부 둘 다 필요한데 아직 안 했을 때
    4. "summary": 정보 수집이 끝나서 요약/통합이 필요할 때 (또는 검색을 마쳤을 때)
    5. "general": 정보 수집 없이 단순 답변만 하면 될 때 (잡담 등)
    6. "graph_memory": 예전 대화 맥락, 팀원의 역할, 과거의 결정 사항 등 '관계 추론'이 강력히 필요할 때

    반드시 위 요원 이름 중 하나만 정확히 출력하세요."""

    decision = external_llm.invoke(prompt).content.strip()
    print(f"🚦 [Supervisor 지시 사항]: {decision} 요원 출동!")

    # 상태에 supervisor의 결정을 저장 (그래프 라우팅용)
    return {"supervisor_decision": decision}


# Supervisor의 결정을 기반으로 실제로 길을 연결하는 함수
def route_from_supervisor(state: AgentState):
    decision = state.get("supervisor_decision", "general")
    if "BOTH" in decision:
        return ["search", "web_search"]
    elif "search" in decision:
        return "search"
    elif "web_search" in decision:
        return "web_search"
    elif "summary" in decision:
        return "summary"
    elif "graph_memory" in decision:
        return "graph_memory"
    else:
        return "external_llm"  # 일반 대화


def critic_router(state: AgentState) -> str:
    if state.get("feedback") == "PASS":
        return END
    return state.get("writer", "external_llm")


# --- 그래프 (파이프라인) 조립 ---
workflow = StateGraph(AgentState)
workflow.add_node("initialize", initialize_persona_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("supervisor", supervisor_node) # 💡 관리자 등록
workflow.add_node("search", search_node)
workflow.add_node("summary", summary_node)
workflow.add_node("commander", commander_node)
workflow.add_node("external_llm", external_llm_node)
workflow.add_node("custom_agent", custom_agent_node)
workflow.add_node("critic", critic_node)
workflow.add_node("web_search", web_search_node)  # 💡 [신규] 등록
workflow.add_node("graph_memory", graph_memory_node)

workflow.set_entry_point("initialize")

# 1. 초기화 후 단순 인사는 빼고 무조건 Supervisor에게 보고!
def init_route(state):
    if "안녕" in state["query"]: return "greeting"
    if state.get("target_agent_name"): return "custom_agent"
    return "supervisor"

workflow.add_conditional_edges("initialize", init_route, {
    "greeting": "greeting",
    "custom_agent": "custom_agent",
    "supervisor": "supervisor"
})

# 2. Supervisor가 지시한 곳으로 요원들 출동
workflow.add_conditional_edges("supervisor", route_from_supervisor, {
    "search": "search",
    "web_search": "web_search",
    "summary": "summary",
    "graph_memory": "graph_memory",
    "external_llm": "external_llm"
})

# 3. 검색 요원들은 임무가 끝나면 ➔ 다시 Supervisor에게 보고! (동적 제어)
# (단, LangGraph 병렬 처리 구조상 곧바로 summary로 넘기는 것이 안정적입니다)
workflow.add_edge("search", "summary")
workflow.add_edge("web_search", "summary")

# 4. 요약 끝 ➔ 커맨더 브리핑 작성 ➔ 검수자
workflow.add_edge("graph_memory", "summary")
workflow.add_edge("summary", "commander")
workflow.add_edge("commander", "critic")
workflow.add_edge("external_llm", "critic")
workflow.add_edge("custom_agent", "critic")

# 5. 검수 결과에 따라 무한 루프 또는 종료
workflow.add_conditional_edges("critic", critic_router, {
    "commander": "commander",
    "external_llm": "external_llm",
    "custom_agent": "custom_agent",
    END: END
})

workflow.add_edge("greeting", END)
langgraph_app = workflow.compile()