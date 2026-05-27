from langgraph.graph import StateGraph, END
from ai_core.state import AgentState
# (만약 파일이 나뉘어 있다면 아래처럼 노드 함수들을 import 해야 합니다)
from ai_core.nodes import *;

# 1. 뼈대 생성 (작전 서류철 규격 지정)
workflow = StateGraph(AgentState)

# 1. 노드 배치 (수색대 3인방 추가)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("clarify", clarify_node)
workflow.add_node("conversation_memory", conversation_memory_node)  # 일반 대화팀 메모리
workflow.add_node("general_agent", general_agent_node)
workflow.add_node("code_search", code_search_node)                  # 코딩팀 레퍼런스 검색
workflow.add_node("coding_math_agent", coding_math_agent_node)

workflow.add_node("search_coordinator", search_coordinator_node)
workflow.add_node("search", search_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("graph_memory", graph_memory_node)
workflow.add_node("expert_agent", expert_agent_node) # Logic Worker 역할

# (이후 대시보드 및 검수 노드 생략 - 이전과 동일하게 add_node 처리)
workflow.add_node("custom_agent_gate", custom_agent_gate_node)
workflow.add_node("dashboard_select", dashboard_select_node)
workflow.add_node("summary", summary_node)
workflow.add_node("persona_agent", persona_agent_node)
workflow.add_node("critic", critic_node)
workflow.add_node("revision_agent", revision_agent_node)

workflow.set_entry_point("supervisor")

# 2. Supervisor의 분기점 설정
def route_from_supervisor(state):
    return state["target_agent_name"]

# 💡 전문 부서(expert_agent)가 선택되면, 답변 작성이 아니라 '검색(search)'부터 시작하도록 라우팅!
workflow.add_conditional_edges(
    "supervisor",
    route_from_supervisor,
    {
        "general_agent": "conversation_memory",  # 일반: 기억 확인 → 대화
        "expert_agent": "search_coordinator",    # 전문: 병렬 수색 팬아웃
        "coding_math_agent": "code_search",      # 코딩: 레퍼런스 검색 → 코딩
        "clarify": "clarify",
    }
)

workflow.add_edge("clarify", END)
workflow.add_edge("conversation_memory", "general_agent")
workflow.add_edge("code_search", "coding_math_agent")

# 3. 수색대 병렬 실행 — search_coordinator → 3개 노드 동시 실행 → expert_agent 합류
workflow.add_edge("search_coordinator", "search")
workflow.add_edge("search_coordinator", "web_search")
workflow.add_edge("search_coordinator", "graph_memory")

# 4. 3개 수색 노드 모두 완료 후 ➔ Logic Worker(전문 대화병)에게 서류 전달
workflow.add_edge("search", "expert_agent")
workflow.add_edge("web_search", "expert_agent")
workflow.add_edge("graph_memory", "expert_agent")

# 5. 모든 부서의 일이 끝나면 ➔ 커스텀 에이전트 게이트(패스스루 또는 추가) ➔ 대시보드
workflow.add_edge("general_agent", "custom_agent_gate")
workflow.add_edge("expert_agent", "custom_agent_gate")
workflow.add_edge("coding_math_agent", "custom_agent_gate")
workflow.add_edge("custom_agent_gate", "dashboard_select")

# (이하 대시보드 ➔ Summary ➔ Critic ➔ END 로직은 이전과 완벽히 동일)
workflow.add_edge("dashboard_select", "summary")
workflow.add_edge("summary", "persona_agent")
workflow.add_edge("persona_agent", "critic")
workflow.add_conditional_edges("critic", check_critic_approval, {"end": END, "revision": "revision_agent"})
workflow.add_edge("revision_agent", "dashboard_select")

langgraph_app = workflow.compile()