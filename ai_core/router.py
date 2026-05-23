from langgraph.graph import StateGraph, END
from ai_core.state import AgentState
# (만약 파일이 나뉘어 있다면 아래처럼 노드 함수들을 import 해야 합니다)
from ai_core.nodes import *;

# 1. 뼈대 생성 (작전 서류철 규격 지정)
workflow = StateGraph(AgentState)

# 1. 노드 배치 (수색대 3인방 추가)
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("clarify", clarify_node)          # 역질문 종착 노드
workflow.add_node("general_agent", general_agent_node)
workflow.add_node("coding_math_agent", coding_math_agent_node)

workflow.add_node("search", search_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("graph_memory", graph_memory_node)
workflow.add_node("expert_agent", expert_agent_node) # Logic Worker 역할

# (이후 대시보드 및 검수 노드 생략 - 이전과 동일하게 add_node 처리)
workflow.add_node("dashboard_select", dashboard_select_node)
workflow.add_node("summary", summary_node)
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
        "general_agent": "general_agent",
        "expert_agent": "search",
        "coding_math_agent": "coding_math_agent",
        "clarify": "clarify",                 # 역질문 → 즉시 종착
    }
)

workflow.add_edge("clarify", END)             # 역질문은 에이전트 파이프라인 건너뜀

# 3. 수색대의 릴레이 정보 수집 (순차적 실행)
workflow.add_edge("search", "web_search")
workflow.add_edge("web_search", "graph_memory")

# 4. 수집 완료 ➔ Logic Worker(전문 대화병)에게 서류 전달하여 초안 작성
workflow.add_edge("graph_memory", "expert_agent")

# 5. 모든 부서의 일이 끝나면 ➔ 대시보드로 집결
workflow.add_edge("general_agent", "dashboard_select")
workflow.add_edge("expert_agent", "dashboard_select")
workflow.add_edge("coding_math_agent", "dashboard_select")

# (이하 대시보드 ➔ Summary ➔ Critic ➔ END 로직은 이전과 완벽히 동일)
workflow.add_edge("dashboard_select", "summary")
workflow.add_edge("summary", "critic")
workflow.add_conditional_edges("critic", check_critic_approval, {"end": END, "revision": "revision_agent"})
workflow.add_edge("revision_agent", "dashboard_select")

langgraph_app = workflow.compile()