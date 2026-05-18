# ai_core/state.py
from typing import TypedDict, Optional, List, Dict

class AgentState(TypedDict):
    query: str
    workspace_id: Optional[str]
    user_id: Optional[str]
    persona: Optional[dict]
    context: Optional[str]
    team_context: str  # 💡 [신규] 사내 검색병 전용 서류철
    web_context: str  # 💡 [신규] 외부 검색병 전용 서류철
    summary: Optional[str]
    history: Optional[List[Dict[str, str]]]
    final_answer: str
    target_agent_name: Optional[str]
    target_agent_prompt: Optional[str]

    # 💡 [Phase 2 신규 추가] 요원 간 협업 및 피드백을 위한 칸
    writer: Optional[str]  # 초안을 작성한 요원 이름 (commander, external_llm 등)
    feedback: Optional[str]  # 검수 요원의 반려 사유 (합격이면 'PASS')
    revision_count: int  # 무한 루프 방지용 수정 횟수