from typing import TypedDict, Annotated, Sequence
import operator


class AgentState(TypedDict):
    # ==========================================
    # 📡 1. 사용자 입력 (User Input)
    # ==========================================
    query: str  # 사용자의 현재 질문
    workspace_id: str  # 보안 격리용 팀 ID
    user_id: str  # 작성자 ID
    history: list  # 이전 대화 맥락 (선택 사항)

    # ==========================================
    # 🧠 2. 핵심 라우팅 및 3분할 제어
    # ==========================================
    target_agent_name: str  # Supervisor가 배정한 부서 (general_agent, expert_agent, coding_math_agent)
    fallback_mode: bool  # 🚨 외부 API(Groq) 토큰 소진 시 True로 변환 (전면 로컬 모드 발동)

    # ==========================================
    # 🗂️ 3. 데이터 수집함 (전문 대화병 / 코딩 대화병 전용)
    # ==========================================
    search_context: str  # 사내 DB(ChromaDB)에서 긁어온 데이터
    web_context: str  # 외부 웹에서 긁어온 최신 데이터
    graph_context: str  # 과거 대화/관계망 메모리

    # ==========================================
    # 📊 4. 융합 및 대시보드 (Summary & Dashboard)
    # ==========================================
    existing_dashboard: dict  # 프론트엔드가 보낸 현재 캔버스 데이터 (병합용)
    dashboard_data: dict  # 프론트엔드 대시보드 렌더링용 JSON 데이터

    # ==========================================
    # ⚖️ 5. 검수(Critic) 및 무한 루프 제어
    # ==========================================
    draft_answer: str  # 부서에서 작성한 최초 답변 초안
    critic_feedback: str  # 로컬 LLM(Critic)이 남긴 지적/반려 사유
    revision_count: int  # 무한 루프 방지용 카운터 (1회 제한)
    final_answer: str  # 프론트엔드로 쏠 최종 통과 답변

    # ==========================================
    # ❓ 6. 역질문 (Clarification)
    # ==========================================
    need_clarification: bool    # 역질문 필요 여부
    clarify_question: str       # 사용자에게 보여줄 역질문
    clarify_options: list       # 선택지 목록
    clarify_multi_select: bool  # 다중 선택 허용 여부