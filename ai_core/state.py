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
    force_agent: str       # 사용자가 수동 선택한 부서 (있으면 supervisor 스킵)
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

    # ==========================================
    # 🎭 7. 커스텀 에이전트 (Custom Ego)
    # ==========================================
    custom_agent_name: str          # 수동 선택된 에이전트 이름
    custom_agent_prompt: str        # 수동 선택된 에이전트 성격/역할
    custom_agents_list: list        # 자동 매칭용 전체 에이전트 목록 [{name, description}, ...]
    matched_custom_agent_name: str  # 실제 호출된 에이전트 이름 (대시보드 표시용)
    multi_agent_responses: list     # 다중 매칭 시 각 에이전트 응답 [{name, response}, ...]

    # ==========================================
    # 🧬 8. 사용자 개인화 (Personalization) — Phase 1: 자유 입력형
    # ==========================================
    # Phase 1 (현재): 사용자가 직접 자연어로 AI 성향을 지시 → GLOBAL_HARNESS에 주입
    # TODO Phase 2: 피드백(👍/👎) 누적 데이터 자동 분석 → 개인화 패턴을 학습하여
    #   사용자별 맞춤 대시보드로 시각화 (자동학습형 개인화, 별도 백그라운드 파이프라인 필요)
    persona_memo: str  # 사용자가 마이페이지에서 입력한 자유 형식 AI 지시문

