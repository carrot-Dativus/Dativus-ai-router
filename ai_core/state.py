from typing import TypedDict, Annotated, Sequence
import operator
from langgraph.graph.message import add_messages


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
    # 🔍 3-1. 검색 인텔리전스 (Phase 1)
    # ==========================================
    query_rewritten: str   # 검색 최적화된 쿼리 (query_rewriter_node 출력)
    sub_queries: list      # 멀티홉 분해 서브쿼리 ["서브1", "서브2", ...]
    search_plan: list      # 활성화할 검색 레인 ["vector", "web", "graph"]
    search_attempts: int   # 재검색 시도 횟수 (루프 가드, 최대 2)
    search_grade: str      # "sufficient" | "rewrite"
    tool_calls_history: list  # ReAct 도구 호출 이력 [{tool, args}, ...]

    # ==========================================
    # 🤖 3-2. ReAct 메시지 (expert_agent 전용)
    # ==========================================
    messages: Annotated[list, add_messages]  # bind_tools 루프용 메시지 누적

    # ==========================================
    # ⚖️ 5. 검수(Critic) 및 무한 루프 제어
    # ==========================================
    draft_answer: str  # 부서에서 작성한 최초 답변 초안
    critic_feedback: str  # Critic JSON 피드백 {"pass": bool, "reasons": [], "fix_targets": []}
    revision_count: int  # 무한 루프 방지용 카운터 (2회 제한)
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
    custom_agent_type: str          # 수동 선택된 에이전트 엔진 (LOCAL / EXTERNAL_API)
    custom_agents_list: list        # 자동 매칭용 전체 에이전트 목록 [{name, description}, ...]
    matched_custom_agent_name: str  # 실제 호출된 에이전트 이름 (대시보드 표시용)
    multi_agent_responses: list     # 다중 매칭 시 각 에이전트 응답 [{name, response}, ...]

    # ==========================================
    # 🧬 8. 사용자 개인화 (Personalization) — Phase 1: 구조화 + 자유 입력형
    # ==========================================
    # Phase 1 (현재): 마이페이지 드롭다운 3개 + 자유 입력 메모 → 각 에이전트 프롬프트에 스타일 참고로 주입
    # TODO Phase 2: 피드백(👍/👎) 누적 데이터 자동 분석 → 개인화 패턴을 학습하여
    #   사용자별 맞춤 대시보드로 시각화 (자동학습형 개인화, 별도 백그라운드 파이프라인 필요)
    persona_expertise: str       # 전문 분야: "기본", "프론트엔드", "백엔드", "데이터 엔지니어"
    persona_tone: str            # 대화 어조: "친절한", "단호하고 전문적인", "사극 이순신 장군"
    persona_decision_style: str  # 판단 스타일: "일반적인", "논리적인", "직관적인"
    persona_memo: str            # 추가 자유 입력 지시문 (구조 변경 불가 주의 안내 포함)

