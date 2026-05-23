import os
import re
import requests
import asyncio
import json
import time
from typing import List, Optional, Union, Literal

from pydantic import BaseModel, Field

from ai_core.state import AgentState
from ai_core.prompts import *
from database.chroma_manager import collection
from database.postgres import get_user_persona
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchResults


# ==========================================
# 📊 대시보드 Pydantic 스키마
# ==========================================
class ChartItem(BaseModel):
    name: str = Field(description="항목명 (한국어)")
    value: Union[float, int, str] = Field(description="pie/bar/progress는 숫자, scorecard는 문자열 가능")
    color: Optional[str] = Field(None, description="hex 색상 (#6366f1 형식), pie/progress에 필요")

class Chart(BaseModel):
    id: str = Field(description="고유 ID (chart_1, chart_2 ...)")
    chartType: Literal["pie", "bar", "progress", "scorecard"]
    title: str = Field(description="차트 제목 (한국어)")
    data: List[ChartItem]

class DashboardData(BaseModel):
    title: str = Field(description="대시보드 제목 (한국어)")
    description: str = Field(description="한 줄 설명 (한국어)")
    charts: List[Chart] = Field(description="2~4개 차트")

class DashboardResponse(BaseModel):
    needed: bool = Field(description="이 답변에 대시보드가 유용한지 여부")
    data: Optional[DashboardData] = Field(None, description="needed=True일 때 대시보드 데이터")

# ==========================================
# 🛠️ 1. 무기 및 엔진 초기화
# ==========================================
model = SentenceTransformer('BAAI/bge-m3')
local_llm = ChatOllama(model="llama3", temperature=0, num_predict=1500)
external_llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", max_tokens=1500)


# ==========================================
# 🧩 2. 공통 유틸리티 (지휘관님 원본 복구)
# ==========================================
def _invoke_with_backoff(llm, prompt, max_retries=3):
    """Groq 413/rate_limit 에러 시 지수 백오프 재시도."""
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except Exception as e:
            err = str(e)
            if ("413" in err or "rate_limit_exceeded" in err.lower()) and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1초 → 2초 → 4초
                print(f"[Rate Limit] {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def get_dynamic_harness():
    base_harness = GLOBAL_HARNESS_MD
    if os.path.exists("added_rules.txt"):
        with open("added_rules.txt", "r", encoding="utf-8") as f:
            additional_rules = f.read()
        return base_harness + "\n[AI가 실시간으로 학습한 추가 보안 규정]\n" + additional_rules
    return base_harness


def format_history(history_list):
    if not history_list: return ""
    formatted = "\n[이전 대화 맥락]\n"
    for msg in history_list:
        is_user = msg.get("role") == "user"
        role = "지휘관" if is_user else "Dati(AI)"
        # 사용자 발언은 400자, AI 응답은 200자 (AI 응답이 훨씬 길어 토큰 폭발 원인)
        limit = 400 if is_user else 200
        content = (msg.get('content') or '').replace('[skip] ', '')[:limit]
        formatted += f"{role}: {content}\n"
    return formatted + "\n"


# ==========================================
# ❓ 3. 역질문 (Clarification) 유틸리티
# ==========================================

# 지시 대명사 / 모호 의문사 감지 → LLM 모호성 판단 트리거
_AMBIGUITY_TRIGGERS = [
    # 지시 대명사
    "이거", "그거", "저거", "이것", "그것", "이게", "그게", "저게",
    # 모호한 의문사
    "뭐가", "뭘", "뭔지",
    "어느 게", "어느게", "어느쪽", "어느 쪽",
    "어떤 게", "어떤게", "어떤거", "어떤 거",  # 추가
]
# 15자 이하의 짧은 쿼리에만 추가로 적용할 모호 동사 패턴
_VAGUE_SHORT_VERBS = [
    "어떻게 해", "추천해줘", "추천해 줘", "골라줘", "정해줘",
    "뭐가 좋", "뭐가 나",
    "어떻게 하면", "어떻게 할까",  # 추가
]


_SAFE_FROM_CLARIFICATION = ["안녕", "감사", "고마워", "반가워", "수고", "잘있어", "bye", "hello"]


def _needs_clarification_precheck(query: str) -> bool:
    if '[추가 정보:' in query or '[skip]' in query:  # 이미 보강됐거나 건너뛴 쿼리
        return False
    if any(s in query for s in _SAFE_FROM_CLARIFICATION):  # 인사/일상 대화는 제외
        return False
    if any(t in query for t in _AMBIGUITY_TRIGGERS):
        return True
    if len(query.strip()) <= 15 and any(v in query for v in _VAGUE_SHORT_VERBS):
        return True
    return False


def _check_and_generate_clarification(query: str, history_str: str) -> dict | None:
    """모호한 질문인지 LLM으로 판단. 모호하면 역질문+선택지 반환, 아니면 None."""
    json_llm = external_llm.bind(response_format={"type": "json_object"})
    prompt = f"""당신은 사용자 의도 파악 전문가입니다.
아래 질문이 모호하거나 추가 정보가 필요한지 판단하고, 모호하다면 구체적인 역질문과 선택지를 제시하세요.

이전 대화: {history_str[:300] if history_str else "없음"}
사용자 질문: {query}

판단 기준:
- 지시 대명사(이거, 그거 등)가 있고 맥락에서도 무엇을 가리키는지 불분명
- 질문이 너무 광범위하여 여러 해석이 가능

JSON 출력:
{{"needed": true, "question": "사용자에게 보여줄 한국어 역질문 (15자 이내)", "options": ["선택지A", "선택지B", "선택지C"], "multi_select": false}}
명확한 질문이면: {{"needed": false}}

선택지 필수 규칙:
- 반드시 서로 완전히 다른 카테고리/관점의 선택지여야 함 (유사/중복 절대 금지)
- 예: ["비용 절감 방안", "인력 운용 전략", "기술 스택 선택", "일정 단축 방법"] — 4가지 모두 다른 주제
- 나쁜 예: ["마케팅 전략", "마케팅 전략 도입", "새 마케팅 채널"] — 같은 주제 반복 금지
- 각 선택지는 10자 이내의 간결한 명사형 한국어
- 3~4개만 생성
"""
    try:
        result = json.loads(_invoke_with_backoff(json_llm, prompt).content)
        return result if result.get("needed") else None
    except Exception as e:
        print(f"[Clarification] 모호성 판단 실패: {e}")
        return None


# ==========================================
# 🧠 4. 라우팅 (Supervisor) — 시맨틱 임베딩 + LLM 하이브리드
# ==========================================
import numpy as np

# 각 에이전트를 대표하는 예시 문장 — 의미 기반 라우팅의 핵심
# 규칙: ① 에이전트 간 의미 대비가 명확할 것 ② 실제 사용자 표현 다양성 커버
_AGENT_EXAMPLES = {
    "expert_agent": [
        # 기술/언어 선택 비교 (어느 게 나은지 — coding이 아닌 의사결정)
        "React랑 Vue 중에 뭐가 나아?",
        "Python이랑 JavaScript 어느 게 더 배우기 좋아?",
        "Django랑 FastAPI 어느 걸 선택해야 해?",
        "SQL이랑 NoSQL 중 우리 서비스에 뭐가 맞을까?",
        "AWS랑 GCP 중 어디가 더 좋아?",
        "어떤 프레임워크를 써야 할지 모르겠어",
        # 전략/계획/로드맵
        "스타트업 1년 로드맵 어떻게 설계해야 해?",
        "마케팅 전략 A안 B안 비교 분석해줘",
        "팀 우선순위 어떻게 정해야 할까?",
        "프로젝트 일정 어떻게 짜야 해?",
        # 아키텍처/기술 의사결정
        "마이크로서비스 vs 모놀리식 아키텍처 장단점 알려줘",
        "백엔드 프레임워크 추천해줘",
        "기술 스택 선택 기준이 뭐야?",
        "서비스 성능 개선 방향을 잡아줘",
        "데이터베이스 설계 어떻게 접근해야 해?",
        # 비즈니스 분석
        "경쟁사 대비 우리 제품 차별화 전략이 뭐야?",
        "신규 기능 도입 시 고려할 점이 뭐야?",
    ],
    "coding_math_agent": [
        # 자료구조 조작 (coding으로 가야 하는 핵심 패턴)
        "파이썬 리스트에서 특정 값의 인덱스 찾는 방법",
        "딕셔너리를 키 기준으로 정렬하는 코드 짜줘",
        "리스트 중복 제거하는 법 알려줘",
        "2차원 배열 반복문으로 순회하는 코드 써줘",
        "튜플과 리스트 차이점이랑 사용법 알려줘",
        "슬라이싱으로 리스트 역순 만드는 법",
        # 디버깅/에러 해결
        "TypeError: NoneType object is not subscriptable 에러 어떻게 고쳐?",
        "이 코드 실행하면 인덱스 에러 나는데 봐줘",
        "함수 반환값이 None으로 나오는 이유가 뭐야?",
        "무한 루프 빠져나오는 방법",
        "AttributeError 원인 찾아줘",
        # 코드 작성 요청
        "피보나치 수열 재귀 함수로 구현해줘",
        "버블 정렬 알고리즘 코드로 짜줘",
        "로그인 JWT 인증 API 만들어줘",
        "SQL INNER JOIN 쿼리 예제 작성해줘",
        "JavaScript 비동기 함수 예제 코드 보여줘",
        # 수학/알고리즘
        "빅오 표기법으로 시간복잡도 분석해줘",
        "정규분포 표준편차 계산 방법 알려줘",
        "확률 계산해줘",
        "재귀와 반복문 시간복잡도 차이",
    ],
    "general_agent": [
        # 인사/일상
        "안녕!", "안녕하세요", "반가워요", "처음 만나요",
        "오늘 기분 어때?", "요즘 어때?", "잘 지내?",
        "수고했어", "고마워", "감사합니다", "도움이 됐어",
        "잘있어", "다음에 또 봐", "bye",
        "심심한데 얘기 좀 해줘", "잠깐 대화하자",
        # 대화 기억/맥락 질문 — 이게 expert/coding으로 가면 엉뚱한 답 나옴
        "방금 내가 뭐 물어봤지?",
        "이전에 어떤 질문을 했었어?",
        "아까 말한 게 뭐야?",
        "우리 지금까지 무슨 얘기 했어?",
        "내 마지막 질문이 뭐였지?",
        "기억해?",
        "내가 뭐라고 했는지 알아?",
        "방금 전 대화 내용이 뭐야?",
    ],
}

# 서버 시작 시 1회 임베딩 캐싱
_AGENT_EXAMPLE_EMBEDDINGS: dict = {}

def _init_semantic_router():
    print("[SemanticRouter] 라우팅 임베딩 사전 캐싱 중...")
    for agent, examples in _AGENT_EXAMPLES.items():
        vecs = model.encode(examples, normalize_embeddings=True)
        _AGENT_EXAMPLE_EMBEDDINGS[agent] = np.array(vecs)
    print("[SemanticRouter] 캐싱 완료 ✓")

_init_semantic_router()

_SEMANTIC_THRESHOLD = 0.60  # 코사인 유사도 기준 (초과 시 LLM 생략)


def _semantic_score(query: str) -> dict:
    """쿼리와 각 에이전트 예시 문장의 최대 코사인 유사도를 반환."""
    query_vec = model.encode(query, normalize_embeddings=True)
    scores = {}
    for agent, example_vecs in _AGENT_EXAMPLE_EMBEDDINGS.items():
        # 정규화된 벡터끼리의 내적 = 코사인 유사도
        sims = example_vecs @ query_vec
        scores[agent] = float(np.max(sims))
    return scores


def supervisor_node(state):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    print("\n[Supervisor] 시맨틱 라우팅으로 부서 분석 중...")

    # --- 0단계: 지시 대명사 감지 → 역질문 트리거 ---
    if _needs_clarification_precheck(query):
        print(f"[Supervisor] 지시 대명사 감지 → 모호성 LLM 판단 중...")
        clarify = _check_and_generate_clarification(query, history_str)
        if clarify:
            print(f"[Supervisor] 역질문 트리거: {clarify.get('question', '')}")
            return {
                "target_agent_name": "clarify",
                "need_clarification": True,
                "clarify_question": clarify.get("question", "무엇을 도와드릴까요?"),
                "clarify_options": clarify.get("options", []),
                "clarify_multi_select": clarify.get("multi_select", False),
                "fallback_mode": False,
            }

    # --- 1단계: 시맨틱 스코어링 (임베딩 코사인 유사도) ---
    scores = _semantic_score(query)
    best_agent = max(scores, key=scores.get)
    best_score = scores[best_agent]
    print(f"[Semantic] 유사도: { {k: round(v,2) for k,v in scores.items()} }")

    if best_score >= _SEMANTIC_THRESHOLD:
        print(f"[Semantic] 신뢰도 충분 ({best_score:.2f}) → {best_agent} 직행 (LLM 생략)")
        return {"target_agent_name": best_agent, "fallback_mode": False}

    # --- 2단계: 유사도가 낮으면 LLM에게 최종 판단 위임 ---
    print(f"[Semantic] 신뢰도 부족 ({best_score:.2f}) → LLM 판단 요청...")
    prompt = f"""당신은 Dativus 시스템의 라우팅 관리자입니다.
사용자 질문을 보고 가장 적합한 부서를 단 1개만 출력하세요.

[부서]
- general_agent: 단순 인사, 안부, 짧은 일상 대화
- expert_agent: 목표설정, 계획수립, 기술비교, 전략분석, 추천, 사내문서 관련
- coding_math_agent: 코드작성, 에러수정, 수학계산

[예시]
"이 프로젝트 목표를 정해줘" → expert_agent
"React vs Vue 비교해줘" → expert_agent
"Python이랑 JavaScript 중 뭐 배우는 게 나아?" → expert_agent
"파이썬 리스트에서 인덱스 찾는 법" → coding_math_agent
"파이썬 에러 고쳐줘" → coding_math_agent
"피보나치 수열 짜줘" → coding_math_agent
"안녕하세요" → general_agent
"방금 내가 뭐 물어봤지?" → general_agent
"기억해?" → general_agent

질문: {query}
출력(부서 이름만):"""

    try:
        decision = external_llm.invoke(prompt).content.strip().lower()
        if decision not in ["general_agent", "expert_agent", "coding_math_agent"]:
            decision = best_agent if best_score > 0 else "general_agent"
        print(f"[Supervisor] LLM 결정: {decision}")
        return {"target_agent_name": decision, "fallback_mode": False}
    except Exception as e:
        print(f"[Supervisor] 외부 LLM 실패: {e} → 로컬 폴백")
        try:
            decision = local_llm.invoke(prompt).content.strip().lower()
            if decision not in ["general_agent", "expert_agent", "coding_math_agent"]:
                decision = best_agent if best_score > 0 else "general_agent"
            return {"target_agent_name": decision, "fallback_mode": True}
        except Exception:
            return {"target_agent_name": "general_agent", "fallback_mode": True}


# ==========================================
# 🔍 4. 수색대 (Memory Workers) - 💡 실제 로직 탑재!
# ==========================================
def search_node(state: AgentState):
    query = state["query"]
    workspace_id = state.get("workspace_id")
    print(f"🔍 [Vector Search] 사내망(ChromaDB)에서 '{query}' 관련 정보를 찾습니다...")

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2,  # 관련도 높은 2개만 추출하여 토큰 절약
        where={"workspace_id": workspace_id} if workspace_id else None
    )

    context_result = results['documents'][0] if results['documents'] else ["관련 문서를 찾을 수 없습니다."]
    return {"search_context": "\n".join(context_result)}


def web_search_node(state: AgentState):
    query = state["query"]
    print(f"🌐 [Web Search] DuckDuckGo를 통해 '{query}' 최신 정보를 수집합니다...")

    search_tool = DuckDuckGoSearchResults(num_results=2)
    search_result = search_tool.invoke(query)
    return {"web_context": search_result}


def graph_memory_node(state: AgentState):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    # 전처리 단계이므로 항상 로컬 Ollama 사용 → Groq TPM 절약
    active_llm = local_llm

    print("🕸️ [GraphRAG] 대화 맥락과 유기적 관계(Entity-Relation)를 분석합니다...")

    prompt = f"""당신은 '지식 그래프(Knowledge Graph) 분석가'입니다.
    아래 대화 기록에서 질문에 답하기 위해 필요한 '핵심 개체(Entity)'와 '관계(Relation)'를 추출하세요.
    형식: [개체A] ➔ (관계) ➔ [개체B]

    [이전 대화 맥락]: {history_str}
    [사용자 질문]: {query}
    """
    graph_memory = active_llm.invoke(prompt).content
    return {"graph_context": graph_memory}


# ==========================================
# 👔 5. 추론 및 생성 부서 (Logic Workers) - 💡 동적 하네스 탑재!
# ==========================================
def expert_agent_node(state: AgentState):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else external_llm

    search_ctx = (state.get("search_context", "") or "")[:500]
    web_ctx = (state.get("web_context", "") or "")[:300]
    graph_ctx = (state.get("graph_context", "") or "")[:300]

    print("👔 [전문 대화병] 수집된 입체적 기억(Graph + Vector + Web)을 융합 추론합니다.")

    # [추가 정보:] 태그가 있으면 역질문 후 보강된 쿼리 — 태그 내용을 핵심 주제로 우선 처리
    clarify_hint = ""
    if '[추가 정보:' in query:
        import re as _re
        tags = _re.findall(r'\[추가 정보:\s*([^\]]+)\]', query)
        if tags:
            clarify_hint = f"\n[역질문 선택 결과 - 반드시 이 주제를 중심으로 답변]: {', '.join(tags)}\n"

    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
    {EXPERT_OUTPUT_FORMAT}
    당신은 최고 전문가(Logic Worker) 요원입니다.
    수집된 아래의 정보를 바탕으로 맥락을 연결(Reasoning)하여 완벽한 답변을 작성하세요.
    ⚠️ 반드시 [사용자 질문]에만 답하세요. 이전 대화는 맥락 참고용이며, 재생성 금지.
    ⚠️ [사내 문서]가 현재 질문과 관련이 없으면 완전히 무시하고 일반 지식으로 답하세요.
    {clarify_hint}
    {history_str}
    [사내 문서(VectorRAG)]: {search_ctx}
    [관계망 기억(GraphRAG)]: {graph_ctx}
    [외부 웹 정보]: {web_ctx}

    [사용자 질문]: {query}
    """
    response = _invoke_with_backoff(active_llm, prompt).content
    return {"draft_answer": response}


def general_agent_node(state: AgentState):
    query = state["query"]
    history_list = state.get("history", [])
    history_str = format_history(history_list)
    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else external_llm

    print(f"🗣️ [일반 대화병] 히스토리 수신: {len(history_list)}개 메시지")

    history_section = f"참고할 이전 대화:\n{history_str}" if history_str else ""

    # 이전 대화 기억 질문 여부 판단
    memory_keywords = ["방금", "기억", "아까", "이전에", "뭐 물어", "뭐라고", "말했잖", "어떤 질문"]
    is_memory_query = any(kw in query for kw in memory_keywords)
    memory_instruction = (
        "⚠️ 사용자가 이전 대화 내용을 묻고 있습니다. "
        "반드시 위의 [참고할 이전 대화]에서 실제 질문/답변 내용을 찾아 구체적으로 답하세요. "
        "'기록하세요', '검색하세요' 같은 일반 조언은 절대 하지 마세요."
    ) if is_memory_query and history_str else ""

    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
    {GENERAL_OUTPUT_FORMAT}
    당신은 친절한 대화 요원입니다.
    절대 '[이전 대화 맥락]', '[사용자 질문]', '참고할 이전 대화' 같은 내부 섹션 헤더를 답변에 포함하지 마세요.
    {history_section}
    {memory_instruction}
    사용자 질문: {query}
    """
    response = _invoke_with_backoff(active_llm, prompt).content
    return {"draft_answer": response}


def coding_math_agent_node(state: AgentState):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else external_llm

    clarify_hint = ""
    if '[추가 정보:' in query:
        import re as _re
        tags = _re.findall(r'\[추가 정보:\s*([^\]]+)\]', query)
        if tags:
            clarify_hint = f"\n[역질문 선택 결과 - 반드시 이 주제를 중심으로 답변]: {', '.join(tags)}\n"

    print("💻 [코딩/수학 대화병] 알고리즘 분석 및 수학 연산을 수행합니다.")

    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
    {CODING_OUTPUT_FORMAT}
    당신은 시니어 개발자 수준의 프로그래밍 및 수학 전문가입니다.
    ⚠️ 반드시 [사용자 질문]에만 답하세요. 이전 대화는 맥락 참고용이며, 이전 대화 내용을 그대로 반복하거나 재생성하지 마세요.
    {clarify_hint}
    {history_str}
    [사용자 질문]: {query}
    """
    response = _invoke_with_backoff(active_llm, prompt).content
    return {"draft_answer": response}


# ==========================================
# 📊 6. 대시보드 및 융합
# ==========================================
_DASHBOARD_KEYWORDS = ['목표', '계획', '로드맵', '비교', '분석', '추천', '장단점', '진행', '현황', '지표', '일정', '단계', '기술', '방법', '옵션']


def _clean_json_from_draft(text: str) -> str:
    # <DASHBOARD> 태그 제거
    text = re.sub(r'<DASHBOARD>[\s\S]*?</DASHBOARD>', '', text)
    # "대시보드 JSON 블록" 헤더부터 코드 블록 끝까지 제거
    text = re.sub(r'\*?대시보드 JSON 블록\*?[\s\S]*?```[\s\S]*?```', '', text, flags=re.IGNORECASE)
    # charts 키를 포함한 JSON 코드 블록 제거
    text = re.sub(r'```(?:json)?\s*\{[\s\S]*?"charts"[\s\S]*?\}\s*```', '', text)
    return text.strip()


def _merge_dashboards(existing: dict, new: dict) -> dict:
    """기존 캔버스에 새 차트를 병합. 제목이 같으면 갱신, 다르면 추가. 최대 6개."""
    if not existing or not existing.get("charts"):
        return new
    existing_charts = {c["title"]: c for c in existing.get("charts", [])}
    for chart in new.get("charts", []):
        existing_charts[chart["title"]] = chart  # 같은 제목이면 덮어쓰기, 새 제목이면 추가
    merged = dict(new)
    merged["charts"] = list(existing_charts.values())[:6]
    return merged


def dashboard_select_node(state: AgentState):
    draft = state.get("draft_answer", "")
    query = state.get("query", "")
    history_str = format_history(state.get("history", []))
    search_ctx = state.get("search_context", "")
    existing_dashboard = state.get("existing_dashboard", {})

    # --- 1단계: <DASHBOARD> 태그 기반 파싱 ---
    s, e = draft.find("<DASHBOARD>"), draft.find("</DASHBOARD>")
    if s != -1 and e != -1:
        try:
            data = json.loads(draft[s + 11:e].strip())
            if data.get("charts"):
                clean = (draft[:s] + draft[e + 12:]).strip()
                merged = _merge_dashboards(existing_dashboard, data)
                print(f"[Dashboard] 태그 파싱 성공: {merged.get('title', '')}")
                return {"dashboard_data": merged, "draft_answer": clean}
        except json.JSONDecodeError:
            pass

    # 태그가 없어도 draft에 JSON 코드 블록이 있을 수 있으므로 미리 제거
    clean_draft = _clean_json_from_draft(draft)

    # --- 2단계: 키워드 체크 ---
    if not any(kw in query for kw in _DASHBOARD_KEYWORDS):
        return {"dashboard_data": {}, "draft_answer": clean_draft}

    # --- 3단계: Groq JSON Mode로 대시보드 생성 ---
    print("[Dashboard] JSON Mode로 대시보드 생성 중...")
    try:
        json_llm = external_llm.bind(response_format={"type": "json_object"})
        existing_titles = [c.get("title", "") for c in existing_dashboard.get("charts", [])]
        existing_hint = f"Existing chart titles (reuse these EXACT titles to update values): {existing_titles}" if existing_titles else "No existing charts yet."

        prompt = f"""You are a dashboard JSON generator. Output ONLY valid JSON, nothing else.

Question: {query}
Conversation History: {history_str}
Uploaded Documents: {search_ctx[:600]}
Context: {draft[:400]}
{existing_hint}

If a visual dashboard is useful, output:
{{"needed": true, "title": "한국어 제목", "description": "한국어 설명", "charts": [
  {{"id": "chart_1", "chartType": "pie", "title": "차트 제목", "data": [
    {{"name": "항목명", "value": 40, "color": "#6366f1"}},
    {{"name": "항목명", "value": 35, "color": "#8b5cf6"}}
  ]}},
  {{"id": "chart_2", "chartType": "scorecard", "title": "지표 제목", "data": [
    {{"name": "지표명", "value": "3개월"}}
  ]}}
]}}

If NOT useful (greetings, simple chat, code debug): {{"needed": false}}

Rules:
- pie/progress: value=number(0-100), include color hex
- bar: value=number, no color
- scorecard: value=string like "3개월","5명","1500만원"
- All titles and names in Korean, 2-4 charts max

JSON:"""

        raw = json_llm.invoke(prompt)
        result = json.loads(raw.content)

        if result.get("needed") and result.get("charts"):
            dashboard_data = {k: v for k, v in result.items() if k != "needed"}
            merged = _merge_dashboards(existing_dashboard, dashboard_data)
            print(f"[Dashboard] JSON Mode 성공: {merged.get('title', '')} (차트 {len(merged.get('charts', []))}개)")
            return {"dashboard_data": merged, "draft_answer": clean_draft}

    except Exception as ex:
        print(f"[Dashboard] JSON Mode 실패: {ex}")

    return {"dashboard_data": {}, "draft_answer": clean_draft}


def summary_node(state: AgentState):
    print("📝 [Summary] 대시보드 세팅과 답변을 통합하여 검수(Critic) 부서로 이관합니다.")
    return {}  # State 자동 갱신


# ==========================================
# ⚖️ 7. 검수 (Critic) 및 수정
# ==========================================
def critic_node(state: AgentState):
    draft_answer = state.get("draft_answer", "")
    count = state.get("revision_count", 0)

    if count >= 1:
        print("🚨 [Critic] 수정 한도 도달. 시스템 과부하 방지를 위해 강제 PASS!")
        return {"final_answer": draft_answer, "critic_feedback": "PASS"}

    print(f"🧐 [Critic] 로컬(Ollama) 엔진으로 품질 검사 중... (현재 수정: {count}회)")

    # 하드코딩 삭제하고 prompts.py의 변수 사용
    prompt = CRITIC_SYSTEM_PROMPT.format(query=state.get("query", ""), draft=draft_answer)

    try:
        decision = local_llm.invoke(prompt).content.strip().upper()
        if "PASS" in decision:
            print("✅ [Critic] 검수 통과!")
            return {"final_answer": draft_answer, "critic_feedback": "PASS"}
        else:
            print(f"❌ [Critic] 검수 반려! 사유: {decision}")
            return {"critic_feedback": decision, "revision_count": count + 1}
    except Exception as e:
        print(f"⚠️ [Critic] 에러 발생, 비상 통과 처리: {e}")
        return {"final_answer": draft_answer, "critic_feedback": "PASS"}


def revision_agent_node(state: AgentState):
    query = state["query"]
    draft = state.get("draft_answer", "")
    feedback = state.get("critic_feedback", "")
    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else external_llm

    print("🛠️ [개선/보충 대화병] Critic의 지적을 반영하여 초안을 긴급 수정합니다.")

    # 다시 작성할 때도 하네스 룰 강제 주입
    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
    당신은 답변 수정 요원입니다.
    🚨 [지적사항]: {feedback}
    [사용자 질문]: {query}
    위 지적사항만 반영해서 답변을 새로 작성하세요.
    JSON, 코드 블록, 대시보드, 수치 목록은 절대 포함하지 마세요. (시스템이 자동 처리)
    서론("수정된 답변:", "아래와 같이" 등) 없이 바로 본문만 출력하세요."""

    response = active_llm.invoke(prompt).content
    return {"draft_answer": response}


def check_critic_approval(state: AgentState):
    if "PASS" in state.get("critic_feedback", ""):
        return "end"
    return "revision"


# ==========================================
# ❓ 역질문 종착 노드 (Clarify Terminal)
# ==========================================
def clarify_node(state: AgentState):
    """역질문 데이터를 그대로 출력 — main.py가 [CLARIFY] SSE 이벤트로 변환."""
    return {
        "need_clarification": True,
        "clarify_question": state.get("clarify_question", ""),
        "clarify_options": state.get("clarify_options", []),
        "clarify_multi_select": state.get("clarify_multi_select", False),
    }