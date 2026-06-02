import os
import re
import requests
import asyncio
import json
import time
import threading
from typing import List, Optional, Union, Literal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from pydantic import BaseModel, Field

from ai_core.state import AgentState
from ai_core.prompts import *
from ai_core import metrics as _metrics
from database.chroma_manager import collection
from database.postgres import get_user_persona
from database.graph_store import save_triples, load_context as graph_load_context
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
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
local_llm = ChatOllama(model="qwen2.5:14b", temperature=0, num_predict=1500)
# 라우팅 · Critic · 역질문 · 대시보드 — 속도 우선
external_llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant", max_tokens=1500)
# 전문 분석 — 품질 우선 (70B)
expert_llm   = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", max_tokens=2000)
# 코딩/수학 · 재작성 — 정밀도 우선 (70B)
coding_llm   = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile", max_tokens=3000)


# ==========================================
# 🔧 ReAct 도구 정의 (expert_agent 전용)
# ==========================================

# tools에서 workspace_id를 읽기 위한 thread-local 컨텍스트
_tool_ctx = threading.local()

@tool
def rag_search(query: str) -> str:
    """Search the internal team knowledge base (ChromaDB) for company documents, uploaded files, or past decisions. Use when the question involves internal/company-specific information."""
    workspace_id = getattr(_tool_ctx, "workspace_id", None)
    try:
        emb = model.encode(query).tolist()
        r = collection.query(
            query_embeddings=[emb],
            n_results=3,
            where={"workspace_id": workspace_id} if workspace_id else None,
        )
        docs = r["documents"][0] if r["documents"] else []
        result = "\n".join(docs) if docs else "관련 문서를 찾을 수 없습니다."
        print(f"[Tool:rag_search] '{query[:30]}' → {len(docs)}건")
        return result
    except Exception as e:
        return f"검색 실패: {e}"


@tool
def web_search_tool(query: str) -> str:
    """Search the web for recent, external, or up-to-date information using DuckDuckGo. Use when the question needs current data, trends, or information not available internally."""
    try:
        search = DuckDuckGoSearchResults(num_results=3)
        raw = search.invoke(query)
        snippets = re.findall(r"snippet:\s*([^,\]]+)", raw)
        result = " ".join(s.strip() for s in snippets[:3])[:600] if snippets else raw[:500]
        print(f"[Tool:web_search] '{query[:30]}' → {len(result)}자")
        return result
    except Exception as e:
        return f"웹 검색 실패: {e}"


_REACT_TOOLS = [rag_search, web_search_tool]
expert_tools_node = ToolNode(_REACT_TOOLS)          # router.py에서 graph에 등록
_expert_llm_with_tools = expert_llm.bind_tools(_REACT_TOOLS)
_MAX_TOOL_ROUNDS = 3                                 # 도구 호출 최대 라운드 (루프 가드)


# ==========================================
# 📡 SSE 로그 버퍼 — main.py의 스트리밍 제너레이터가 읽어서 [LOG] 이벤트로 전송
# ==========================================
_log_buffer = []
_log_lock = threading.Lock()

def _push_log(msg: str):
    with _log_lock:
        _log_buffer.append(msg)

def pop_pending_logs() -> list:
    with _log_lock:
        logs = _log_buffer[:]
        _log_buffer.clear()
        return logs


# ==========================================
# 🧩 2. 공통 유틸리티 (지휘관님 원본 복구)
# ==========================================
def _invoke_with_backoff(llm, prompt, max_retries=3):
    """Groq 413/rate_limit 에러 시 지수 백오프 재시도. 모두 실패 시 로컬 Ollama 자동 폴백."""
    last_err = None
    for attempt in range(max_retries):
        try:
            result = llm.invoke(prompt)
            _metrics.record_llm_call()
            return result
        except Exception as e:
            last_err = e
            err = str(e)
            is_rate_limit = "413" in err or "rate_limit_exceeded" in err.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** attempt  # 1초 → 2초 → 4초
                print(f"[Rate Limit] {wait}초 대기 후 재시도 ({attempt+1}/{max_retries})...")
                time.sleep(wait)
            else:
                break  # rate_limit 아닌 에러 또는 마지막 시도 → 즉시 탈출

    # 모든 재시도 실패 — rate_limit 또는 connection 에러이고 외부 LLM이었으면 로컬로 폴백
    _err_str = str(last_err).lower()
    _is_fallback_target = (
        "rate_limit_exceeded" in _err_str
        or "413" in str(last_err)
        or "connection error" in _err_str   # SSL/네트워크 단절 포함
        or "connecterror" in _err_str
    )
    if last_err and _is_fallback_target:
        if llm is not local_llm:
            msg = "⚠️ Groq API 한도 소진 → 로컬 Ollama로 전환 (답변 품질이 낮을 수 있습니다)"
            print(f"[Fallback] {msg}")
            _push_log(msg)
            _metrics.record_llm_call(is_fallback=True)
            return local_llm.invoke(prompt)  # 로컬도 실패하면 예외 그대로 전파

    raise last_err


def get_dynamic_harness(persona_memo: str = ""):
    base_harness = GLOBAL_HARNESS_MD
    if os.path.exists("added_rules.txt"):
        with open("added_rules.txt", "r", encoding="utf-8") as f:
            additional_rules = f.read()
        base_harness = base_harness + "\n[AI가 실시간으로 학습한 추가 보안 규정]\n" + additional_rules

    return base_harness


_TONE_DESC = {
    "친절한":           "친근하고 따뜻한 어조로 작성하세요.",
    "단호하고 전문적인": "감정 없이 간결하고 단호한 전문가 어조로 작성하세요. '~입니다', '~해야 합니다' 위주로.",
    "사극 이순신 장군":  "조선 시대 사극 말투로 작성하세요. '하옵니다', '소장', '그대', '~이오', '~하소서' 등 고어체를 반드시 사용하세요.",
}
_EXPERTISE_DESC = {
    "기본":         "일반적인 수준으로 설명하세요.",
    "프론트엔드":   "독자가 프론트엔드 개발자이므로 React/Vue/CSS 등 UI 관점 예시를 사용하고 기초 설명은 생략하세요.",
    "백엔드":       "독자가 백엔드 개발자이므로 서버/DB/API 관점 예시를 사용하고 기초 설명은 생략하세요.",
    "데이터 엔지니어": "독자가 데이터 엔지니어이므로 파이프라인/SQL/분산처리 관점 예시를 사용하세요.",
}
_DECISION_DESC = {
    "일반적인": "",
    "간단하게": "SIMPLE",  # 특수 처리: A/B/C 구조 제거, 요약+핵심답변+넥스트 스텝만
    "창의적인": "",        # v2 예정 — 현재는 기본 동작
}


def build_persona_style_block(state: "AgentState") -> str:
    """마이페이지 개인화 설정을 에이전트 프롬프트용 스타일 참고 블록으로 변환.
    history처럼 '이렇게 써줘' 힌트로만 작동하며, A/B/C 구조 등 제품 설계는 건드리지 않음."""
    expertise = (state.get("persona_expertise") or "기본").strip()
    tone = (state.get("persona_tone") or "친절한").strip()
    decision_style = (state.get("persona_decision_style") or "일반적인").strip()
    memo = (state.get("persona_memo") or "").strip()

    # 톤은 persona_agent_node에서 전담 — 여기선 구조에 영향 없는 힌트만 주입
    lines = []
    expertise_desc = _EXPERTISE_DESC.get(expertise, "")
    if expertise != "기본" and expertise_desc:
        lines.append(f"- 설명 깊이: {expertise_desc}")

    # "간단하게"는 구조 자체를 바꾸므로 persona_agent_node가 처리, 여기선 주입 안 함
    is_simple = decision_style == "간단하게"

    if not lines and not is_simple:
        return ""

    block = "\n".join(lines) if lines else ""
    guardrail = "" if is_simple else "A/B/C 구조·요약·넥스트 스텝은 절대 제거 금지 — "
    return (
        f"\n[사용자 스타일 설정 — {guardrail}스타일 힌트만 반영]\n"
        f"{block}\n" if block else f"\n[사용자 스타일 설정]\n"
    )


def format_history(history_list):
    if not history_list: return ""
    formatted = "\n[이전 대화 맥락]\n"
    for msg in history_list:
        is_user = msg.get("role") == "user"
        role = "지휘관" if is_user else "Dati(AI)"
        limit = 400 if is_user else 150
        content = (msg.get('content') or '').replace('[skip] ', '')
        if not is_user:
            # AI 포맷 마커 제거 — 다음 프롬프트에 오염 방지
            content = re.sub(r'^[①②③④⑤]\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^#{1,4}\s+', '', content, flags=re.MULTILINE)
            content = re.sub(r'^>\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
            # 프롬프트 내부 지시어가 AI 답변에 섞여 나온 경우 제거
            content = re.sub(r'⚠️[^\n]*', '', content)
            content = re.sub(r'\[사용자 질문[^\]]*\][^\n]*', '', content)
            content = re.sub(r'\[이전 대화[^\]]*\][^\n]*', '', content)
            content = re.sub(r'\[전 요원[^\]]*\][^\n]*', '', content)
            content = re.sub(r'\[스타일 참고[^\]]*\][\s\S]*?(?=\n\[|\Z)', '', content)
            content = re.sub(r'^※[^\n]*', '', content, flags=re.MULTILINE)
        content = content.strip()[:limit]
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


def _needs_clarification_precheck(query: str, history_list: list = None) -> bool:
    if '[추가 정보:' in query or '[skip]' in query:
        return False
    if any(s in query for s in _SAFE_FROM_CLARIFICATION):
        return False
    # 이전 대화 2턴 이상 → 지시 대명사는 맥락에서 해석 가능, 역질문 불필요
    has_context = bool(history_list and len(history_list) >= 2)
    if any(t in query for t in _AMBIGUITY_TRIGGERS):
        return not has_context
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
        raw = _invoke_with_backoff(json_llm, prompt).content
        # 중첩 없는 단일 JSON 객체만 추출 (greedy 방지)
        m = re.search(r'\{[^{}]*\}', raw)
        result = json.loads(m.group(0)) if m else {}
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
        # 팀/조직 계획 — '짜줘'가 coding으로 잘못 라우팅되는 것 방지
        "5명으로 팀 구성 계획 짜줘",
        "6명으로 스타트업 조직도 짜줘",
        "팀 역할 분배 다시 짜줘",
        "인력 구성 새로 계획해줘",
        "조직 구성 변경해줘",
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
        # 단순 개념 설명/정의 — "X가 뭐야?" / "X이란?" 형태 → general
        "인공지능이 뭐야?", "인공지능이란 뭐야?",
        "머신러닝이란 뭐야?", "머신러닝이 뭐야?",
        "딥러닝이 뭐야?", "딥러닝이란?",
        "블록체인이 뭐야?", "블록체인이란?",
        "API가 뭐야?", "REST API가 뭐야?",
        "클라우드가 뭔지 설명해줘", "클라우드가 뭐야?",
        "애자일 방법론이 뭐야?", "애자일이란?",
        "SaaS가 뭐야?", "SaaS란?",
        "React가 뭐야?", "React란?",
        "Vue가 뭐야?", "Vue.js가 뭐야?",
        "Node.js가 뭐야?", "JavaScript가 뭐야?",
        "TypeScript가 뭐야?", "Docker가 뭐야?",
        "Kubernetes란?", "Git이 뭐야?",
        "GraphQL이 뭐야?", "DevOps가 뭐야?",
        # 대화 기억/맥락 질문
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


# "X가 뭐야?" / "X이란?" 단순 정의 질문 → general_agent 직행
_DEFINITION_RE = re.compile(
    r'^[\w가-힣\s.#+/-]{1,25}\s*(?:이(?:란|란게|라는게)?|가|은|는)?\s*'
    r'(?:뭐야|뭐야\?|뭔가요|뭔가요\?|뭔지|뭔지\?|무엇이야\??|무엇인가요\??)$',
    re.IGNORECASE,
)


def supervisor_node(state):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    print("\n[Supervisor] 시맨틱 라우팅으로 부서 분석 중...")

    # --- 0단계: 사용자가 수동으로 부서를 선택한 경우 → supervisor 스킵 ---
    _VALID_AGENTS = {"general_agent", "expert_agent", "coding_math_agent"}
    force = state.get("force_agent", "")
    if force == "local_test":
        print(f"[Supervisor] 로컬 테스트 모드 → general_agent (Groq 없이 qwen2.5:14b 전담)")
        return {"target_agent_name": "general_agent", "fallback_mode": True}
    if force in _VALID_AGENTS:
        print(f"[Supervisor] 수동 선택 감지 → {force} 직행")
        return {"target_agent_name": force, "fallback_mode": False}

    # --- 1단계: 단순 정의 질문 패턴 → general_agent 직행 (시맨틱·LLM 생략) ---
    if _DEFINITION_RE.match(query.strip()):
        print(f"[Supervisor] 정의 질문 패턴 감지 → general_agent 직행")
        return {"target_agent_name": "general_agent", "fallback_mode": False}

    # --- 1단계: 지시 대명사 감지 → 역질문 트리거 ---
    history_list = state.get("history", [])
    if _needs_clarification_precheck(query, history_list):
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

    # --- 1단계: 시맨틱 점수를 힌트로 LLM에게 전달 → LLM이 항상 최종 결정 ---
    scores = _semantic_score(query)
    print(f"[Semantic] 유사도: { {k: round(v,2) for k,v in scores.items()} }")

    prompt = f"""당신은 Dativus 시스템의 라우팅 관리자입니다.
사용자 질문을 보고 가장 적합한 부서를 단 1개만 출력하세요.

[부서]
- general_agent: 단순 인사·안부, 짧은 일상 대화, "X가 뭐야?" 같은 단순 개념 정의
- expert_agent: 목표설정, 계획수립, 기술 비교·선택, 전략분석, 추천, 사내문서 관련
- coding_math_agent: 코드 작성, 에러 수정, 알고리즘, 수학 계산

[시맨틱 유사도 참고값] (높을수록 해당 부서 예시문장과 유사 — 최종 판단은 의도 우선)
- general_agent: {scores['general_agent']:.2f}
- expert_agent: {scores['expert_agent']:.2f}
- coding_math_agent: {scores['coding_math_agent']:.2f}

[판단 예시]
"이 프로젝트 목표를 정해줘" → expert_agent
"React vs Vue 비교해줘" → expert_agent
"Python이랑 JavaScript 중 뭐 배우는 게 나아?" → expert_agent
"파이썬 리스트에서 인덱스 찾는 법" → coding_math_agent
"파이썬 에러 고쳐줘" → coding_math_agent
"피보나치 수열 짜줘" → coding_math_agent
"안녕하세요" → general_agent
"방금 내가 뭐 물어봤지?" → general_agent
"React가 뭐야?" → general_agent  (시맨틱이 expert 높아도 의도는 단순 정의 → general)
"딥러닝이란?" → general_agent
"Docker가 뭔가요?" → general_agent

질문: {query}
출력(부서 이름만):"""

    try:
        decision = external_llm.invoke(prompt).content.strip().lower()
        if decision not in ["general_agent", "expert_agent", "coding_math_agent"]:
            decision = "general_agent"
        print(f"[Supervisor] LLM 결정: {decision}")
        return {"target_agent_name": decision, "fallback_mode": False}
    except Exception as e:
        print(f"[Supervisor] 외부 LLM 실패: {e} → 로컬 폴백")
        try:
            decision = local_llm.invoke(prompt).content.strip().lower()
            if decision not in ["general_agent", "expert_agent", "coding_math_agent"]:
                decision = "general_agent"
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

    def _do_search():
        query_embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=2,
            where={"workspace_id": workspace_id} if workspace_id else None
        )
        docs = results['documents'][0] if results['documents'] else ["관련 문서를 찾을 수 없습니다."]
        return "\n".join(docs)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            search_context = executor.submit(_do_search).result(timeout=5.0)
        return {"search_context": search_context}
    except FuturesTimeoutError:
        print("🔍 [Vector Search] 5초 타임아웃 — 빈 결과로 계속 진행")
        return {"search_context": ""}
    except Exception as e:
        print(f"🔍 [Vector Search] 실패: {e}")
        return {"search_context": ""}


def web_search_node(state: AgentState):
    query = state["query"]
    print(f"🌐 [Web Search] DuckDuckGo를 통해 '{query}' 최신 정보를 수집합니다...")

    def _do_search():
        search_tool = DuckDuckGoSearchResults(num_results=3)
        raw = search_tool.invoke(query)
        snippets = re.findall(r'snippet:\s*([^,\]]+)', raw)
        if snippets:
            return ' '.join(s.strip() for s in snippets[:2])[:500]
        return re.sub(r'https?://\S+', '', raw)[:400]

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            web_ctx = executor.submit(_do_search).result(timeout=8.0)
        return {"web_context": web_ctx}
    except FuturesTimeoutError:
        print("🌐 [Web Search] 8초 타임아웃 — 빈 결과로 계속 진행")
        return {"web_context": ""}
    except Exception as e:
        print(f"🌐 [Web Search] 검색 실패 (무시): {e}")
        return {"web_context": ""}


def search_coordinator_node(state: AgentState):
    """VectorRAG / WebSearch / GraphRAG 3개 수색 노드를 병렬 실행하는 팬아웃 트리거."""
    print("🚀 [검색 코디네이터] VectorRAG · WebSearch · GraphRAG 병렬 수색 시작...")
    return {}


def conversation_memory_node(state: AgentState):
    """일반 대화팀 전용 — 이전 대화 맥락 확인 (LLM 없음, 빠름)."""
    history_list = state.get("history", [])
    if not history_list:
        print("💬 [대화 기억] 이전 대화 없음")
    else:
        print(f"💬 [대화 기억] {len(history_list)}턴 대화 맥락 확인 완료")
    return {}


def code_search_node(state: AgentState):
    """코딩팀 전용 레퍼런스 검색 — 업로드 문서에서 코드 예시 추출 (VectorRAG)."""
    query = state["query"]
    workspace_id = state.get("workspace_id")
    print(f"📚 [레퍼런스 검색] 코드 예시·기술 문서 검색 중...")

    def _do_search():
        embedding = model.encode(query).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=2,
            where={"workspace_id": workspace_id} if workspace_id else None
        )
        docs = results['documents'][0] if results['documents'] else []
        return "\n".join(docs)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            ctx = executor.submit(_do_search).result(timeout=5.0)
        return {"search_context": ctx}
    except FuturesTimeoutError:
        print("📚 [레퍼런스 검색] 5초 타임아웃")
        return {"search_context": ""}
    except Exception as e:
        print(f"📚 [레퍼런스 검색] 실패: {e}")
        return {"search_context": ""}


def graph_memory_node(state: AgentState):
    history_list = state.get("history", [])
    if not history_list:
        print("🕸️ [GraphRAG] 이전 대화 없음 — 건너뜀")
        return {"graph_context": ""}

    query = state["query"]
    history_str = format_history(history_list)
    print("🕸️ [GraphRAG] 대화 맥락과 유기적 관계(Entity-Relation)를 분석합니다...")

    prompt = f"""당신은 '지식 그래프(Knowledge Graph) 분석가'입니다.
    아래 대화 기록에서 질문에 답하기 위해 필요한 '핵심 개체(Entity)'와 '관계(Relation)'를 추출하세요.
    형식: [개체A] ➔ (관계) ➔ [개체B]

    [이전 대화 맥락]: {history_str}
    [사용자 질문]: {query}
    """

    def _do_graph():
        return local_llm.invoke(prompt).content

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            graph_memory = executor.submit(_do_graph).result(timeout=10.0)
        return {"graph_context": graph_memory}
    except FuturesTimeoutError:
        print("🕸️ [GraphRAG] 10초 타임아웃 — 빈 컨텍스트로 계속 진행")
        return {"graph_context": ""}
    except Exception as e:
        print(f"🕸️ [GraphRAG] 실패: {e}")
        return {"graph_context": ""}


# ==========================================
# 🔎 4-2. Phase 1 — 검색 인텔리전스 노드
# ==========================================

def query_rewriter_node(state: AgentState):
    """검색 최적화 쿼리 재작성 + 레인 선택 + 멀티홉 분해."""
    query = state["query"]
    history_str = format_history(state.get("history", []))

    prompt = QUERY_REWRITER_PROMPT.format(
        query=query,
        history=history_str[:300] if history_str else "없음",
    )
    try:
        json_llm = external_llm.bind(response_format={"type": "json_object"})
        raw = _invoke_with_backoff(json_llm, prompt).content
        result = json.loads(raw)
        rewritten = result.get("rewritten", query).strip() or query
        lanes = result.get("search_lanes", ["vector", "web", "graph"])
        if not isinstance(lanes, list) or not lanes:
            lanes = ["vector", "web", "graph"]
        # 멀티홉 서브쿼리 — 중복·빈 값 제거
        raw_subs = result.get("sub_queries", [])
        sub_queries = [s.strip() for s in raw_subs if s.strip() and s.strip() != rewritten][:3]
        if sub_queries:
            print(f"[QueryRewriter] 멀티홉 분해: {sub_queries}")
        print(f"[QueryRewriter] '{query[:30]}' → '{rewritten[:30]}' | 레인: {lanes}")
        return {
            "query_rewritten": rewritten,
            "sub_queries": sub_queries,
            "search_plan": lanes,
            "search_attempts": 0,
        }
    except Exception as e:
        print(f"[QueryRewriter] 실패 → 원문 그대로 사용: {e}")
        return {
            "query_rewritten": query,
            "sub_queries": [],
            "search_plan": ["vector", "web", "graph"],
            "search_attempts": 0,
        }


def selective_search_node(state: AgentState):
    """search_plan에 따라 선택적 검색 실행. sub_queries가 있으면 각각 검색 후 결과 합산 (멀티홉)."""
    plan = state.get("search_plan", ["vector", "web", "graph"])
    main_query = state.get("query_rewritten") or state["query"]
    sub_queries = state.get("sub_queries") or []
    workspace_id = state.get("workspace_id")

    # 실제로 검색할 쿼리 목록: main + sub (중복 제거)
    all_queries = [main_query] + [q for q in sub_queries if q != main_query]
    print(f"[SelectiveSearch] 레인: {plan} | 쿼리 {len(all_queries)}개 {[q[:25] for q in all_queries]}")

    def _vector_one(q: str) -> str:
        try:
            emb = model.encode(q).tolist()
            r = collection.query(
                query_embeddings=[emb],
                n_results=2,
                where={"workspace_id": workspace_id} if workspace_id else None,
            )
            return "\n".join(r["documents"][0]) if r["documents"] else ""
        except Exception as e:
            print(f"[SelectiveSearch] vector 실패({q[:20]}): {e}")
            return ""

    def _web_one(q: str) -> str:
        try:
            search = DuckDuckGoSearchResults(num_results=3)
            raw = search.invoke(q)
            snippets = re.findall(r"snippet:\s*([^,\]]+)", raw)
            return " ".join(s.strip() for s in snippets[:2])[:400] if snippets else raw[:350]
        except Exception as e:
            print(f"[SelectiveSearch] web 실패({q[:20]}): {e}")
            return ""

    def _graph() -> str:
        history_list = state.get("history", [])
        if not history_list:
            return ""
        try:
            history_str = format_history(history_list)
            prompt = (
                f"대화 기록에서 질문과 관련된 핵심 개체와 관계를 추출하세요.\n"
                f"형식: [개체A] → (관계) → [개체B]\n"
                f"[이전 대화]: {history_str}\n[질문]: {main_query}"
            )
            return local_llm.invoke(prompt).content
        except Exception as e:
            print(f"[SelectiveSearch] graph 실패: {e}")
            return ""

    vector_parts, web_parts = [], []
    graph_ctx = ""

    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = {}

        # vector: 쿼리별 병렬
        if "vector" in plan:
            for i, q in enumerate(all_queries):
                futures[f"v_{i}"] = exe.submit(_vector_one, q)

        # web: 쿼리별 병렬
        if "web" in plan:
            for i, q in enumerate(all_queries):
                futures[f"w_{i}"] = exe.submit(_web_one, q)

        # graph: main_query 1회만
        if "graph" in plan:
            futures["graph"] = exe.submit(_graph)

        # 결과 수집
        for i in range(len(all_queries)):
            if f"v_{i}" in futures:
                try:
                    part = futures[f"v_{i}"].result(timeout=5.0)
                    if part:
                        vector_parts.append(f"[서브쿼리: {all_queries[i]}]\n{part}")
                except Exception:
                    pass
            if f"w_{i}" in futures:
                try:
                    part = futures[f"w_{i}"].result(timeout=8.0)
                    if part:
                        web_parts.append(f"[서브쿼리: {all_queries[i]}]\n{part}")
                except Exception:
                    pass
        if "graph" in futures:
            try:
                graph_ctx = futures["graph"].result(timeout=10.0)
            except Exception:
                pass

    search_ctx = "\n\n".join(vector_parts)[:1000]
    web_ctx    = "\n\n".join(web_parts)[:600]
    return {"search_context": search_ctx, "web_context": web_ctx, "graph_context": graph_ctx}


def search_grader_node(state: AgentState):
    """검색 결과 품질 채점. 부족하면 query_rewritten 갱신 + search_attempts 증가."""
    query = state.get("query_rewritten") or state["query"]
    search_ctx = (state.get("search_context", "") or "")[:400]
    web_ctx = (state.get("web_context", "") or "")[:200]
    graph_ctx = (state.get("graph_context", "") or "")[:200]
    attempts = state.get("search_attempts", 0)

    # 검색 결과 자체가 없으면 sufficient(검색 불가 환경)
    if not search_ctx and not web_ctx and not graph_ctx:
        print("[SearchGrader] 수집 결과 없음 → sufficient 패스스루")
        return {"search_grade": "sufficient"}

    prompt = SEARCH_GRADER_PROMPT.format(
        query=query,
        search_ctx=search_ctx or "없음",
        web_ctx=web_ctx or "없음",
        graph_ctx=graph_ctx or "없음",
    )
    try:
        json_llm = external_llm.bind(response_format={"type": "json_object"})
        raw = _invoke_with_backoff(json_llm, prompt).content
        result = json.loads(raw)
        grade = result.get("grade", "sufficient")

        if grade == "rewrite":
            refined = result.get("refined_query", query).strip() or query
            reason = result.get("reason", "")
            print(f"[SearchGrader] 부족({reason}) → 재쿼리: '{refined[:40]}' (시도 {attempts+1}/2)")
            return {
                "search_grade": "rewrite",
                "query_rewritten": refined,
                "search_attempts": attempts + 1,
            }
        else:
            print(f"[SearchGrader] 충분 ✓ ({attempts+1}회차)")
            return {"search_grade": "sufficient"}

    except Exception as e:
        print(f"[SearchGrader] 실패 → sufficient 처리: {e}")
        return {"search_grade": "sufficient"}


def route_from_search_grader(state: AgentState):
    grade = state.get("search_grade", "sufficient")
    attempts = state.get("search_attempts", 0)
    if grade == "rewrite" and attempts < 2:
        return "selective_search"
    return "expert_agent"


# ==========================================
# 👔 5. 추론 및 생성 부서 (Logic Workers) - 💡 동적 하네스 탑재!
# ==========================================

def expert_agent_react_node(state: AgentState):
    """ReAct expert agent: bind_tools + agent→tool→agent 조건부 루프."""

    # 폴백 모드: qwen은 function calling 신뢰도 낮음 → 도구 없이 기존 방식
    if state.get("fallback_mode", False):
        print("⚠️ [ReAct] 폴백 모드 → 도구 없이 expert_agent 실행")
        return expert_agent_node(state)

    # workspace_id를 thread-local에 저장 (tools 내부에서 사용)
    _tool_ctx.workspace_id = state.get("workspace_id", "")

    messages = list(state.get("messages") or [])

    # ── 첫 진입: 초기 SystemMessage + HumanMessage 구성 ──
    if not messages:
        query = state.get("query_rewritten") or state["query"]
        history_str = format_history(state.get("history", []))
        persona_block = build_persona_style_block(state)

        # 사전 수집된 검색 컨텍스트 (selective_search 결과)
        search_ctx = (state.get("search_context", "") or "")[:500]
        web_ctx    = (state.get("web_context",    "") or "")[:300]
        graph_ctx  = (state.get("graph_context",  "") or "")[:200]
        pre_ctx = ""
        if search_ctx:
            pre_ctx += f"\n[사전 수집 — 사내 문서]:\n{search_ctx}"
        if web_ctx:
            pre_ctx += f"\n[사전 수집 — 웹]:\n{web_ctx}"
        if graph_ctx:
            pre_ctx += f"\n[사전 수집 — 대화 맥락]:\n{graph_ctx}"

        # clarify_hint
        clarify_hint = ""
        if "[추가 정보:" in query:
            tags = re.findall(r"\[추가 정보:\s*([^\]]+)\]", query)
            if tags:
                clarify_hint = f"\n[역질문 선택 결과 — 이 주제 중심으로 답변]: {', '.join(tags)}\n"

        system_content = (
            f"{get_dynamic_harness()}\n\n"
            f"{EXPERT_OUTPUT_FORMAT}\n"
            f"{persona_block}\n"
            f"당신은 최고 전문가 요원입니다.\n"
            f"• 제공된 사전 정보가 충분하면 즉시 최종 답변을 작성하세요.\n"
            f"• 추가 정보가 필요하면 rag_search 또는 web_search_tool을 호출하세요 (최대 {_MAX_TOOL_ROUNDS}회).\n"
            f"• 반드시 [사용자 질문]에만 답하세요. 이전 대화는 맥락 참고용, 재생성 금지.\n"
            f"• 사내 문서가 질문과 무관하면 완전히 무시하고 일반 지식으로 답하세요.\n"
            f"{clarify_hint}"
            f"{history_str}"
            f"{pre_ctx}"
        )
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=query),
        ]

    # ── LLM 호출 ──
    # 이미 도구를 MAX-1회 호출했으면 마지막 호출에서 tools 제거 → 텍스트 강제
    current_tool_rounds = sum(1 for m in messages if getattr(m, "tool_calls", None))
    active_llm_for_react = expert_llm if current_tool_rounds >= _MAX_TOOL_ROUNDS - 1 else _expert_llm_with_tools
    if current_tool_rounds >= _MAX_TOOL_ROUNDS - 1:
        print(f"⚠️ [ReAct] 도구 {current_tool_rounds}회차 — 마지막 호출, 텍스트 강제")
    else:
        print(f"🤖 [ReAct] LLM 호출 (메시지 {len(messages)}개, 도구 {current_tool_rounds}회차)")
    response = _invoke_with_backoff(active_llm_for_react, messages)

    tool_calls = getattr(response, "tool_calls", None)

    if not tool_calls:
        # 도구 호출 없음 → 최종 답변 확정
        print("✅ [ReAct] 최종 답변 생성 완료")
        return {
            "messages": [response],
            "draft_answer": response.content,
        }

    # 도구 호출 있음 → tool_calls_history 기록 후 ToolNode로 이동
    history = list(state.get("tool_calls_history") or [])
    for tc in tool_calls:
        history.append({"tool": tc["name"], "args": tc.get("args", {})})
        print(f"🔧 [ReAct] 도구 호출 예약: {tc['name']}({tc.get('args', {})})")

    return {
        "messages": [response],
        "tool_calls_history": history,
    }


def should_use_tools(state: AgentState) -> str:
    """ReAct 분기: 마지막 메시지에 tool_calls가 있으면 'tools', 없으면 'end'."""
    messages = state.get("messages") or []
    if not messages:
        return "end"
    last = messages[-1]
    if not getattr(last, "tool_calls", None):
        return "end"
    # 루프 가드: 도구 호출 라운드 횟수 제한
    tool_rounds = sum(1 for m in messages if getattr(m, "tool_calls", None))
    if tool_rounds >= _MAX_TOOL_ROUNDS:
        print(f"⛔ [ReAct] 도구 {_MAX_TOOL_ROUNDS}라운드 한도 도달 → end")
        return "end"
    return "tools"


def expert_agent_node(state: AgentState):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else expert_llm

    search_ctx = (state.get("search_context", "") or "")[:500]
    web_ctx = (state.get("web_context", "") or "")[:300]
    graph_ctx = (state.get("graph_context", "") or "")[:300]
    persona_block = build_persona_style_block(state)

    print("👔 [전문 대화병] 수집된 입체적 기억(Graph + Vector + Web)을 융합 추론합니다.")

    # [추가 정보:] 태그가 있으면 역질문 후 보강된 쿼리 — 태그 내용을 핵심 주제로 우선 처리
    clarify_hint = ""
    if '[추가 정보:' in query:
        import re as _re
        tags = _re.findall(r'\[추가 정보:\s*([^\]]+)\]', query)
        if tags:
            clarify_hint = f"\n[역질문 선택 결과 - 반드시 이 주제를 중심으로 답변]: {', '.join(tags)}\n"

    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
    {persona_block}
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
    persona_block = build_persona_style_block(state)

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
    {persona_block}
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
    active_llm = local_llm if fallback_mode else coding_llm
    search_ctx = (state.get("search_context", "") or "")[:400]
    persona_block = build_persona_style_block(state)

    clarify_hint = ""
    if '[추가 정보:' in query:
        tags = re.findall(r'\[추가 정보:\s*([^\]]+)\]', query)
        if tags:
            clarify_hint = f"\n[역질문 선택 결과 - 반드시 이 주제를 중심으로 답변]: {', '.join(tags)}\n"

    print("💻 [코딩/수학 대화병] 알고리즘 분석 및 수학 연산을 수행합니다.")

    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
    {persona_block}
    {CODING_OUTPUT_FORMAT}
    당신은 시니어 개발자 수준의 프로그래밍 및 수학 전문가입니다.
    ⚠️ 반드시 [사용자 질문]에만 답하세요. 이전 대화는 맥락 참고용이며, 이전 대화 내용을 그대로 반복하거나 재생성하지 마세요.
    {clarify_hint}
    {history_str}
    [참고 문서(레퍼런스)]: {search_ctx if search_ctx else '없음'}
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
# 🧬 7-1. 개인화 에이전트 (Persona Agent)
# ==========================================
def persona_agent_node(state: AgentState):
    """톤/말투 전담 변환 노드. 내용·구조는 절대 건드리지 않고 표현 방식만 바꿈.
    tone이 기본값('친절한')이고 memo도 없으면 passthrough."""
    # 폴백 모드: qwen2.5:14b도 하네스 포함 긴 프롬프트에서 신뢰도 낮음 → 원본 그대로 통과
    if state.get("fallback_mode", False):
        print("⚠️ [Persona] 폴백 모드 — 톤 변환 생략, 원본 그대로 통과")
        return {}

    tone = (state.get("persona_tone") or "친절한").strip()
    memo = (state.get("persona_memo") or "").strip()
    decision_style = (state.get("persona_decision_style") or "일반적인").strip()
    is_simple = decision_style == "간단하게"

    # 변환할 내용 없으면 즉시 패스스루
    if tone == "친절한" and not memo and not is_simple:
        return {}

    draft = state.get("draft_answer", "")
    if not draft:
        return {}

    tone_desc = _TONE_DESC.get(tone, f"'{tone}' 스타일로 작성하세요.")
    style_lines = [f"- 말투: {tone_desc}"]
    if memo:
        style_lines.append(f"- 추가 지시: {memo}")
    style_block = "\n".join(style_lines)

    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else expert_llm

    # 테이블(|)과 블록쿼트(>) 행 보호 — LLM 재작성 중 구조 날아가는 것 방지
    lines = draft.split('\n')
    protected = {
        i: line for i, line in enumerate(lines)
        if line.strip().startswith('|') or line.strip().startswith('>')
    }
    prose_only = '\n'.join('' if i in protected else line for i, line in enumerate(lines))

    print(f"🎭 [개인화 노드] 톤 변환 중... ('{tone}', 보호 {len(protected)}행)")

    if is_simple:
        prompt = f"""아래 [원본 텍스트]를 간결하게 재구성하세요.

[출력 형식 — 반드시 이 3단계만]
1. 한 줄 요약 (> **요약:** 으로 시작)
2. 핵심 답변 (A/B/C 구조 없이 2~4문장으로 직접 답변)
3. 넥스트 스텝 (> 마음에 드는 방안을 고르시면 세부 실행 계획을 설계해 드릴까요?)

[절대 금지]
- A안/B안/C안 구조 사용 금지
- 장점/단점 불릿 나열 금지
- 표(|) 사용 금지
- 서론 없이 바로 본문만 출력

{f"[말투]{chr(10)}{style_block}" if style_block.strip() else ""}

[원본 텍스트]
{draft}"""
    else:
        prompt = f"""아래 [원본 텍스트]의 각 문장을 [스타일 지시]에 맞게 자연스럽게 다시 써주세요.

[절대 금지]
- 내용·정보 추가 금지 — 원본에 없는 문장, 설명, 예시를 절대 만들지 말 것
- 원본보다 길어지는 것 금지 — 각 문장은 원본과 비슷한 길이를 유지할 것
- 문장 끝에 ", 하옵니다" 등 경어를 단순히 덧붙이는 것 금지
- 잘못된 예: "유지보수가 용이합니다, 하옵니다"
- 올바른 예: "유지보수가 용이하옵니다" (문장 자체를 재작성)

[지켜야 할 것]
- ###, A안/B안/C안, **장점:**, **단점:** 등 마크다운 헤더와 구조는 그대로 유지
- 서론 없이 바로 본문만 출력

[스타일 지시]
{style_block}

[원본 텍스트]
{prose_only}"""

    try:
        result = _invoke_with_backoff(active_llm, prompt).content.strip()
        result_lines = result.split('\n')
        for idx, original_line in protected.items():
            if idx < len(result_lines):
                result_lines[idx] = original_line
            else:
                result_lines.append(original_line)
        return {"draft_answer": '\n'.join(result_lines)}
    except Exception as e:
        print(f"[개인화 노드 실패] {e} — 원본 유지")
        return {}


# ==========================================
# ⚖️ 7. 검수 (Critic) 및 수정
# ==========================================
def critic_node(state: AgentState):
    draft_answer = state.get("draft_answer", "")
    count = state.get("revision_count", 0)

    # 폴백 모드: 하네스 없는 단순 프롬프트로 qwen 간이 검수
    if state.get("fallback_mode", False):
        print("⚠️ [Critic] 폴백 모드 — qwen 간이 검수 실행")
        simple_prompt = (
            f"아래 [답변]이 [질문]에 제대로 답하고 있으면 PASS, 핵심 내용이 빠졌거나 엉뚱하면 FAIL만 출력하세요.\n\n"
            f"[질문]: {state.get('query', '')}\n"
            f"[답변]: {draft_answer[:600]}\n\n"
            f"출력 (PASS 또는 FAIL만):"
        )
        try:
            decision = local_llm.invoke(simple_prompt).content.strip().upper()
            if "PASS" in decision:
                print("✅ [Critic] qwen 간이 검수 통과!")
                return {"final_answer": draft_answer, "critic_feedback": json.dumps({"pass": True}, ensure_ascii=False)}
            else:
                print(f"❌ [Critic] qwen 간이 검수 반려")
                fb = {"pass": False, "reasons": ["답변이 질문을 충분히 다루지 못함"], "fix_targets": ["질문의 핵심에 직접 답변하세요"]}
                return {"critic_feedback": json.dumps(fb, ensure_ascii=False), "revision_count": count + 1}
        except Exception as e:
            print(f"⚠️ [Critic] qwen 간이 검수 실패 → 자동 PASS: {e}")
            return {"final_answer": draft_answer, "critic_feedback": json.dumps({"pass": True}, ensure_ascii=False)}

    # 수정 한도 도달 (Phase 1: 2회) → 강제 PASS
    if count >= 2:
        print("🚨 [Critic] 수정 한도 도달(2회). 강제 PASS!")
        return {"final_answer": draft_answer, "critic_feedback": json.dumps({"pass": True}, ensure_ascii=False)}

    # general_agent 답변은 짧은 대화 — Critic 불필요
    if state.get("target_agent_name") == "general_agent":
        print("✅ [Critic] 일반 대화 — 검수 생략")
        return {"final_answer": draft_answer, "critic_feedback": json.dumps({"pass": True}, ensure_ascii=False)}

    # 코딩 응답에 코드 블록 없으면 즉시 FAIL (결정론적)
    if state.get("target_agent_name") == "coding_math_agent" and "```" not in draft_answer:
        print(f"❌ [Critic] 코드 블록 없음 → 즉시 반려")
        fb = {
            "pass": False,
            "reasons": ["코드 블록(```) 누락"],
            "fix_targets": ["반드시 실제 코드를 ```언어명 ... ``` 형식으로 포함하세요"],
        }
        return {"critic_feedback": json.dumps(fb, ensure_ascii=False), "revision_count": count + 1}

    print(f"🧐 [Critic] JSON 모드 품질 검사 중... (수정 {count}회)")
    critic_draft = (draft_answer[:700] + "\n...(중략)...\n" + draft_answer[-300:]) if len(draft_answer) > 1000 else draft_answer
    prompt = CRITIC_SYSTEM_PROMPT_JSON.format(query=state.get("query", ""), draft=critic_draft)

    try:
        json_llm = external_llm.bind(response_format={"type": "json_object"})
        raw = _invoke_with_backoff(json_llm, prompt).content
        result = json.loads(raw)
        if result.get("pass"):
            print("✅ [Critic] 검수 통과!")
            return {"final_answer": draft_answer, "critic_feedback": json.dumps(result, ensure_ascii=False)}
        else:
            reasons = result.get("reasons", [])
            print(f"❌ [Critic] 반려 — {', '.join(reasons)}")
            return {"critic_feedback": json.dumps(result, ensure_ascii=False), "revision_count": count + 1}
    except Exception as e:
        print(f"⚠️ [Critic] 에러 → 비상 통과: {e}")
        return {"final_answer": draft_answer, "critic_feedback": json.dumps({"pass": True}, ensure_ascii=False)}


def revision_agent_node(state: AgentState):
    query = state["query"]
    draft = state.get("draft_answer", "")
    fallback_mode = state.get("fallback_mode", False)
    active_llm = local_llm if fallback_mode else coding_llm

    # Phase 1: JSON 피드백에서 fix_targets 추출
    try:
        fb = json.loads(state.get("critic_feedback", "{}"))
        reasons = fb.get("reasons", [])
        fix_targets = fb.get("fix_targets", [])
        feedback_text = ""
        if reasons:
            feedback_text += f"수정 사유: {', '.join(reasons)}\n"
        if fix_targets:
            feedback_text += f"수정 지시: {chr(10).join(f'- {t}' for t in fix_targets)}"
        feedback_text = feedback_text.strip() or "전반적인 품질 개선"
    except Exception:
        feedback_text = state.get("critic_feedback", "전반적인 품질 개선")

    print(f"🛠️ [개선/보충 대화병] Critic 지적 반영 수정 중... | {feedback_text[:60]}")

    prompt = f"""[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n
당신은 답변 수정 요원입니다.

[원본 답변]:
{draft}

🚨 [지적사항]:
{feedback_text}

[사용자 질문]: {query}

규칙:
- [지적사항]에 해당하는 부분만 최소한으로 수정하세요.
- 지적받지 않은 부분은 원문 그대로 유지하세요.
- JSON, 대시보드 데이터는 포함하지 마세요.
- 서론("수정된 답변:" 등) 없이 수정된 본문만 출력하세요."""

    response = _invoke_with_backoff(active_llm, prompt).content
    return {"draft_answer": response}


def check_critic_approval(state: AgentState):
    raw = state.get("critic_feedback", "")
    try:
        fb = json.loads(raw)
        return "end" if fb.get("pass") else "revision"
    except Exception:
        # 레거시 문자열 포맷 호환
        return "end" if "PASS" in raw.upper() else "revision"


# ==========================================
# 🎭 커스텀 에이전트 게이트 (Custom Ego Gate)
# ==========================================
_CUSTOM_AGENT_AUTO_THRESHOLD = 0.38  # 이 유사도 이상이면 자동 선택


def custom_agent_gate_node(state: AgentState):
    """커스텀 에이전트 게이트 — 수동 선택 또는 자동 매칭 후 메인 답변 뒤에 관점 추가."""
    query = state["query"]
    draft = state.get("draft_answer", "")
    history_str = format_history(state.get("history", []))
    fallback_mode = state.get("fallback_mode", False)

    # 1. 수동 선택된 에이전트 우선
    agent_name = state.get("custom_agent_name", "")
    agent_prompt = state.get("custom_agent_prompt", "")
    agent_type = state.get("custom_agent_type", "EXTERNAL_API")  # LOCAL / EXTERNAL_API

    # 2. 수동 선택 없으면 자동 매칭
    if not agent_name or not agent_prompt:
        agents_list = state.get("custom_agents_list", [])
        if not agents_list:
            return {}

        query_vec = model.encode(query, normalize_embeddings=True)
        passing_agents = []
        for agent in agents_list:
            desc = agent.get("description", "")
            if not desc:
                continue
            desc_vec = model.encode(desc, normalize_embeddings=True)
            score = float(np.dot(query_vec, desc_vec))
            agent_threshold = agent.get("threshold", _CUSTOM_AGENT_AUTO_THRESHOLD)
            if score >= agent_threshold:
                passing_agents.append((score, agent))

        if not passing_agents:
            print(f"🎭 [커스텀 에이전트] threshold를 넘은 에이전트 없음 — 스킵")
            return {}

        # 점수 내림차순 정렬
        passing_agents.sort(key=lambda x: x[0], reverse=True)

        # ── 다중 매칭: 각 에이전트 독립 실행 → 별도 버블로 스트리밍 ──
        if len(passing_agents) > 1:
            print(f"🎭 [커스텀 에이전트 다중 매칭] {len(passing_agents)}개 활성화")
            multi_responses = []
            for score, agent in passing_agents:
                a_name = agent["name"]
                a_prompt_text = agent["description"]
                a_type = agent.get("agent_type", "EXTERNAL_API")
                a_llm = local_llm if (fallback_mode or a_type == "LOCAL") else expert_llm
                print(f"  → '{a_name}' ({a_type}, 유사도: {score:.2f})")
                prompt = CUSTOM_AGENT_PROMPT.format(
                    agent_name=a_name,
                    agent_prompt=a_prompt_text,
                    history_str=history_str,
                    query=query,
                )
                try:
                    response = _invoke_with_backoff(a_llm, prompt).content.strip()
                    multi_responses.append({"name": a_name, "response": response})
                except Exception as e:
                    print(f"  ⚠️ '{a_name}' 실패 (무시): {e}")
            return {"multi_agent_responses": multi_responses}

        # ── 단일 매칭: 기존 동작 ──
        best_score, best_agent = passing_agents[0]
        agent_name = best_agent["name"]
        agent_prompt = best_agent["description"]
        agent_type = best_agent.get("agent_type", "EXTERNAL_API")
        print(f"🎭 [커스텀 에이전트 자동 선택] '{agent_name}' ({agent_type}, 유사도: {best_score:.2f})")

    # 수동 선택 + 자동 단일 매칭 공통: agent_type으로 LLM 결정
    active_llm = local_llm if (fallback_mode or agent_type == "LOCAL") else expert_llm
    print(f"🎭 [커스텀 에이전트: {agent_name}] {'로컬 LLM' if agent_type == 'LOCAL' else '외부 API'} 으로 관점 추가 중...")
    prompt = CUSTOM_AGENT_PROMPT.format(
        agent_name=agent_name,
        agent_prompt=agent_prompt,
        history_str=history_str,
        query=query,
    )

    try:
        opinion = _invoke_with_backoff(active_llm, prompt).content.strip()
        appended = f"{draft}\n\n> 💡 **[{agent_name}]:** {opinion}"
        return {"draft_answer": appended, "matched_custom_agent_name": agent_name}
    except Exception as e:
        print(f"🎭 [커스텀 에이전트] 실패 (무시): {e}")
        return {}


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