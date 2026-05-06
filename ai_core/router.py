import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# 내부 모듈 임포트
from database.chroma_manager import collection
from database.postgres import get_user_persona

load_dotenv()

# ==========================================
# 1. 모델 로딩 및 초기화
# ==========================================
print("라우터용 임베딩 모델(BAAI/bge-m3) 로딩 중...")
model = SentenceTransformer('BAAI/bge-m3')

print("로컬 LLM (Llama-3) 연결 준비 중...")
local_llm = ChatOllama(model="llama3", temperature=0)

print("외부 LLM (Groq) 연결 준비 중...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
external_llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)


# ==========================================
# 2. 상태(State) 정의 - (협업을 위한 기억 공간 추가!)
# ==========================================
class AgentState(TypedDict):
    query: str
    workspace_id: Optional[str]
    user_id: Optional[str]
    persona: Optional[dict]

    # 💡 에이전트끼리 주고받을 서류철 2개 추가!
    context: Optional[str]  # 1번 요원이 2번 요원에게 줄 원본 자료
    summary: Optional[str]  # 2번 요원이 3번 요원에게 줄 요약본

    final_answer: str


# ==========================================
# 3. 노드(Node) 함수 정의
# ==========================================
def initialize_persona_node(state: AgentState):
    user_id = state.get("user_id")
    persona = None
    if user_id:
        print(f"[시스템] 유저 ID({user_id})의 페르소나를 조회합니다...")
        persona = get_user_persona(user_id)
        if persona:
            print(f"[시스템] 페르소나 장착 완료: {persona.get('tone', '')}")
    return {"persona": persona}


def greeting_node(state: AgentState):
    print("[Level 1] 인사말 노드 실행")
    return {"final_answer": "안녕하세요! Dativus 팀 협업 AI Dati입니다. 무엇을 도와드릴까요?"}


# ----------------------------------------------------
# 💡 [핵심 개조] 3인 1조 RAG 특수부대 파이프라인
# ----------------------------------------------------

# 요원 1: 자료 검색병 -> [Agent: Retriever]
def search_node(state: AgentState):
    print("[Agent: Retriever] ChromaDB 사내 지식베이스 수색 중...")
    query = state["query"]
    workspace_id = state.get("workspace_id")

    query_embedding = model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where={"workspace_id": workspace_id} if workspace_id else None
    )

    if not results['documents'] or not results['documents'][0]:
        return {"context": "관련 문서를 찾을 수 없습니다."}

    best_knowledge = results['documents'][0][0]
    print("[Agent: Retriever] 자료 확보 완료! Summarizer에게 전달합니다.")
    return {"context": best_knowledge}


# 요원 2: 문서 요약병 -> [Agent: Summarizer]
def summary_node(state: AgentState):
    # 💡 텍스트만 바꾸는 게 아니라, 진짜로 외부망(Groq)을 끊고 내부망(Local)을 씁니다!
    print("[Agent: Summarizer] 검색된 자료를 핵심만 정제 중 (Local Llama-3 가동)...")
    context = state.get("context", "")
    query = state["query"]

    if context == "관련 문서를 찾을 수 없습니다.":
        return {"summary": "자료 없음"}

    # 🚨 [보안 패치 1] 요약병에게 타 팀 검색 시도를 원천 차단하는 지시 하달
    prompt = f"""당신은 Dativus 팀의 분석가입니다. 
    다음 자료를 바탕으로 질문에 대한 핵심 정보만 3줄 이내로 간결하게 요약하세요.

    [절대 보안 수칙]:
    만약 사용자의 질문이 현재 워크스페이스가 아닌 다른 특정 팀(예: 거북선2팀, 타 부서 등)의 파일이나 정보를 요구하는 내용이라면, 원본 자료에 무슨 내용(코드 등)이 있든 절대 요약하지 말고 오직 "SECURITY_ALERT_CROSS_TEAM" 이라는 문구만 정확히 출력하세요.

    [질문]: {query}
    [원본 자료]: {context}
    [핵심 요약]:"""

    # 🚨 [핵심 보안 수정] external_llm -> local_llm으로 완벽 교체! 기밀 데이터 외부 유출 원천 차단!
    response = local_llm.invoke(prompt)
    print("[Agent: Summarizer] 요약 완료! Commander에게 보고서를 올립니다.")
    return {"summary": response.content.strip()}


# 요원 3: AI 어시스턴트 -> [Agent: Commander]
def commander_node(state: AgentState):
    print("[Agent: Commander] 요약 보고서를 바탕으로 최종 답변 스트리밍 준비 중 (Local Llama-3 가동)...")
    persona = state.get("persona")
    summary = state.get("summary", "")
    query = state["query"]

    # 🚨 [보안 패치 2] 요약병이 올린 보안 경고를 확인하면 즉시 방어 태세 돌입!
    if "SECURITY_ALERT_CROSS_TEAM" in summary:
        return {"final_answer": "🚨 [보안 경고] 타 워크스페이스(팀)의 데이터에는 접근할 권한이 없습니다."}

    if summary == "자료 없음":
        return {"final_answer": "해당 내용과 관련된 팀 지식을 찾지 못했습니다."}

    # 💡 DB에서 가져온 페르소나가 있다면 적용하고, 없다면 '기본 AI 어시스턴트'로 동작!
    if persona and (persona.get('decision_style') or persona.get('expertise') or persona.get('tone')):
        system_msg = f"""당신은 Dativus 팀의 스마트한 AI 어시스턴트입니다.
        사용자가 설정한 다음 페르소나에 맞춰 대답하세요:
        - 판단 스타일: {persona.get('decision_style', '일반적인')}
        - 전문 분야: {persona.get('expertise', '기본')}
        - 어조: {persona.get('tone', '친절한')}"""
    else:
        system_msg = "당신은 Dativus 팀의 스마트하고 친절한 AI 어시스턴트입니다."

    prompt = f"""{system_msg}
    반드시 100% 한국어로만 대답하세요. 
    당신은 사용자에게 기술적인 '코드의 원리'를 설명할 수는 있지만, 당신 스스로가 시스템 관리자처럼 타 팀의 데이터를 직접 꺼내줄 수는 없습니다. 자신이 할 수 없는 행동을 할 수 있다고 허풍떨지 마세요.

    다음 '요약 보고서'를 바탕으로 사용자에게 답변하세요.

    [요약 보고서]: {summary}
    [사용자 질문]: {query}
    [최종 답변]:"""

    response = local_llm.invoke(prompt)
    return {"final_answer": response.content}


# ----------------------------------------------------

def external_llm_node(state: AgentState):
    print("[Level 3] 외부망 LLM (Groq) 실행 중...")
    query = state["query"]
    persona = state.get("persona")

    if persona:
        system_msg = f"어조: {persona.get('tone', '친절한')}에 맞춰 대답하세요."
    else:
        system_msg = "친절하게 대답하세요."

    prompt = f"{system_msg}\n\n[질문]: {query}"
    response = external_llm.invoke(prompt)
    return {"final_answer": response.content}


# ==========================================
# 4. 교통경찰 (조건부 엣지 라우터)
# ==========================================
def route_query(state: AgentState) -> str:
    query = state["query"]
    print(f"\n라우터 판단 중... 질문: '{query}'")
    print(f"[AI 교통경찰] 의도 분석 중... (TEAM vs GENERAL)")
    if "안녕" in query or "반가워" in query:
        return "greeting"

    prompt = f"""당신은 라우터입니다. 오직 'TEAM' 또는 'GENERAL' 중 하나만 출력하세요.
    '문서', '파일', '업로드' 같은 단어가 있으면 무조건 TEAM 입니다.
    [질문]: {query}
    [결과]:"""

    decision = external_llm.invoke(prompt).content.strip().upper()
    if "TEAM" in decision:
        print("[SECURITY_CHECK] Classification: INTERNAL_DATA (Sensitive).")
        print("[판단] ➔ 사내망 라우팅(Level 2). 외부 인터넷 연결 전면 차단.")
        return "search"  # 💡 rag가 아니라 search(1번 요원)로 보냅니다!
    else:
        print(f"[판단] ➔ 일반 지식(Level 3). 외부망(Groq) 라우팅 허용.")
        return "external_llm"


# ==========================================
# 5. 그래프 조립 (LangGraph)
# ==========================================
workflow = StateGraph(AgentState)

# (1) 모든 요원(노드) 등록
workflow.add_node("initialize", initialize_persona_node)
workflow.add_node("greeting", greeting_node)
workflow.add_node("search", search_node)
workflow.add_node("summary", summary_node)
workflow.add_node("commander", commander_node)
workflow.add_node("external_llm", external_llm_node)

# (2) 시작점
workflow.set_entry_point("initialize")

# (3) 교차로 (라우터)
workflow.add_conditional_edges(
    "initialize",
    route_query,
    {
        "greeting": "greeting",
        "search": "search",
        "external_llm": "external_llm"
    }
)

# (4) 💡 3인 1조 릴레이 연결
workflow.add_edge("search", "summary")
workflow.add_edge("summary", "commander")

# (5) 종착역
workflow.add_edge("greeting", END)
workflow.add_edge("commander", END)
workflow.add_edge("external_llm", END)

# 앱 컴파일
langgraph_app = workflow.compile()