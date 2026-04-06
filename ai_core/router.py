import os
from langchain_ollama import ChatOllama
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from database.chroma_manager import collection
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# 모델 로딩 (라우터가 혼자 생각할 수 있도록 번역기를 달아줍니다)
print("🧠 라우터용 임베딩 모델(BAAI/bge-m3) 로딩 중...")
model = SentenceTransformer('BAAI/bge-m3')

# 👇 로컬 LLM(Llama-3) 엔진 세팅 추가!
print("🤖 로컬 LLM(Llama-3) 연결 준비 중...")
local_llm = ChatOllama(model="llama3", temperature=0) # temperature=0 은 가장 정확하고 진지하게 답하라는 뜻입니다!

print("⚡ 외부 LLM(Groq) 연결 준비 중...")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
external_llm = ChatGroq(
    temperature=0,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"  # Groq 서버에서 제공하는 초고속 Llama-3 모델
)

# 1. 상태(State) 정의: AI의 뇌혈관을 타고 흐를 데이터의 형태입니다.
class AgentState(TypedDict):
    query: str
    workspace_id: str
    final_answer: str


# 2. 노드(행동) 함수들 정의: 각각의 도착지에서 AI가 할 행동입니다.
def greeting_node(state: AgentState):
    print("👋 [Level 1] 인사말 노드 실행: 단순 인사말로 판별되었습니다.")
    return {"final_answer": "안녕하세요! Dativus 팀 협업 AI입니다. 무엇을 도와드릴까요?"}


def rag_node(state: AgentState):
    print("🗄️ [Level 2] RAG 검색 노드 실행: 팀 내부 지식을 탐색합니다.")
    query = state["query"]
    workspace_id = state["workspace_id"]

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=1,
        where={"workspace_id": workspace_id}
    )

    if not results['documents'] or not results['documents'][0]:
        return {"final_answer": "ChromaDB를 뒤져봤지만, 해당 내용과 관련된 팀 지식을 찾지 못했습니다."}

    best_knowledge = results['documents'][0][0]

    # 🚨 프롬프트(명령서)를 더욱 강력하게 수정!
    prompt = f"""[System]
    당신은 Dativus 팀의 스마트하고 친절한 AI 어시스턴트입니다.
    아래 [팀 지식]을 바탕으로 사용자의 [질문]에 대해 요약해서 답변하세요.

    중요한 규칙:
    1. 반드시 100% 한국어(Korean)로만 대답해야 합니다. 절대 영어를 사용하지 마세요.
    2. 친절하고 자연스러운 대화체로 답변하세요.

    [팀 지식]: 
    {best_knowledge}

    [질문]: {query}
    """

    print("💭 Llama-3가 문서를 읽고 답변을 작성 중입니다...")
    # Llama-3 가동! (그래픽카드가 열일하는 순간)
    response = local_llm.invoke(prompt)

    return {"final_answer": response.content}


def external_llm_node(state: AgentState):
    print("🧠 [Level 3] 외부 LLM 토론 노드 실행: Groq API를 호출합니다.")
    query = state["query"]

    # 일반 지식을 묻는 질문이므로, 팀 지식(RAG) 없이 바로 질문을 던집니다.
    prompt = f"""[System]
당신은 Dativus 팀의 똑똑한 AI 어시스턴트입니다.
사용자의 질문에 대해 친절하고 명확하게 '한국어(Korean)'로 답변해주세요.

[사용자 질문]: {query}
"""
    print("⚡ Groq가 초고속으로 답변을 생성 중입니다...")
    response = external_llm.invoke(prompt)

    return {"final_answer": response.content}


# 3. 라우터(조건부 엣지) 함수: 어디로 갈지 길을 정해주는 핵심 '뇌'입니다!
def route_query(state: AgentState):
    query = state["query"]
    print(f"\n🤔 라우터 판단 중... 입력된 질문: '{query}'")

    # 1. 인사말 컷 (Level 1)
    if "안녕" in query or "반가워" in query:
        return "greeting_node"

    # 🚨 2. AI 교통경찰 출동 (의도 파악 프롬프트)
    prompt = f"""당신은 사용자의 질문 의도를 분류하는 AI 라우터입니다.
아래 질문을 읽고, 오직 'TEAM' 또는 'GENERAL' 중 하나의 단어만 출력하세요. 다른 말은 절대 하지 마세요.

[분류 기준]
TEAM: 특정 팀, 회의록, 우리 프로젝트, 프론트엔드 에이스, 팀원, 비밀문서 등 '우리 팀 내부의 기밀이나 현황'을 묻는 질문
GENERAL: 날씨, 일반적인 프로그래밍 지식(데이터베이스, 정규화 등), 상식 등 누구나 아는 외부 지식

[질문]: {query}
[결과]:"""

    # 초고속 Groq에게 판단을 맡깁니다! (보안이 극도로 중요하다면 local_llm을 써도 됩니다)
    print("🚦 AI 교통경찰이 질문의 의도를 분석 중입니다...")
    decision = external_llm.invoke(prompt).content.strip().upper()

    # 3. 교통경찰의 판단에 따라 길을 엽니다!
    if "TEAM" in decision:
        print("🎯 [교통경찰 판단] 팀 내부 지식입니다! -> Level 2(로컬 보안 AI)로 연결")
        return "rag_node"
    else:
        print("🌐 [교통경찰 판단] 일반/외부 지식입니다! -> Level 3(외부 AI)로 연결")
        return "external_llm_node"


# 4. 그래프 조립하기 (LangGraph의 핵심!)
workflow = StateGraph(AgentState)

# 노드 등록
workflow.add_node("greeting_node", greeting_node)
workflow.add_node("rag_node", rag_node)
workflow.add_node("external_llm_node", external_llm_node)

# 시작점을 라우터로 설정 (모든 질문은 라우터를 먼저 거칩니다)
workflow.set_conditional_entry_point(route_query)

# 각 노드가 끝나면 무조건 종료(END)되도록 길을 연결
workflow.add_edge("greeting_node", END)
workflow.add_edge("rag_node", END)
workflow.add_edge("external_llm_node", END)

# 뇌 구조 완성!
app = workflow.compile()

# ==========================================
# 🧪 테스트 코드 (이 파일만 단독으로 실행해서 확인해 봅니다)
# ==========================================
if __name__ == "__main__":
    print("=== LangGraph 라우팅 테스트 시작 ===")

    test_queries = [
        "안녕! 반가워",
        "우리 팀 프론트엔드 에이스가 누구야?",  # <- 요런 거 하나 추가!
        "데이터베이스 정규화가 뭐야?"
    ]

    REAL_WORKSPACE_ID = "fff7c64d-6da9-4bc8-a731-47b0f7e39dae"

    for q in test_queries:
        result = app.invoke({"query": q, "workspace_id": REAL_WORKSPACE_ID})
        print(f"👉 최종 답변: {result['final_answer']}\n")