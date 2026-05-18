# ai_core/nodes.py
import os
import requests
import asyncio
from ai_core.state import AgentState
from ai_core.prompts import *
from database.chroma_manager import collection
from database.postgres import get_user_persona
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchResults
import time
# 모델 초기화
model = SentenceTransformer('BAAI/bge-m3')
local_llm = ChatOllama(model="llama3", temperature=0)
external_llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")


def get_dynamic_harness():
    import os
    # 💡 1. prompts.py에 정의된 순수 코드 변수를 기본 베이스로 가져옵니다.
    # (GLOBAL_HARNESS_MD는 이미 nodes.py 상단에서 'from ai_core.prompts import *'로 가져온 상태입니다)
    base_harness = GLOBAL_HARNESS_MD

    # 💡 2. 만약 AI가 자가 진화하여 누적한 추가 규칙 파일이 있다면 뒤에 이어 붙입니다.
    if os.path.exists("added_rules.txt"):
        with open("added_rules.txt", "r", encoding="utf-8") as f:
            additional_rules = f.read()
        return base_harness + "\n[AI가 실시간으로 학습한 추가 보안 규정]\n" + additional_rules

    return base_harness

def format_history(history_list):
    if not history_list: return ""
    formatted = "\n[이전 대화 맥락]\n"
    for msg in history_list:
        role = "지휘관" if msg.get("role") == "user" else "Dati(AI)"
        formatted += f"{role}: {msg.get('content')}\n"
    return formatted + "\n"


def initialize_persona_node(state: AgentState):
    user_id = state.get("user_id")
    persona = get_user_persona(user_id) if user_id else None

    if state.get("target_agent_name") and state.get("target_agent_prompt"):
        return {"persona": persona}

    if user_id:
        try:
            res = requests.get(f"http://127.0.0.1:8080/api/v1/agents/user/{user_id}")
            if res.status_code == 200:
                agents = res.json()
                if agents:
                    agents_info = "\n".join([f"- 이름: {a['name']}, 역할: {a['description']}" for a in agents])
                    prompt = ORCHESTRATOR_SYSTEM_PROMPT.format(agents_info=agents_info, query=state["query"])
                    decision = external_llm.invoke(prompt).content.strip()
                    for a in agents:
                        if a["name"].lower() in decision.lower():
                            return {"persona": persona, "target_agent_name": a["name"],
                                    "target_agent_prompt": a["description"]}
        except Exception as e:
            print(f"[오류] {e}")
    return {"persona": persona, "revision_count": 0, "feedback": None, "writer": None}


# 💡 [개조 완료] 사내 검색병: 'team_context' 서류철에만 보고서를 씁니다.
def search_node(state: AgentState):
    query = state["query"]
    print(f"📁 [사내 검색병] 기밀문서에서 '{query}' 관련 정보를 찾습니다...")
    workspace_id = state.get("workspace_id")
    if "팀 지식" in query or "전체 요약" in query:
        all_docs = collection.get(where={"workspace_id": workspace_id} if workspace_id else None)
        if not all_docs or not all_docs.get("documents"):
            return {"team_context": "문서가 없습니다."}

        # 파일별 주요 내용 추출 로직
        docs, metas = all_docs["documents"], all_docs["metadatas"]
        file_dict = {}
        for d, m in zip(docs, metas):
            fname = m.get("file_name", "Unknown")
            file_dict.setdefault(fname, []).append(d)

        selected = []
        for fname, chunks in file_dict.items():
            selected.append(f"\n=== [{fname}] ===\n" + "\n".join(chunks[:5]))
        return {"team_context": "\n".join(selected)}

    query_embedding = model.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=1,
                               where={"workspace_id": workspace_id} if workspace_id else None)

    context_result = results['documents'][0][0] if results['documents'] and results['documents'][
        0] else "관련 문서를 찾을 수 없습니다."
    return {"team_context": f"[사내 문서 검색 결과]\n{context_result}"}


# 💡 [개조 완료] 웹 검색병: 'web_context' 서류철에만 보고서를 씁니다.
def web_search_node(state: AgentState):
    query = state["query"]
    print(f"🌐 [웹 검색병] 외부 인터넷망에서 '{query}' 관련 최신 정보를 수집합니다...")
    search_tool = DuckDuckGoSearchResults(num_results=3)
    search_result = search_tool.invoke(query)
    return {"web_context": f"[웹 검색 결과]\n{search_result}"}


# 💡 [개조 완료] 요약병: 두 요원의 서류철을 가져와서 하나로 융합(Synthesis)합니다.
def summary_node(state: AgentState):
    query = state["query"]
    print("📝 [요약/통합병] 수집된 모든 정보를 교차 검증하며 융합합니다...")

    # 두 검색병이 가져온 서류를 꺼냄 (출동 안 한 요원의 서류는 빈칸)
    team_data = state.get("team_context", "")
    web_data = state.get("web_context", "")

    # 두 정보를 융합
    combined_context = f"{team_data}\n\n{web_data}".strip()

    if "팀 지식" in query:
        prompt = SUMMARY_BYPASS_PROMPT.format(context=combined_context)
    else:
        prompt = SUMMARY_SECURITY_PROMPT.format(query=query, context=combined_context)

    response = local_llm.invoke(prompt)
    return {"summary": response.content.strip()}


def commander_node(state: AgentState):
    summary = state.get("summary", "")
    persona = state.get("persona") or {}
    feedback = state.get("feedback")
    draft = state.get("final_answer", "") # 💡 [추가] 자기가 썼던 초안 기억하기

    if "SECURITY_ALERT_CROSS_TEAM" in summary:
        return {"final_answer": "🚨 타 팀 데이터 접근 권한이 없습니다.", "writer": "commander"}

    prompt = COMMANDER_SYSTEM_PROMPT.format(
        decision_style=persona.get('decision_style', '일반적인'),
        expertise=persona.get('expertise', '기본'),
        tone=persona.get('tone', '친절한'),
        history_str=format_history(state.get("history", [])),
        summary=summary,
        query=state["query"]
    )

    prompt = f"[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n\n" + prompt

    if feedback and feedback != "PASS":
        print(f"💦 [Commander] 검수자 지적사항 반영하여 초안 수정 중... ({feedback})")
        prompt += f"\n\n[이전 작성 초안]:\n{draft}\n\n🚨 [검수자 지적사항]: {feedback}\n위 지적사항을 완벽하게 반영하여 '이전 작성 초안'을 보완한 새로운 답변을 작성하세요!"

    return {"final_answer": local_llm.invoke(prompt).content, "writer": "commander"}


def external_llm_node(state: AgentState):
    query = state["query"]
    history_str = format_history(state.get("history", []))
    feedback = state.get("feedback")
    draft = state.get("final_answer", "") # 💡 [추가] 자기가 썼던 초안 기억하기

    # 💡 [수정] 직접 쓴 프롬프트 대신, prompts.py의 EXTERNAL_LLM_PROMPT를 호출하여 하네스 룰 적용!
    prompt = EXTERNAL_LLM_PROMPT.format(
        history_str=history_str,
        query=query
    )

    prompt = f"[전 요원 필독 하네스 룰]\n{get_dynamic_harness()}\n\n" + prompt

    if feedback and feedback != "PASS":
        print(f"💦 [External LLM] 검수자 지적사항 반영하여 초안 수정 중... ({feedback})")
        prompt += f"\n\n[이전 작성 초안]:\n{draft}\n\n🚨 [검수자 지적사항]: {feedback}\n위 지적사항을 완벽하게 반영하여 '이전 작성 초안'을 보완한 새로운 답변을 작성하세요!"

    return {"final_answer": external_llm.invoke(prompt).content, "writer": "external_llm"}

def custom_agent_node(state: AgentState):
    feedback = state.get("feedback")
    draft = state.get("final_answer", "") # 💡 [추가] 자기가 썼던 초안 기억하기

    prompt = CUSTOM_AGENT_PROMPT.format(
        agent_name=state.get("target_agent_name"),
        agent_prompt=state.get("target_agent_prompt"),
        history_str=format_history(state.get("history", [])),
        query=state["query"]
    )

    if feedback and feedback != "PASS":
        print(f"💦 [{state.get('target_agent_name')} 요원] 검수자 지적사항 반영하여 초안 수정 중... ({feedback})")
        prompt += f"\n\n[이전 작성 초안]:\n{draft}\n\n🚨 [검수자 지적사항]: {feedback}\n위 지적사항을 완벽하게 반영하여 '이전 작성 초안'을 보완한 새로운 답변을 작성하세요!"

    return {"final_answer": external_llm.invoke(prompt).content, "writer": "custom_agent"}

def greeting_node(state: AgentState):
    return {"final_answer": "안녕하세요! Dativus 팀 협업 AI Dati입니다."}


def critic_node(state: AgentState):
    query = state["query"]
    draft = state.get("final_answer", "")
    count = state.get("revision_count", 0)

    if count >= 2:
        print("🚨 [Critic 요원] 수정 한도 초과. 현재 초안을 강제로 승인합니다.")
        return {"feedback": "PASS"}

    print(f"🧐 [Critic 요원] 초안 답변 품질 검사 중... (현재 수정: {count}회)")
    prompt = CRITIC_SYSTEM_PROMPT.format(query=query, draft=draft)
    decision = external_llm.invoke(prompt).content.strip()

    if decision.startswith("PASS"):
        print("✅ [Critic 요원] 완벽한 답변입니다. 승인 (PASS)!")
        return {"feedback": "PASS"}
    else:
        print(f"❌ [Critic 요원] 답변 반려! 재작성 지시 ➔ {decision}")
        time.sleep(5)
        return {"feedback": decision, "revision_count": count + 1}


# 💡 [신규] GraphRAG를 흉내 내는 '관계망 추론 요원'
def graph_memory_node(state: AgentState):
    query = state["query"]
    print(f"🕸️ [관계망 추론 요원] 대화 맥락과 팀원 간의 유기적 관계(Entity-Relation)를 분석합니다...")

    history_str = format_history(state.get("history", []))
    team_data = state.get("team_context", "")

    # LLM에게 텍스트에서 '관계(Graph)'를 뽑아내도록 지시하는 프롬프트
    prompt = f"""당신은 Dativus 시스템의 '지식 그래프(Knowledge Graph) 분석가'입니다.
    아래의 대화 기록과 사내 데이터를 분석하여, 사용자의 질문에 답하기 위해 필요한 '핵심 개체(Entity)'와 그들 간의 '관계(Relation)'를 추출하세요.

    반드시 아래와 같은 [노드 ➔ 엣지 ➔ 노드] 형식의 '관계망 형태'로만 요약해서 제출하세요.
    예시: [팀원 A] ➔ (담당한다) ➔ [프론트엔드 UI]
    예시: [프론트엔드 UI] ➔ (통신한다) ➔ [Spring Boot 서버]

    [이전 대화 맥락]:
    {history_str}

    [사내 기밀 데이터]:
    {team_data}

    [사용자 질문]: {query}

    [관계망 추론 결과]:"""

    # 이 복잡한 추론은 똑똑한 외부 LLM(Groq)이 담당합니다.
    graph_memory = external_llm.invoke(prompt).content

    # 분석한 관계망을 사내 서류철에 스며들게(추가) 합니다.
    new_team_context = f"{team_data}\n\n[🕸️ 유기적 관계망 분석]\n{graph_memory}"

    return {"team_context": new_team_context}