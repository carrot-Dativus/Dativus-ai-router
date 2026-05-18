from fastapi import BackgroundTasks, UploadFile, File, FastAPI, Depends, HTTPException, status, Form
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import shutil
import os
import asyncio
import requests
from jose import JWTError, jwt
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from database.chroma_manager import collection
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import uuid
from pydantic import BaseModel
from typing import Optional
from ai_core.router import langgraph_app
from fastapi.middleware.cors import CORSMiddleware
import time # 💡 운영 로그 시간 측정을 위한 모듈
from langchain_groq import ChatGroq

app = FastAPI(title="Dativus AI Core API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str
    workspace_id: Optional[str] = None
    history: Optional[list] = []
    target_agent_name: Optional[str] = None
    target_agent_prompt: Optional[str] = None

load_dotenv()

print("임베딩 모델(BAAI/bge-m3) 로딩 중...")
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로딩 완료!")

security = HTTPBearer()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        if "user_id" not in payload:
            raise HTTPException(status_code=401, detail="토큰에 유저 식별 정보(user_id)가 없습니다.")
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다."
        )

@app.get("/")
def read_root():
    return {"message": "Dativus AI Core (FastAPI)가 정상 작동 중입니다! (Port 8000)"}

@app.post("/api/v1/chat")
async def chat_with_ai(
        request: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    user_id = token_payload.get("user_id")
    workspace_id = token_payload.get("workspace_id")
    print(f"\n==================================================================")
    print(f"[채팅 요청] 유저: {user_id} | 워크스페이스: {workspace_id}")
    print(f"\n==================================================================")

    # 💡 [일반 채팅 운영 로그 기록 시작]
    start_time = time.time()

    result = langgraph_app.invoke({
        "query": request.query,
        "workspace_id": workspace_id,
        "user_id": user_id
    })

    # 💡 [일반 채팅 운영 로그 연산]
    latency = round(time.time() - start_time, 2)
    final_answer = result.get("final_answer", "")
    estimated_tokens = int((len(request.query) + len(final_answer)) * 0.8)

    print(f"⏱️ [운영 로그] 일반 동기식 답변 생성 완료.")
    print(f"   ➔ 소요 시간: {latency}초 | 소모 토큰 추정: {estimated_tokens} Tokens")
    print(f"==================================================================")

    return {
        "status": "success",
        "query": request.query,
        "answer": final_answer,
        "latency": latency,         # 프론트엔드 연동용 데이터 추가
        "tokens": estimated_tokens   # 프론트엔드 연동용 데이터 추가
    }

@app.post("/api/v1/chat/stream")
async def chat_with_ai_stream(
        request: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    user_id = token_payload.get("user_id")
    workspace_id = request.workspace_id or token_payload.get("workspace_id")

    print(f"\n==================================================================")
    print(f"[스트리밍 요청] 유저: {user_id} | 워크스페이스: {workspace_id}")
    print(f"\n==================================================================")

    async def event_generator():
        # 💡 [스트리밍 운영 로그 기록 시작] 최고 관리자(Supervisor) 작전 타임 측정 시작
        start_time = time.time()

        inputs = {
            "query": request.query,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "history": request.history,
            "target_agent_name": request.target_agent_name,
            "target_agent_prompt": request.target_agent_prompt
        }

        node_kor_name = {
            "initialize": "초기화(System)",
            "greeting": "인사말 담당(Greeter)",
            "supervisor": "최고 관리자(Supervisor)", # 💡 추가된 관리자 한글 매핑
            "search": "자료 검색병(Retriever)",
            "summary": "문서 요약병(Summarizer)",
            "commander": "최종 커맨더(Commander)",
            "external_llm": "외부망 연결(Groq)",
            "custom_agent": "특수 빙의 요원(Ego)",
            "critic": "품질 검수 요원(Critic)",
            "web_search": "웹 검색병(Web Searcher)",
            "graph_memory": "관계망 추론병(Graph Memory)" # 💡 추가된 관계망 요원 매핑
        }

        current_final_answer = ""

        async for event in langgraph_app.astream(inputs):
            for node_name, output in event.items():

                if output and isinstance(output, dict) and "final_answer" in output:
                    current_final_answer = output["final_answer"]

                agent_name = node_kor_name.get(node_name, node_name)

                if node_name == "critic":
                    if output.get("feedback") == "PASS":
                        yield f"data: [LOG]✅ [Critic] 검수 통과! 완벽한 답변입니다.\n\n"
                    else:
                        yield f"data: [LOG]❌ [Critic] 답변 반려 및 재작성 지시: {output.get('feedback')}\n\n"
                    await asyncio.sleep(0.05)
                else:
                    yield f"data: [LOG]🟢 [{agent_name}] 작전 수행 완료.\n\n"
                    await asyncio.sleep(0.05)

                if output and isinstance(output, dict):
                    if "team_context" in output and node_name == "search": # team_context 구조 반영
                        yield f"data: [LOG]📄 사내 지식베이스 수색 및 관련 문서 조각 확보 완료.\n\n"
                        await asyncio.sleep(0.05)
                    if "web_context" in output and node_name == "web_search": # web_context 구조 반영
                        yield f"data: [LOG]🌐 외부 웹망 실시간 데이터 크롤링 및 수집 완료.\n\n"
                        await asyncio.sleep(0.05)
                    if "team_context" in output and node_name == "graph_memory": # graph_memory 로그 세분화
                        yield f"data: [LOG]🕸️ 문맥 내 핵심 Entity 간 유기적 관계성(Knowledge Graph) 추론 완료.\n\n"
                        await asyncio.sleep(0.05)
                    if "summary" in output and node_name == "summary":
                        yield f"data: [LOG]📝 보안 검사 통과 및 지식 통합 브리핑 생성 완료.\n\n"
                        await asyncio.sleep(0.05)

                if node_name == "critic" and output.get("feedback") == "PASS":
                    for char in current_final_answer:
                        yield f"data: {char}\n\n"
                        await asyncio.sleep(0.01)
                elif node_name == "greeting" and "final_answer" in output:
                    for char in current_final_answer:
                        yield f"data: {char}\n\n"
                        await asyncio.sleep(0.01)

        # 💡 [스트리밍 최종 운영 로그 연산 레이어]
        latency = round(time.time() - start_time, 2)
        # 총 텍스트 길이를 바탕으로 현업 표준 텍스트 대 토큰 가중치(0.8)를 적용해 비용 산정
        estimated_tokens = int((len(request.query) + len(current_final_answer)) * 0.8)

        print(f"⏱️ [운영 로그] 스트리밍 전체 작전 종료.")
        print(f"   ➔ 총 소요 시간: {latency}초 | 총 소모 토큰 추정: {estimated_tokens} Tokens")
        print(f"==================================================================")

        # 💡 [프론트엔드 전송 레이어] 실시간 화면 로그창에 대기업 솔루션처럼 속도와 비용 지표를 쏴줍니다!
        yield f"data: [LOG]⏱️ 작전 소요 시간: {latency}초 | 🪙 소모 토큰: {estimated_tokens} Tokens\n\n"
        await asyncio.sleep(0.05)

        yield f"data: [LOG]🏁 모든 에이전트 통신 및 답변 생성 종료.\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/api/v1/documents/upload")
async def upload_document(
        background_tasks: BackgroundTasks,
        document_id: str = Form(...),
        file: UploadFile = File(...),
        token_payload: dict = Depends(verify_token)
):
    workspace_id = token_payload.get("workspace_id")
    UPLOAD_DIR = "temp_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_and_store_document, file_path, file.filename, workspace_id, document_id)

    return {
        "status": "success",
        "message": f"'{file.filename}' 파일 접수 완료! 백그라운드에서 AI 분석을 시작합니다."
    }

def process_and_store_document(file_path: str, filename: str, workspace_id: str, document_id: str):
    try:
        print(f"[백그라운드] '{filename}' 파일 AI 뇌 각인 시작...")
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        else:
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        ids = []
        embeddings = []
        metadatas = []
        documents_text = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            text = chunk.page_content
            ids.append(chunk_id)
            documents_text.append(text)
            embeddings.append(model.encode(text).tolist())
            metadatas.append({
                "workspace_id": workspace_id,
                "file_name": filename,
                "chunk_index": i
            })

        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )

        print(f"[백그라운드] '{filename}' 분석 완료! 스프링(지휘소)으로 무전을 칩니다!")
        webhook_url = "http://127.0.0.1:8080/api/v1/documents/webhook"
        requests.post(webhook_url, json={"documentId": document_id, "status": "DONE"})

    except Exception as e:
        print(f"[백그라운드] 에러 발생: {e}")
        requests.post("http://127.0.0.1:8080/api/v1/documents/webhook",
                      json={"documentId": document_id, "status": "FAILED"})
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.delete("/api/v1/documents")
async def delete_document_vectors(workspace_id: str, file_name: str):
    try:
        collection.delete(
            where={
                "$and": [
                    {"workspace_id": workspace_id},
                    {"file_name": file_name}
                ]
            }
        )
        print(f"🔥 [망각 완료] 워크스페이스({workspace_id})의 '{file_name}' 기억이 뇌에서 영구 삭제되었습니다.")
        return {"status": "success", "message": "기억 삭제 완료"}
    except Exception as e:
        print(f"🚨 [망각 실패] {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
#--------------------------------------------------------------------------
#프롬프트 자동 최적화
#--------------------------------------------------------------------------
class FailedLogItem(BaseModel):
    query: str
    answer: str

class OptimizeRequest(BaseModel):
    logs: list[FailedLogItem]

@app.post("/api/v1/prompts/optimize")
async def optimize_system_prompt(request: OptimizeRequest):
    print("🧠 [Auto-Optimizer] 오답 노트 분석 및 프롬프트 자가 진화 시작...")

    optimizer_llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

    log_text = ""
    for i, log in enumerate(request.logs):
        log_text += f"[{i + 1}번 실패 사례]\n- 사유: {log.query}\n- 답변: {log.answer}\n\n"

    prompt = f"""당신은 Dativus 시스템의 프롬프트 최적화 수석 엔지니어입니다.
    아래의 오답 노트를 보고, AI가 앞으로 절대 같은 실수를 하지 않도록 방어하는 [새로운 추가 규칙 1줄]을 작성하세요.

    [오답 노트 기록]
    {log_text}

    반드시 문장 앞에 '-' 기호를 붙여 핵심 추가 규칙 딱 1줄만 출력하세요.
    예시: - 데이터베이스 관련 기술 요약 브리핑 시 '진격'과 같은 가벼운 군대식 페르소나 단어 사용을 엄격히 금지할 것."""

    new_rule = optimizer_llm.invoke(prompt).content.strip()

    # 💡 [핵심] 기존 코드는 건드리지 않고, 새로 깨달은 규칙만 가벼운 파일에 한 줄씩 추가(Append)합니다.
    with open("added_rules.txt", "a", encoding="utf-8") as f:
        f.write(f"{new_rule}\n")

    print(f"✨ [진화 완료] 시스템 신규 누적 규칙 각인: {new_rule}")
    return {"status": "success", "new_rule": new_rule}

