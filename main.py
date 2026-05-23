import sys
import os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import BackgroundTasks, UploadFile, File, FastAPI, Depends, HTTPException, status, Form, Request
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
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from langchain_groq import ChatGroq
import sys

sys.stdout.reconfigure(line_buffering=True)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Dativus AI Core API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


SPRING_BASE_URL = os.getenv("SPRING_BASE_URL", "http://127.0.0.1:8080")

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    history: Optional[list] = []
    target_agent_name: Optional[str] = None
    target_agent_prompt: Optional[str] = None
    existing_dashboard: Optional[dict] = None


load_dotenv()

print("임베딩 모델(BAAI/bge-m3) 로딩 중...", flush=True)
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로딩 완료!", flush=True)

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


from fastapi.responses import JSONResponse

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"error": "서버 내부 오류가 발생했습니다."})


@app.get("/")
def read_root():
    return {"message": "Dativus AI Core (FastAPI)가 정상 작동 중입니다! (Port 8000)"}


@app.post("/api/v1/chat")
@limiter.limit("20/minute")
async def chat_with_ai(
        request: Request,
        body: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    user_id = token_payload.get("user_id")
    workspace_id = token_payload.get("workspace_id")
    print(f"\n==================================================================")
    print(f"[채팅 요청] 유저: {user_id} | 워크스페이스: {workspace_id}")
    print(f"\n==================================================================")

    start_time = time.time()

    inputs = {
        "query": body.query,
        "workspace_id": workspace_id,
        "user_id": user_id
    }
    result = await asyncio.to_thread(langgraph_app.invoke, inputs)

    latency = round(time.time() - start_time, 2)
    final_answer = result.get("final_answer", "")
    estimated_tokens = int((len(body.query) + len(final_answer)) * 0.8)

    print(f"⏱️ [운영 로그] 일반 동기식 답변 생성 완료.")
    print(f"   ➔ 소요 시간: {latency}초 | 소모 토큰 추정: {estimated_tokens} Tokens")
    print(f"==================================================================")

    return {
        "status": "success",
        "query": body.query,
        "answer": final_answer,
        "latency": latency,
        "tokens": estimated_tokens
    }


async def save_message_to_backend(session_id: str, user_id: str, sender_type: str,
                                   sender_name: str, content: str, is_private: bool,
                                   latency: float, tokens: int, bearer_token: str):
    if not session_id:
        return
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                f"{SPRING_BASE_URL}/api/v1/chats/messages",
                json={
                    "sessionId": session_id,
                    "userId": user_id,
                    "senderType": sender_type,
                    "senderName": sender_name,
                    "content": content,
                    "isPrivate": is_private,
                    "latency": latency,
                    "tokens": tokens,
                },
                headers={"Authorization": f"Bearer {bearer_token}"},
            )
    except Exception as e:
        print(f"[메시지 저장 실패] {e}")


@app.post("/api/v1/chat/stream")
@limiter.limit("20/minute")
async def chat_with_ai_stream(
        request: Request,
        body: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    user_id = token_payload.get("user_id")
    workspace_id = body.workspace_id or token_payload.get("workspace_id")
    bearer_token = request.headers.get("Authorization", "").replace("Bearer ", "")

    print(f"\n==================================================================")
    print(f"[스트리밍 요청] 유저: {user_id} | 워크스페이스: {workspace_id}", flush=True)
    print(f"\n==================================================================")

    async def event_generator():
        print("★ [진입 성공] 프론트엔드 요청이 FastAPI에 도달했습니다!", flush=True)
        start_time = time.time()
        inputs = {
            "query": body.query,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "history": body.history,
            "target_agent_name": body.target_agent_name,
            "target_agent_prompt": body.target_agent_prompt,
            "existing_dashboard": body.existing_dashboard or {},
        }

        node_kor_name = {
            "initialize": "초기화(System)",
            "greeting": "인사말 담당(Greeter)",
            "supervisor": "최고 관리자(Supervisor)",
            "search": "자료 검색병(Retriever)",
            "summary": "문서 요약병(Summarizer)",
            "commander": "최종 커맨더(Commander)",
            "external_llm": "일반 대화병(External)",
            "custom_agent": "특수 빙의 요원(Ego)",
            "critic": "품질 검수 요원(Critic)",
            "web_search": "웹 검색병(Web Searcher)",
            "graph_memory": "관계망 추론병(Graph Memory)"
        }

        queue = asyncio.Queue()

        async def run_langgraph():
            try:
                async for event in langgraph_app.astream(inputs):
                    await queue.put(("event", event))
            except Exception as e:
                import traceback
                traceback.print_exc()
                await queue.put(("error", str(e)))
            except BaseException as e:
                import traceback
                print("\n🚨🚨🚨 [통신 단절 / 강제 종료 감지] 🚨🚨🚨")
                print(f"범인(에러)의 정체: {type(e).__name__}")
                traceback.print_exc()

                if type(e).__name__ != "CancelledError":
                    await queue.put(("error", f"💣 시스템 에러 발생: {str(e)}"))

                raise e
            finally:
                await queue.put(("done", None))

        task = asyncio.create_task(run_langgraph())
        current_final_answer = ""
        current_dashboard_data = {}

        try:
            while True:
                try:
                    msg_type, data = await asyncio.wait_for(queue.get(), timeout=3.0)

                    if msg_type == "done":
                        break
                    elif msg_type == "error":
                        yield f"data: [LOG]{data}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    elif msg_type == "event":
                        event = data
                        for node_name, output in event.items():
                            is_dict = isinstance(output, dict) if output is not None else False

                            if is_dict and "final_answer" in output:
                                current_final_answer = output["final_answer"]

                            # dashboard_select 노드에서 대시보드 데이터 캡처
                            if node_name == "dashboard_select" and is_dict:
                                d = output.get("dashboard_data")
                                if d and isinstance(d, dict) and d.get("charts"):
                                    current_dashboard_data = d

                            agent_name = node_kor_name.get(node_name, node_name)

                            if node_name == "clarify" and is_dict and output.get("need_clarification"):
                                # 역질문 이벤트 즉시 전송
                                import json as _json
                                payload = {
                                    "question": output.get("clarify_question", ""),
                                    "options": output.get("clarify_options", []),
                                    "multi_select": output.get("clarify_multi_select", False),
                                }
                                yield f"data: [CLARIFY]{_json.dumps(payload, ensure_ascii=False)}\n\n"
                                await asyncio.sleep(0.01)
                            elif node_name == "critic":
                                critic_result = output.get("critic_feedback") if is_dict else None
                                if critic_result == "PASS":
                                    yield f"data: [LOG]✅ [Critic] 검수 통과! 완벽한 답변입니다.\n\n"
                                else:
                                    safe_feedback = str(critic_result or "").replace('\n', ' ')
                                    yield f"data: [LOG]❌ [Critic] 답변 반려 및 재작성 지시: {safe_feedback}\n\n"
                            else:
                                yield f"data: [LOG]🟢 [{agent_name}] 작전 수행 완료.\n\n"

                            await asyncio.sleep(0.01)

                            # 💡 여기도 안전 검증을 수행합니다.
                            critic_result = output.get("critic_feedback") if is_dict else None
                            if node_name == "critic" and critic_result == "PASS":
                                # 대시보드 데이터가 있으면 텍스트보다 먼저 전송
                                if current_dashboard_data:
                                    import json as _json
                                    yield f"data: [DASHBOARD]{_json.dumps(current_dashboard_data, ensure_ascii=False)}\n\n"
                                    await asyncio.sleep(0.01)
                                for char in current_final_answer:
                                    yield f"data: {char}\n\n"
                                    await asyncio.sleep(0.01)
                            elif node_name == "greeting" and is_dict and "final_answer" in output:
                                for char in current_final_answer:
                                    yield f"data: {char}\n\n"
                                    await asyncio.sleep(0.01)

                except asyncio.TimeoutError:
                    yield ": keep-alive ping\n\n"

        finally:
            task.cancel()

        latency = round(time.time() - start_time, 2)
        estimated_tokens = int((len(body.query) + len(current_final_answer)) * 0.8)

        yield f"data: [LOG]소요 시간: {latency}초 | 소모 토큰: {estimated_tokens}\n\n"
        await asyncio.sleep(0.05)
        yield f"data: [LOG]모든 에이전트 응답 완료.\n\n"

        if current_final_answer and body.session_id:
            asyncio.create_task(save_message_to_backend(
                session_id=body.session_id,
                user_id=user_id,
                sender_type="LOCAL_AI",
                sender_name="Dati",
                content=current_final_answer,
                is_private=False,
                latency=latency,
                tokens=estimated_tokens,
                bearer_token=bearer_token,
            ))

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


# --------------------------------------------------------------------------
# 프롬프트 자동 최적화
# --------------------------------------------------------------------------
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

    with open("added_rules.txt", "a", encoding="utf-8") as f:
        f.write(f"{new_rule}\n")

    print(f"✨ [진화 완료] 시스템 신규 누적 규칙 각인: {new_rule}")
    return {"status": "success", "new_rule": new_rule}