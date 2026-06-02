import sys
import os
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi import BackgroundTasks, UploadFile, File, FastAPI, Depends, HTTPException, status, Form, Request
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import re
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
from ai_core.nodes import pop_pending_logs
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
    force_agent: Optional[str] = None          # 빌트인 강제 라우팅 (general_agent / expert_agent / coding_math_agent)
    target_agent_name: Optional[str] = None    # 커스텀 에이전트 이름 (수동 선택)
    target_agent_prompt: Optional[str] = None  # 커스텀 에이전트 성격/역할 (수동 선택)
    custom_agents_list: Optional[list] = []    # 자동 매칭용 전체 커스텀 에이전트 목록
    existing_dashboard: Optional[dict] = None
    # Phase 1 개인화: 드롭다운 3개 + 자유 입력 메모
    persona_expertise: Optional[str] = ""       # 전문 분야
    persona_tone: Optional[str] = ""            # 대화 어조
    persona_decision_style: Optional[str] = ""  # 판단 스타일
    persona_memo: Optional[str] = ""            # 추가 자유 입력 지시문


load_dotenv()

print("임베딩 모델(BAAI/bge-m3) 로딩 중...", flush=True)
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로딩 완료!", flush=True)

security = HTTPBearer()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

if not JWT_SECRET_KEY or not JWT_ALGORITHM:
    raise RuntimeError("JWT_SECRET_KEY, JWT_ALGORITHM 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")


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


def _send_webhook(url: str, document_id: str, status: str, bearer_token: str = ""):
    """Spring 웹훅 호출 — Bearer 토큰 포함으로 403 방지."""
    try:
        headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
        resp = requests.post(url, json={"documentId": document_id, "status": status}, headers=headers, timeout=10)
        print(f"[웹훅] {status} 전송 → HTTP {resp.status_code} | documentId={document_id}")
    except Exception as e:
        print(f"[웹훅 실패] {e}")


def _clean_final_answer(text: str) -> str:
    """스트리밍 직전 — 알려진 프롬프트 오염 패턴을 제거."""
    text = re.sub(r'\[사용자 질문[^\]]*\][^\n]*\n?', '', text)
    text = re.sub(r'\[이전 대화[^\]]*\][^\n]*\n?', '', text)
    text = re.sub(r'\[전 요원[^\]]*\][^\n]*\n?', '', text)
    text = re.sub(r'\[출력 구조[^\]]*\][^\n]*\n?', '', text)
    text = re.sub(r'^\[대화 요원[^\]]*\][^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\[전문가 요원[^\]]*\][^\n]*\n?', '', text, flags=re.MULTILINE)
    # 개인화 스타일 블록 헤더 누출 제거 - LLM이 프롬프트 지시문을 출력하는 경우
    text = re.sub(r'^\[사용자 스타일 설정[^\]]*\][^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\[스타일 참고[^\]]*\][^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^※\s*A/B/C[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^※\s*위 스타일[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\u26a0\ufe0f[^\n]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # llama 모델이 표 셀에 중국어(CJK)를 출력하는 버그 후처리 — 한국어 전용 서비스이므로 제거
    text = re.sub(r'[一-鿿㐀-䶿豈-﫿]+', '', text)
    return text.strip()

async def _stream_answer(text: str):
    """코드 블록은 통째로, 일반 텍스트는 단어 단위로 스트리밍."""
    # ``` 기준으로 코드 블록과 일반 텍스트 분리
    segments = re.split(r'(```[\s\S]*?```)', text)
    for seg in segments:
        if seg.startswith('```'):
            # 코드 블록 — 각 줄을 data: 필드로, 하나의 SSE 이벤트로 전송
            # 프론트 파서가 data: 필드들을 \n으로 합쳐서 코드 블록 원형 복원
            for line in seg.split('\n'):
                yield f"data: {line}\n"
            yield "\n"  # SSE 이벤트 종료
            await asyncio.sleep(0.005)
        else:
            # 일반 텍스트 — 단어 단위 스트리밍
            for part in re.split(r'(\n)', seg):
                if part == '\n':
                    yield "data: \n\n"
                    await asyncio.sleep(0.003)
                elif part:
                    words = part.split(' ')
                    for i, word in enumerate(words):
                        token = word if i == len(words) - 1 else word + ' '
                        if token:
                            yield f"data: {token}\n\n"
                            await asyncio.sleep(0.008)


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

    async def event_generator():
        # ── 순수 LLM 모드: LangGraph 파이프라인 완전 스킵 ──
        if body.force_agent == "pure_llm":
            from ai_core.nodes import local_llm
            prompt = (
                "반드시 한국어로만 답변하세요. 영어나 다른 언어는 절대 사용하지 마세요.\n\n"
                f"사용자 질문: {body.query}"
            )
            try:
                async for chunk in local_llm.astream(prompt):
                    text = chunk.content if hasattr(chunk, "content") else str(chunk)
                    if text:
                        yield f"data: {text}\n\n"
            except Exception as e:
                yield f"data: [LOG]오류: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
            return

        start_time = time.time()
        inputs = {
            "query": body.query,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "history": body.history,
            "force_agent": body.force_agent or "",
            "custom_agent_name": body.target_agent_name or "",
            "custom_agent_prompt": body.target_agent_prompt or "",
            "custom_agents_list": body.custom_agents_list or [],
            "existing_dashboard": body.existing_dashboard or {},
            "persona_expertise": body.persona_expertise or "",
            "persona_tone": body.persona_tone or "",
            "persona_decision_style": body.persona_decision_style or "",
            "persona_memo": body.persona_memo or "",
        }

        node_kor_name = {
            "supervisor":           "최고 관리자",
            "clarify":              "역질문 처리",
            "conversation_memory":  "대화 기억병",
            "general_agent":        "일반 대화병",
            "code_search":          "레퍼런스 검색병",
            "coding_math_agent":    "코딩/수학 전문병",
            "search_coordinator":   "수색대 출격",
            "search":               "자료 검색병",
            "web_search":           "웹 검색병",
            "graph_memory":         "관계망 추론병",
            "expert_agent":         "전문 분석병",
            "dashboard_select":     "대시보드 분석",
            "summary":              "문서 요약병",
            "critic":               "품질 검수 요원",
            "revision_agent":       "재작성 요원",
            "custom_agent_gate":    "커스텀 에이전트",
            "persona_agent":        "개인화 적용 중",
        }

        queue = asyncio.Queue()
        current_multi_agent_responses = []

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

                            if node_name == "supervisor" and is_dict:
                                target = output.get("target_agent_name", "")
                                if target in ("general_agent", "expert_agent", "coding_math_agent"):
                                    yield f"data: [ROUTE]{target}\n\n"
                                yield f"data: [LOG]🟢 [{agent_name}] 작전 수행 완료.\n\n"
                            elif node_name == "clarify" and is_dict and output.get("need_clarification"):
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
                                    yield f"data: [LOG]✅ [품질 검수 요원] 검수 통과! 완벽한 답변입니다.\n\n"
                                else:
                                    safe_feedback = str(critic_result or "").replace('\n', ' ')
                                    yield f"data: [LOG]❌ [품질 검수 요원] 답변 반려 및 재작성 지시: {safe_feedback}\n\n"
                            elif node_name == "custom_agent_gate" and is_dict:
                                multi = output.get("multi_agent_responses", [])
                                if multi:
                                    current_multi_agent_responses.extend(multi)
                                    names = ", ".join(a["name"] for a in multi)
                                    yield f"data: [LOG]🎭 다중 에이전트 활성화: {names}\n\n"
                                matched = output.get("matched_custom_agent_name", "")
                                if matched:
                                    yield f"data: [LOG]🎭 [{matched}] 관점 추가 완료.\n\n"
                            else:
                                yield f"data: [LOG]🟢 [{agent_name}] 작전 수행 완료.\n\n"

                            await asyncio.sleep(0.01)

                            # 폴백 발생 시 SSE 로그 방출
                            for pending_log in pop_pending_logs():
                                yield f"data: [LOG]{pending_log}\n\n"

                            # 💡 여기도 안전 검증을 수행합니다.
                            critic_result = output.get("critic_feedback") if is_dict else None
                            if node_name == "critic" and critic_result == "PASS":
                                # 대시보드 데이터가 있으면 텍스트보다 먼저 전송
                                if current_dashboard_data:
                                    import json as _json
                                    yield f"data: [DASHBOARD]{_json.dumps(current_dashboard_data, ensure_ascii=False)}\n\n"
                                    await asyncio.sleep(0.01)
                                # 스트리밍 전 경량 오염 클린업
                                answer_to_stream = _clean_final_answer(current_final_answer)
                                async for chunk in _stream_answer(answer_to_stream):
                                    yield chunk
                                # 다중 에이전트 응답 — 각각 별도 버블로 순차 스트리밍
                                for agent_resp in current_multi_agent_responses:
                                    await asyncio.sleep(0.1)
                                    yield f"data: [AGENT_START:{agent_resp['name']}]\n\n"
                                    await asyncio.sleep(0.05)
                                    async for chunk in _stream_answer(_clean_final_answer(agent_resp["response"])):
                                        yield chunk
                                    yield f"data: [AGENT_END]\n\n"
                                    await asyncio.sleep(0.05)
                            elif node_name == "greeting" and is_dict and "final_answer" in output:
                                answer_to_stream = _clean_final_answer(current_final_answer)
                                async for chunk in _stream_answer(answer_to_stream):
                                    yield chunk

                except asyncio.TimeoutError:
                    yield ": keep-alive ping\n\n"

        finally:
            task.cancel()

        latency = round(time.time() - start_time, 2)
        estimated_tokens = int((len(body.query) + len(current_final_answer)) * 0.8)

        yield f"data: [LOG]소요 시간: {latency}초 | 소모 토큰: {estimated_tokens}\n\n"
        await asyncio.sleep(0.05)
        yield f"data: [LOG]모든 에이전트 응답 완료.\n\n"

        # 메시지 저장은 프론트엔드(useChatSession.js)에서 단일 처리 — 여기서 중복 저장하면 재로그인 시 답변 2개 발생

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/v1/documents/upload")
async def upload_document(
        request: Request,
        background_tasks: BackgroundTasks,
        document_id: str = Form(...),
        file: UploadFile = File(...),
        token_payload: dict = Depends(verify_token)
):
    workspace_id = token_payload.get("workspace_id")
    # 업로드 요청의 Bearer 토큰을 백그라운드 태스크에 전달 → 웹훅 인증에 사용
    bearer_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    UPLOAD_DIR = "temp_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(process_and_store_document, file_path, file.filename, workspace_id, document_id, bearer_token)

    return {
        "status": "success",
        "message": f"'{file.filename}' 파일 접수 완료! 백그라운드에서 AI 분석을 시작합니다."
    }


def process_and_store_document(file_path: str, filename: str, workspace_id: str, document_id: str, bearer_token: str = ""):
    try:
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

        _send_webhook(f"{SPRING_BASE_URL}/api/v1/documents/webhook", document_id, "DONE", bearer_token)

    except Exception as e:
        print(f"[백그라운드] 에러 발생: {e}")
        _send_webhook(f"{SPRING_BASE_URL}/api/v1/documents/webhook", document_id, "FAILED", bearer_token)
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