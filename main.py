from fastapi import BackgroundTasks,UploadFile, File, FastAPI, Depends, HTTPException, status, Form
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
import PyPDF2
import io
import uuid
from pydantic import BaseModel

# 💡 LangGraph 앱 임포트 (router.py에서 정의한 workflow)
from ai_core.router import langgraph_app
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # 💡 1. CORS 도구 불러오기

app = FastAPI(title="Dativus AI Core API")

# 💡 2. 리액트(브라우저)의 접근을 전면 허용하는 방어막 해제 코드!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 주소 허용 (실전에서는 "http://localhost:5173" 등으로 제한)
    allow_credentials=True,
    allow_methods=["*"],  # GET, POST, OPTIONS 등 모든 무전 방식 허용
    allow_headers=["*"],  # 모든 헤더(토큰 등) 허용
)

# ... 기존 코드들 ...



class ChatRequest(BaseModel):
    query: str


# .env 파일 로드
load_dotenv()

# 1. 임베딩 모델 로딩
print("임베딩 모델(BAAI/bge-m3) 로딩 중...")
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로딩 완료!")

# 2. 보안 검색대 세팅
security = HTTPBearer()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")


# 🛡️ 3. JWT 검증 함수 (v4.0: user_id 추출 로직 강화)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        # 💡 토큰에 user_id가 반드시 포함되어 있어야 합니다.
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


# --- 실전 채팅 API (v4.0 페르소나 반영) ---

@app.post("/api/v1/chat")
async def chat_with_ai(
        request: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    # 💡 토큰에서 유저 정보와 워크스페이스 정보를 추출
    user_id = token_payload.get("user_id")
    workspace_id = token_payload.get("workspace_id")
    print(f"\n==================================================================")
    print(f"[채팅 요청] 유저: {user_id} | 워크스페이스: {workspace_id}")
    print(f"\n==================================================================")

    # 💡 LangGraph 호출 시 user_id를 넘겨주어 페르소나를 불러오게 합니다.
    result = langgraph_app.invoke({
        "query": request.query,
        "workspace_id": workspace_id,
        "user_id": user_id
    })

    return {
        "status": "success",
        "query": request.query,
        "answer": result.get("final_answer")  # 💡 final_answer 로 변경!
    }


# --- 실전 스트리밍(SSE) 채팅 API (v4.0 페르소나 반영) ---

@app.post("/api/v1/chat/stream")
async def chat_with_ai_stream(
        request: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    user_id = token_payload.get("user_id")
    workspace_id = token_payload.get("workspace_id")
    print(f"\n==================================================================")
    print(f"[스트리밍 요청] 유저: {user_id} | 워크스페이스: {workspace_id}")
    print(f"\n==================================================================")
    async def event_generator():
        inputs = {
            "query": request.query,
            "workspace_id": workspace_id,
            "user_id": user_id
        }

        # 💡 LangGraph의 비동기 스트림(astream)을 사용하여 페르소나가 반영된 답변 출력
        async for event in langgraph_app.astream(inputs):
            for node_name, output in event.items():
                if "final_answer" in output:  # 💡 final_answer 로 변경!
                    final_text = output["final_answer"]
                    # 타자기 효과를 위해 한 글자씩 전송
                    for char in final_text:
                        yield f"data: {char}\n\n"
                        await asyncio.sleep(0.01)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# --- 문서 업로드 API (기존 로직 유지) ---

@app.post("/api/v1/documents/upload")
async def upload_document(
        background_tasks: BackgroundTasks,  # 🌟 핵심: 백그라운드 작업자 고용
        document_id: str = Form(...),       # 💡 스프링이 꼬리표로 달아준 문서 ID 받기
        file: UploadFile = File(...),
        token_payload: dict = Depends(verify_token)
):
    workspace_id = token_payload.get("workspace_id")

    # 1. 파일 임시 저장 (이건 1초도 안 걸립니다)
    UPLOAD_DIR = "temp_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 2. 백그라운드 작업자에게 중노동(26초짜리)을 지시하고 우리는 빠집니다!
    background_tasks.add_task(process_and_store_document, file_path, file.filename, workspace_id, document_id)

    # 3. 스프링(우체부)에게는 "접수 완료!" 라고 1초 만에 바로 응답 발사!
    return {
        "status": "success",
        "message": f"'{file.filename}' 파일 접수 완료! 백그라운드에서 AI 분석을 시작합니다."
    }


# =====================================================================
# 💡 [새로 추가된 함수] 26초 걸리는 중노동을 대신 해줄 '백그라운드 작업자'
# =====================================================================
def process_and_store_document(file_path: str, filename: str, workspace_id: str, document_id: str):
    try:
        print(f"[백그라운드] '{filename}' 파일 AI 뇌 각인 시작...")

        # 1. 파일 읽기
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        else:
            return

        # 2. 텍스트 쪼개기
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)

        # 3. 임베딩 및 ChromaDB 저장
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

        # 4. 스프링 서버로 "완료(DONE)" 무전 발송!
        webhook_url = "http://127.0.0.1:8080/api/v1/documents/webhook"
        requests.post(webhook_url, json={"documentId": document_id, "status": "DONE"})

    except Exception as e:
        print(f"[백그라운드] 에러 발생: {e}")
        # 실패하면 "실패(FAILED)" 무전 발송
        requests.post("http://127.0.0.1:8080/api/v1/documents/webhook",
                      json={"documentId": document_id, "status": "FAILED"})

    finally:
        # 5. 다 쓴 임시 파일은 깨끗하게 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

# --- 지식망 문서 삭제 (망각 API) ---
@app.delete("/api/v1/documents")
async def delete_document_vectors(workspace_id: str, file_name: str):
    try:
        # 💡 [핵심] ChromaDB에서 '해당 방 번호'와 '해당 파일명'을 가진 조각들을 싹 다 찾아냅니다.
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