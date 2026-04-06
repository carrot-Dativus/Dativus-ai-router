from fastapi import UploadFile, File
import shutil
import os
from fastapi.responses import StreamingResponse
import asyncio
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from database.chroma_manager import collection

from fastapi import File, UploadFile, Form
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import PyPDF2
import io
import uuid

from pydantic import BaseModel
from ai_core.router import app as langgraph_app  # FastAPI의 app과 이름이 안 겹치게 변경
from ai_core.router import collection, model

class ChatRequest(BaseModel):
    query: str

# .env 파일 로드 (방금 설치한 라이브러리가 여기서 활약합니다!)
load_dotenv()

app = FastAPI(title="Dativus AI Core API")

# 1. 기존 퀘스트에서 만든 로컬 임베딩 모델 로딩
print("임베딩 모델(BAAI/bge-m3) 로딩 중...")
model = SentenceTransformer('BAAI/bge-m3')
print("모델 로딩 완료!")

# 2. 보안 검색대 세팅
security = HTTPBearer()
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM")

# 🛡️ 3. JWT 검증 함수 (이 문을 통과해야만 채팅 가능!)
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # 프론트엔드가 보낸 토큰을 비밀키로 열어봅니다.
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="유효하지 않은 토큰입니다."
        )

# --- API 엔드포인트들 ---

@app.get("/")
def read_root():
    return {"message": "Dativus AI Core (FastAPI)가 정상 작동 중입니다! (Port 8000)"}

@app.get("/api/v1/embed-test")
def test_embedding(text: str = "이것은 테스트 문장입니다."):
    embedding = model.encode(text).tolist()
    return {
        "original_text": text,
        "vector_length": len(embedding),
        "sample_vector": embedding[:5]
    }

# 🛡️ 4. 보호받는 보안 검색대 테스트 API
@app.get("/api/v1/secure-test")
async def secure_test(user_info: dict = Depends(verify_token)):
    return {
        "message": "보안 검색대 통과 완료! 인증된 사용자입니다.",
        "your_info": user_info
    }


# 5. 내부 AI 임베딩 API (Spring Boot가 파일을 여기로 보냅니다!)
@app.post("/internal/ai/embed")
async def embed_document(
        workspace_id: str = Form(...),
        author: str = Form(...),
        file: UploadFile = File(...)
):
    # 1) 파일 텍스트 추출 (Extract) - TXT와 PDF 지원
    text_content = ""
    file_extension = file.filename.split(".")[-1].lower()

    content = await file.read()

    if file_extension == "txt":
        text_content = content.decode("utf-8")
    elif file_extension == "pdf":
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in pdf_reader.pages:
            if page.extract_text():
                text_content += page.extract_text() + "\n"
    else:
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다. (TXT, PDF만 가능)")

    # 2) 텍스트 청킹 (Transform) - 의미 단위로 조각내기
    # 너무 길면 AI가 까먹고, 너무 짧으면 맥락이 끊기므로 500글자씩 자르고 50글자씩 겹치게 만듭니다.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text_content)

    if not chunks:
        return {"status": "failed", "message": "텍스트를 추출하지 못했거나 빈 파일입니다."}

    # 3) 벡터 변환 및 메타데이터 세팅 (Load 준비)
    print(f"총 {len(chunks)}개의 조각으로 나누었습니다. 임베딩을 시작합니다...")
    embeddings = model.encode(chunks).tolist()

    # 조각마다 고유 ID와 꼬리표(메타데이터)를 달아줍니다. (팀별 데이터 격리의 핵심!)
    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "workspace_id": workspace_id,
            "author": author,
            "source_type": file_extension,
            "file_name": file.filename
        }
        for _ in chunks
    ]

    # 4) ChromaDB에 최종 적재!
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas
    )

    print("✅ ChromaDB 저장 완료!")

    # 명세서 v3.0 응답 양식 준수
    return {
        "status": "success",
        "message": "문서 벡터화 및 ChromaDB 저장 완료",
        "chunks_processed": len(chunks)
    }


# 6. 내부 AI 검색(Retrieval) API - 질문에 맞는 지식 찾아오기
@app.get("/internal/ai/search")
async def search_knowledge(
        workspace_id: str,
        query: str,
        top_k: int = 3
):
    print(f"[{workspace_id}] 검색 요청: '{query}'")

    # 1) 사용자의 질문도 숫자로 변환 (벡터화)
    query_embedding = model.encode(query).tolist()

    # 2) ChromaDB에서 유사도(코사인 거리)가 가장 높은 데이터 쏙 뽑아오기
    # where 조건절을 써서 우리 팀(workspace_id) 데이터만 격리해서 검색합니다!
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"workspace_id": workspace_id}
    )

    # 3) 찾은 데이터가 없으면 빈 배열 반환
    if not results['documents'] or not results['documents'][0]:
        return {"status": "success", "message": "관련된 문서를 찾을 수 없습니다.", "results": []}

    # 4) 찾은 문서 조각들과 메타데이터를 예쁘게 정리해서 반환
    fetched_data = []
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        fetched_data.append({
            "content": doc,
            "metadata": meta
        })

    return {
        "status": "success",
        "query": query,
        "results": fetched_data
    }


# 7. 실전 채팅 API (LangGraph 뇌 + JWT 보안 검색대 결합!)
@app.post("/api/v1/chat")
async def chat_with_ai(
        request: ChatRequest,
        # 👇 여기서 보안 검색대를 통과해야만 아래 코드가 실행됩니다!
        token_payload: dict = Depends(verify_token)
):
    # 보안 검색대에서 압수(...)한 토큰에서 지휘관님의 진짜 신분증을 꺼냅니다.
    workspace_id = token_payload.get("workspace_id")

    print(f"💬 [채팅 요청] 워크스페이스: {workspace_id} | 질문: {request.query}")

    # 아까 만든 똑똑한 LangGraph 뇌로 질문과 신분증을 넘깁니다!
    result = langgraph_app.invoke({
        "query": request.query,
        "workspace_id": workspace_id
    })

    # 뇌가 고민해서 뱉어낸 최종 답변을 프론트엔드에게 돌려줍니다.
    return {
        "status": "success",
        "query": request.query,
        "answer": result["final_answer"]
    }


# 8. 실전 스트리밍(SSE) 채팅 API - 타자기 효과!
@app.post("/api/v1/chat/stream")
async def chat_with_ai_stream(
        request: ChatRequest,
        token_payload: dict = Depends(verify_token)
):
    workspace_id = token_payload.get("workspace_id")
    print(f"💬 [스트리밍 요청] 워크스페이스: {workspace_id} | 질문: {request.query}")

    # 💡 데이터 방출기(Generator) 함수: 프론트엔드로 조각을 계속 던져줍니다.
    async def event_generator():
        # 1. 뇌(LangGraph)를 깨워서 답변을 가져옵니다.
        result = langgraph_app.invoke({
            "query": request.query,
            "workspace_id": workspace_id
        })
        final_text = result["final_answer"]

        # 2. 타자기 효과 (SSE 규격에 맞춰서 한 글자씩 쏩니다!)
        for char in final_text:
            yield f"data: {char}\n\n"
            await asyncio.sleep(0.02)  # 0.02초 간격으로 전송 (속도 조절 가능)

        # 3. 모든 전송이 끝났음을 프론트엔드에 알립니다.
        yield "data: [DONE]\n\n"

    # 일반 JSON이 아니라, '스트리밍 모드(text/event-stream)'로 응답을 내보냅니다!
    return StreamingResponse(event_generator(), media_type="text/event-stream")


# 9. 지식 자동 적재 파이프라인 (ETL: Extract, Transform, Load)
@app.post("/api/v1/documents/upload")
async def upload_document(
        # 프론트엔드에서 날아오는 파일과 출입증(토큰)을 동시에 받습니다.
        file: UploadFile = File(...),
        token_payload: dict = Depends(verify_token)
):
    workspace_id = token_payload.get("workspace_id")

    # 1️⃣ Extract (추출): 파일을 서버의 임시 폴더에 저장
    UPLOAD_DIR = "temp_uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📄 [ETL 1단계] 파일 임시 저장 완료: {file.filename}")

    # 2️⃣ Transform (변환): PDF나 TXT를 읽고 AI가 소화하기 좋게 쪼개기(Chunking)
    try:
        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file.filename.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            documents = loader.load()
        else:
            return {"status": "error", "message": "PDF나 TXT 파일만 지원합니다."}

        # AI가 읽기 딱 좋은 크기(500자)로 문서를 예쁘게 자릅니다.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(documents)
        print(f"✂️ [ETL 2단계] 총 {len(chunks)}개의 조각(Chunk)으로 분할 완료!")

        # 3️⃣ Load (적재): 쪼개진 조각들을 임베딩(수치화)해서 ChromaDB에 꽂아넣기
        ids = []
        embeddings = []
        metadatas = []
        documents_text = []

        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())  # 고유 주민번호 부여
            text = chunk.page_content

            ids.append(chunk_id)
            documents_text.append(text)
            embeddings.append(model.encode(text).tolist())  # 수학적 벡터로 변환!
            metadatas.append({
                "workspace_id": workspace_id,
                "file_name": file.filename,
                "chunk_index": i
            })

        # ChromaDB 창고에 한 방에 밀어넣습니다!
        collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents_text
        )
        print(f"💾 [ETL 3단계] ChromaDB 적재 완료!")

    finally:
        # 작업이 끝났으니 서버 용량 확보를 위해 임시 파일은 삭제합니다.
        if os.path.exists(file_path):
            os.remove(file_path)

    return {
        "status": "success",
        "message": f"'{file.filename}' 파일이 AI 뇌에 성공적으로 저장되었습니다!",
        "chunks_saved": len(chunks)
    }