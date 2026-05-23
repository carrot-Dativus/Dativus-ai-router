import chromadb
import os

# 1. DB가 저장될 실제 경로 설정
# (현재는 프로젝트 폴더 내부에 chroma_storage라는 폴더를 자동 생성하여 저장합니다.
# 나중에 48GB 외장 SSD를 연결하면 이 경로만 외장하드 경로(예: D:/chroma_storage)로 바꾸면 됩니다!)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DATA_PATH = os.path.join(BASE_DIR, "chroma_storage")

# 2. ChromaDB 클라이언트 초기화 (PersistentClient를 써야 서버가 꺼져도 데이터가 날아가지 않습니다)
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# 3. v2.0 명세서에 지정된 공통 컬렉션 이름 사용
COLLECTION_NAME = "team_knowledge_base"

# 4. 컬렉션(바구니) 가져오기, 만약 없다면 새로 만듭니다.
# 메타데이터 검색에 최적화되도록 세팅합니다.
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"} # 유사도 검색 방식: 코사인 유사도
)

print(f"[ChromaDB] 초기화 완료 (저장 경로: {CHROMA_DATA_PATH})")
print(f"[ChromaDB] 사용 중인 컬렉션: '{COLLECTION_NAME}'")