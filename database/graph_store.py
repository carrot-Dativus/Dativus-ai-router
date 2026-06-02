# database/graph_store.py
import re
import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

TRIPLE_RE = re.compile(
    r'\[([^\]]+)\]\s*[→➔\->]+\s*\(([^)]+)\)\s*[→➔\->]+\s*\[([^\]]+)\]'
)


def _connect():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME", "dativus_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
    )


def ensure_table():
    """서버 시작 시 1회 호출 — 테이블/인덱스 생성."""
    conn = _connect()
    try:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_triples (
                    id BIGSERIAL PRIMARY KEY,
                    workspace_id VARCHAR(255) NOT NULL,
                    entity_a TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    entity_b TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (workspace_id, entity_a, relation, entity_b)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_graph_triples_ws "
                "ON graph_triples(workspace_id)"
            )
    finally:
        conn.close()


def save_triples(workspace_id: str, text: str) -> int:
    """LLM 출력 텍스트에서 트리플 파싱 후 DB upsert. 저장된 개수 반환."""
    if not workspace_id or not text:
        return 0
    triples = TRIPLE_RE.findall(text)
    if not triples:
        return 0
    conn = _connect()
    saved = 0
    try:
        with conn:
            for a, r, b in triples:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO graph_triples (workspace_id, entity_a, relation, entity_b)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (workspace_id, entity_a, relation, entity_b) DO NOTHING
                    """,
                    (workspace_id, a.strip(), r.strip(), b.strip()),
                )
                saved += cur.rowcount
    finally:
        conn.close()
    return saved


def load_context(workspace_id: str, limit: int = 20) -> str:
    """워크스페이스의 누적 트리플을 최신순으로 조회 후 텍스트로 반환."""
    if not workspace_id:
        return ""
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT entity_a, relation, entity_b
            FROM graph_triples
            WHERE workspace_id = %s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (workspace_id, limit),
        )
        rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return ""
    return "\n".join(f"[{a}] → ({r}) → [{b}]" for a, r, b in rows)
