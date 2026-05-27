# database/postgres.py

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()


def get_user_persona(user_id: str) -> dict:
    """PostgreSQL에 접속해서 유저의 페르소나 정보를 가져오는 함수"""
    try:
        # 환경변수에서 DB 정보 가져오기 (없으면 기본값 사용)
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "dativus_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        cursor = conn.cursor()

        # UUID로 페르소나 컬럼 조회
        cursor.execute("""
            SELECT persona_decision_style, persona_expertise, persona_tone 
            FROM users 
            WHERE id = %s
        """, (user_id,))

        row = cursor.fetchone()
        conn.close()

        if row and (row[0] or row[1] or row[2]):
            return {
                "decision_style": row[0] or "일반적인",
                "expertise": row[1] or "기본",
                "tone": row[2] or "친절한"
            }
        return None  # 페르소나 설정이 없는 유저

    except Exception as e:
        print(f"🚨 DB 페르소나 조회 실패: {e}")
        return None