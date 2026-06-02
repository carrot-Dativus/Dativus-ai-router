from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
import os

# 🌐 1. 메인 엔진 (외부 API - 고성능/빠른 속도)
# 주로 초안 작성(External LLM)과 1차 판단에 사용됩니다.
external_llm = ChatGroq(
    temperature=0.2, # 관리자와 초안 작성용이므로 약간의 일관성 유지
    model_name="llama3-70b-8192",
    api_key=os.environ.get("GROQ_API_KEY"),
    max_retries=1 # 💡 에러 시 빠르게 Fallback(로컬)으로 넘어가도록 재시도 횟수 제한
)

# 🏠 2. 보조 엔진 (로컬 LLM - 무비용/검수 및 비상용)
# O/X 검수(Critic)와 Groq 한도 초과 시 비상용으로 사용됩니다.
local_llm = ChatOllama(
    model="qwen2.5:14b",
    temperature=0.1
)