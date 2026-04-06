# 🥕 Dativus (다티부스) - AI Routing Server

> **LangGraph 기반 멀티 에이전트 라우팅 챗봇 시스템**
> 본 저장소는 다티부스 프로젝트의 `지능형 라우팅 및 LLM 추론`을 담당하는 AI 코어 파트입니다.

## 📖 목차
1. [프로젝트 개요](#-프로젝트-개요)
2. [시스템 아키텍처](#-시스템-아키텍처)
3. [주요 기능](#-주요-기능)
4. [기술 스택](#-기술-스택)
5. [시작하기 (Getting Started)](#-시작하기-getting-started)
6. [팀 멤버 (Team)](#-팀-멤버-team)

---

## 💡 프로젝트 개요
다티부스(Dativus)는 사용자의 맥락을 이해하고 최적의 답변을 생성하는 개인화된 '제2의 뇌' AI 챗봇 시스템입니다. 
단순한 단일 LLM 호출을 넘어, LangGraph를 활용한 다중 에이전트(Multi-Agent) 협업 및 라우팅 구조를 통해 할루시네이션을 최소화하고 응답의 퀄리티를 극대화합니다.

## 🏗 시스템 아키텍처
*(추후 아키텍처 다이어그램 업데이트 예정)*
* **Frontend:** React 기반 UI/UX 및 실시간 채팅 워크스페이스
* **Main Backend:** Spring Boot 기반 인증(Zero-Trust) 및 비즈니스 로직, PostgreSQL DB
* **AI Core (현재 저장소):** FastAPI 기반 LangGraph 라우팅, 외부 도메인 데이터 연동 및 프롬프트 처리

## ✨ 주요 기능
* **지능형 에이전트 라우팅:** LangGraph를 통해 사용자의 의도를 파악하고 가장 적합한 특화 에이전트에게 작업을 동적으로 할당합니다.
* **RAG 기반 도메인 지식 연동:** 벡터 DB(ChromaDB)를 활용해 특정 도메인(예: 식재료) 데이터를 검색하고 팩트 기반의 답변을 생성합니다.
* **할루시네이션 통제 파이프라인:** 에이전트 간의 교차 검증 및 프롬프트 엔지니어링을 통해 AI의 환각 현상을 억제합니다.

## 🛠 기술 스택
### AI Core (Backend)
* **Framework:** FastAPI
* **Language:** Python 3.11
* **AI/LLM:** LangChain, LangGraph, OpenAI API
* **Database:** ChromaDB (Vector DB)

## 🚀 시작하기 (Getting Started)
본 프로젝트를 로컬 환경에서 실행하기 위한 가이드입니다.

### 1. 환경 설정 및 복제
```bash
$ git clone [https://github.com/Team-Carrot/dativus-ai-router.git](https://github.com/Team-Carrot/dativus-ai-router.git)
$ cd dativus-ai-router
```

### 2. 패키지 설치
```bash
$ pip install -r requirements.txt
```

### 3. 환경 변수 설정 (.env)
루트 디렉토리에 `.env` 파일을 생성하고 아래의 API 키를 세팅하세요.
```text
OPENAI_API_KEY="your_openai_api_key_here"
```

### 4. 서버 실행
```bash
$ uvicorn main:app --reload
```

## 👥 팀 멤버 (Team)
| 역할 | 이름 | 담당 업무 | GitHub |
| --- | --- | --- | --- |
| **Team Leader** | 강동균 | AI 코어 개발, 인프라 구축, 3-Tier 라우팅 로직 | [@GitHub아이디](링크) |
| **Backend** | 김성원 | Spring Boot 인증 로직, DB 스키마 및 세션 관리 | [@GitHub아이디](링크) |
| **Frontend** | 고결 | React UI/UX, 대시보드 및 페르소나 설정 화면 구현 | [@GitHub아이디](링크) |
