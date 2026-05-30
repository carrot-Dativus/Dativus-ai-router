"""
Dativus 5대 메트릭 측정 스크립트
-------------------------------
메트릭 1 : 라우팅 정확도      — 예상 에이전트 vs 실제 라우팅 일치율
메트릭 2 : 응답 시간 비교     — 단순 LLM.invoke() vs 전체 파이프라인
메트릭 3 : Critic 1차 통과율  — revision_count == 0 비율
메트릭 4 : LLM-as-Judge 품질  — 단순 LLM vs Dativus 점수 비교
메트릭 5 : Groq→Ollama 폴백률 — _invoke_with_backoff 카운터 기반
"""

import os, json, re, time
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

load_dotenv()

# 공통 설정
WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
USER_ID      = "22222222-2222-2222-2222-222222222222"

# eval 환경에서 Groq SSL 이슈로 Ollama를 비교 기준으로 사용
# (파이프라인도 Ollama 폴백으로 동작 중이므로 비교 공정함)
judge_llm  = ChatOllama(model="llama3", temperature=0, num_predict=1500)
single_llm = ChatOllama(model="llama3", temperature=0, num_predict=2000)

# ──────────────────────────────────────────────
# 테스트 데이터셋
# ──────────────────────────────────────────────

# 메트릭 1: 라우팅 정확도 — (쿼리, 예상 에이전트)
ROUTING_CASES = [
    ("안녕하세요!",                             "general_agent"),
    ("방금 내가 뭐 물어봤지?",                   "general_agent"),
    ("인공지능이 뭐야?",                         "general_agent"),
    ("오늘 기분 어때요?",                        "general_agent"),
    ("React랑 Vue 중 어떤 게 나아?",             "expert_agent"),
    ("마이크로서비스 vs 모놀리식 장단점 비교해줘", "expert_agent"),
    ("스타트업 6개월 로드맵 짜줘",                "expert_agent"),
    ("파이썬 리스트 인덱스 찾는 법",              "coding_math_agent"),
    ("TypeError 에러 고쳐줘",                    "coding_math_agent"),
    ("피보나치 수열 재귀 함수 구현해줘",           "coding_math_agent"),
]

# 메트릭 2,3 공용: 파이프라인 실행 쿼리 (에이전트 유형별 1개씩)
PIPELINE_CASES = [
    "React랑 Vue 중 어떤 게 프로젝트에 더 적합해?",
    "파이썬 버블 정렬 코드 짜줘",
    "안녕! 오늘 뭐 도와줄 수 있어?",
]

# 메트릭 4: LLM-as-Judge 비교 쿼리 + 평가 기준
JUDGE_CASES = [
    {
        "query": "프론트엔드 상태 관리를 어떻게 하는 게 좋을까?",
        "criteria": "3가지 대안(A/B/C)을 제시하고, 각 장단점을 비교한 뒤 사용자의 선택을 물어봐야 한다.",
    },
    {
        "query": "안녕! 오늘 날씨가 어때?",
        "criteria": "단순 인사이므로 A/B/C 구조 없이 가볍고 친절하게 대답해야 한다.",
    },
    {
        "query": "우리 팀이 어제 결정한 데이터베이스 마이그레이션 전략 요약해줘.",
        "criteria": "사내 검색을 시도해야 하며, 데이터가 없다면 솔직하게 없다고 대답해야 한다.",
    },
]


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────

def _invoke_pipeline(query: str) -> dict:
    from ai_core.router import langgraph_app
    return langgraph_app.invoke({
        "query": query,
        "workspace_id": WORKSPACE_ID,
        "user_id": USER_ID,
    })


def _judge(query: str, answer: str, criteria: str) -> dict:
    """LLM에게 100점 만점 채점 요청 → {"score": int, "reason": str}"""
    prompt = f"""AI 답변의 품질을 평가하세요.
[사용자 질문]: {query}
[AI 답변]: {answer[:600]}
[평가 기준]: {criteria}
JSON만 출력: {{"score": 90, "reason": "1-2줄 이유"}}"""
    try:
        raw = judge_llm.invoke(prompt).content.strip()
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except Exception as e:
        print(f"    [Judge 오류] {e}")
    return {"score": 0, "reason": "채점 실패"}


def _sep(title: str = ""):
    print("=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


# ──────────────────────────────────────────────
# 메트릭 1: 라우팅 정확도
# ──────────────────────────────────────────────

def run_metric1_routing():
    _sep("메트릭 1 — 라우팅 정확도")
    from ai_core.router import langgraph_app

    correct, total = 0, len(ROUTING_CASES)
    wrong_cases = []

    for query, expected in ROUTING_CASES:
        result = langgraph_app.invoke({
            "query": query,
            "workspace_id": WORKSPACE_ID,
            "user_id": USER_ID,
        })
        actual = result.get("target_agent_name", "unknown")
        ok = (actual == expected)
        if ok:
            correct += 1
            print(f"  [OK] [{expected:22s}]  {query[:40]}")
        else:
            wrong_cases.append((query, expected, actual))
            print(f"  [FAIL] [예상:{expected:15s} / 실제:{actual:15s}]  {query[:40]}")

    acc = correct / total * 100
    print(f"\n  결과: {correct}/{total}  →  정확도 {acc:.1f}%")
    if wrong_cases:
        print("  [오답 목록]")
        for q, exp, act in wrong_cases:
            print(f"    Q: {q}  |  예상: {exp}  |  실제: {act}")
    return acc


# ──────────────────────────────────────────────
# 메트릭 2: 응답 시간 비교 (단일 LLM vs 파이프라인)
# ──────────────────────────────────────────────

def run_metric2_latency():
    _sep("메트릭 2 — 응답 시간 비교")

    single_times, pipe_times = [], []

    for query in PIPELINE_CASES:
        print(f"\n  Q: {query[:50]}")

        # 단일 LLM
        t0 = time.perf_counter()
        single_llm.invoke(query)
        t_single = time.perf_counter() - t0
        single_times.append(t_single)
        print(f"    단일 LLM  : {t_single:.2f}s")

        # 전체 파이프라인
        t0 = time.perf_counter()
        _invoke_pipeline(query)
        t_pipe = time.perf_counter() - t0
        pipe_times.append(t_pipe)
        overhead = t_pipe - t_single
        print(f"    파이프라인: {t_pipe:.2f}s  (오버헤드 +{overhead:.2f}s)")

    avg_s = sum(single_times) / len(single_times)
    avg_p = sum(pipe_times) / len(pipe_times)
    print(f"\n  평균 단일 LLM  : {avg_s:.2f}s")
    print(f"  평균 파이프라인: {avg_p:.2f}s  (×{avg_p/avg_s:.1f})")
    return avg_s, avg_p


# ──────────────────────────────────────────────
# 메트릭 3: Critic 1차 통과율
# ──────────────────────────────────────────────

def run_metric3_critic():
    _sep("메트릭 3 — Critic 1차 통과율")

    first_pass, total = 0, len(PIPELINE_CASES)

    for query in PIPELINE_CASES:
        result = _invoke_pipeline(query)
        rev_count = result.get("revision_count", 0)
        passed_first = (rev_count == 0)
        if passed_first:
            first_pass += 1
        icon = "[PASS]" if passed_first else f"[REVISION] revision_count={rev_count}"
        print(f"  {icon}  |  {query[:50]}")

    rate = first_pass / total * 100
    print(f"\n  결과: {first_pass}/{total}  →  1차 통과율 {rate:.1f}%")
    return rate


# ──────────────────────────────────────────────
# 메트릭 4: LLM-as-Judge — 단일 LLM vs Dativus 품질 비교
# ──────────────────────────────────────────────

def run_metric4_judge():
    _sep("메트릭 4 — LLM-as-Judge 품질 비교")

    single_total, dativus_total = 0, 0

    for case in JUDGE_CASES:
        query    = case["query"]
        criteria = case["criteria"]
        print(f"\n  Q: {query[:50]}")

        # 단일 LLM 답변
        single_ans = single_llm.invoke(query).content
        single_eval = _judge(query, single_ans, criteria)

        # Dativus 파이프라인 답변
        pipe_result = _invoke_pipeline(query)
        dativus_ans = pipe_result.get("final_answer", "")
        dativus_eval = _judge(query, dativus_ans, criteria)

        single_total  += single_eval["score"]
        dativus_total += dativus_eval["score"]

        print(f"    단일 LLM  점수: {single_eval['score']:3d}/100  |  {single_eval['reason'][:60]}")
        print(f"    Dativus   점수: {dativus_eval['score']:3d}/100  |  {dativus_eval['reason'][:60]}")

    n = len(JUDGE_CASES)
    avg_s = single_total / n
    avg_d = dativus_total / n
    diff = avg_d - avg_s
    print(f"\n  평균 단일 LLM: {avg_s:.1f}/100")
    print(f"  평균 Dativus : {avg_d:.1f}/100  ({'+' if diff >= 0 else ''}{diff:.1f}점)")
    return avg_s, avg_d


# ──────────────────────────────────────────────
# 메트릭 5: Groq→Ollama 폴백 비율
# ──────────────────────────────────────────────

def run_metric5_fallback():
    _sep("메트릭 5 — Groq→Ollama 폴백 비율")
    from ai_core import metrics as _m

    stats = _m.get_stats()
    total = stats["total_llm_calls"]
    fb    = stats["fallback_calls"]
    ratio = stats["fallback_ratio"] * 100

    print(f"  총 LLM 호출 (누적): {total}")
    print(f"  Ollama 폴백 발생  : {fb}")
    print(f"  폴백 비율         : {ratio:.1f}%")

    if total == 0:
        print("  [INFO] 메트릭 1~4 실행 후 호출됩니다. 정상입니다.")
    elif ratio == 0:
        print("  [OK] Groq API 정상 운영 중 (폴백 없음)")
    elif ratio < 10:
        print(f"  [WARN] 소수 폴백 발생 ({fb}회) — Rate Limit 주의")
    else:
        print(f"  [ALERT] 폴백 빈번 ({ratio:.1f}%) — Groq 한도 확인 필요")

    return ratio


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def run_all():
    print("\n")
    _sep("Dativus 5대 메트릭 평가 시작")
    print()

    from ai_core import metrics as _m
    _m.reset()  # 이전 세션 카운터 초기화

    results = {}

    print()
    results["routing_acc"] = run_metric1_routing()

    print()
    results["latency_single"], results["latency_pipe"] = run_metric2_latency()

    print()
    results["critic_rate"] = run_metric3_critic()

    print()
    results["judge_single"], results["judge_dativus"] = run_metric4_judge()

    print()
    results["fallback_ratio"] = run_metric5_fallback()

    # 최종 요약
    print()
    _sep("최종 요약")
    print(f"  메트릭 1  라우팅 정확도   : {results['routing_acc']:.1f}%")
    print(f"  메트릭 2  파이프라인 지연 : {results['latency_pipe']:.2f}s  (단일 LLM {results['latency_single']:.2f}s 대비 ×{results['latency_pipe']/results['latency_single']:.1f})")
    print(f"  메트릭 3  Critic 통과율   : {results['critic_rate']:.1f}%")
    print(f"  메트릭 4  품질 향상       : {results['judge_single']:.1f}→{results['judge_dativus']:.1f}점 ({'+' if results['judge_dativus']>=results['judge_single'] else ''}{results['judge_dativus']-results['judge_single']:.1f}점)")
    print(f"  메트릭 5  Groq 폴백 비율  : {results['fallback_ratio']:.1f}%")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all()
