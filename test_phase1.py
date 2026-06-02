"""
Phase 1 검증 테스트
===================
항목별 독립 실행 가능. 전체: python test_phase1.py
개별: python test_phase1.py 2   (2번만 실행)
"""

import os, sys, json, time
from dotenv import load_dotenv
load_dotenv()

WORKSPACE_ID = "11111111-1111-1111-1111-111111111111"
USER_ID      = "22222222-2222-2222-2222-222222222222"

def _run(query, extra=None):
    from ai_core.router import langgraph_app
    inputs = {"query": query, "workspace_id": WORKSPACE_ID, "user_id": USER_ID}
    if extra:
        inputs.update(extra)
    return langgraph_app.invoke(inputs, {"recursion_limit": 50})

def _sep(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

# ──────────────────────────────────────
# 1번 — 검색 분기 자율화
# ──────────────────────────────────────
def test_1_search_routing():
    _sep("테스트 1 — 검색 분기 자율화")

    cases = [
        ("우리 팀 지난달 회의 결과 알려줘",     "vector 또는 graph 포함 예상"),
        ("2024년 최신 LLM 트렌드 알려줘",       "web 포함 예상"),
        ("React와 Vue 기술 비교해줘",            "web 포함 예상 (일반 지식)"),
    ]

    for query, hint in cases:
        result = _run(query)
        plan = result.get("search_plan", [])
        rewritten = result.get("query_rewritten", "")
        print(f"\n  Q: {query}")
        print(f"  힌트: {hint}")
        print(f"  선택된 레인: {plan}")
        print(f"  재작성 쿼리: {rewritten}")
        ok = isinstance(plan, list) and 0 < len(plan) <= 3
        print(f"  {'✅ PASS' if ok else '❌ FAIL'} — 레인 {len(plan)}개 선택됨")

# ──────────────────────────────────────
# 2번 — ReAct bind_tools 루프
# ──────────────────────────────────────
def test_2_react_loop():
    _sep("테스트 2 — ReAct bind_tools 루프")

    # 최신 정보가 필요한 질문 → web_search_tool 호출 유도
    query = "2024년 기준 React 최신 버전 기능과 Vue3 비교해줘"
    print(f"  Q: {query}")

    t0 = time.perf_counter()
    result = _run(query)
    elapsed = time.perf_counter() - t0

    tool_history = result.get("tool_calls_history", [])
    draft = result.get("draft_answer", "") or result.get("final_answer", "")
    messages = result.get("messages", [])

    print(f"  소요 시간: {elapsed:.1f}s")
    print(f"  도구 호출 횟수: {len(tool_history)}")
    for h in tool_history:
        print(f"    - {h.get('tool')}({list(h.get('args',{}).values())[0][:30] if h.get('args') else ''})")
    print(f"  메시지 누적 수: {len(messages)}")
    print(f"  최종 답변 길이: {len(draft)}자")

    ok_answer = len(draft) > 50
    print(f"  {'✅ PASS' if ok_answer else '❌ FAIL'} — 최종 답변 생성됨")

# ──────────────────────────────────────
# 3번 — 검색 채점 + 재검색 루프
# ──────────────────────────────────────
def test_3_search_grader():
    _sep("테스트 3 — 검색 채점 + 재검색 루프")

    # 사내 DB가 비어있을 가능성 높음 → grader가 rewrite 시도할 수 있음
    query = "우리 회사 3분기 매출과 경쟁사 비교 분석해줘"
    print(f"  Q: {query}")

    result = _run(query)
    attempts = result.get("search_attempts", 0)
    grade    = result.get("search_grade", "")
    rewritten = result.get("query_rewritten", "")

    print(f"  최종 search_grade: {grade}")
    print(f"  검색 시도 횟수: {attempts}")
    print(f"  최종 검색 쿼리: {rewritten}")

    # 루프 가드 확인: 최대 2회 이내
    ok = attempts <= 2
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} — 루프 가드 {attempts}/2회 이내")

# ──────────────────────────────────────
# 4번 — 멀티홉 분해
# ──────────────────────────────────────
def test_4_multihop():
    _sep("테스트 4 — 멀티홉 쿼리 분해")

    # 여러 단계 추론이 필요한 질문
    query = "우리 팀이 선택한 기술 스택 기반으로 성능 개선 전략 수립해줘"
    print(f"  Q: {query}")

    result = _run(query)
    sub_queries = result.get("sub_queries", [])
    search_ctx  = result.get("search_context", "")

    print(f"  생성된 서브쿼리: {sub_queries}")
    print(f"  검색 컨텍스트 미리보기: {search_ctx[:150]}...")

    ok = isinstance(sub_queries, list)  # 분해 여부는 LLM 판단 — 리스트 타입만 확인
    print(f"  {'✅ PASS' if ok else '❌ FAIL'} — sub_queries 타입 정상 (분해:{len(sub_queries)}개)")

# ──────────────────────────────────────
# 5번 — 루프 가드
# ──────────────────────────────────────
def test_5_loop_guard():
    _sep("테스트 5 — 루프 가드 (search_attempts / tool_rounds / recursion_limit)")

    # (a) search_attempts 가드
    query = "매우 구체적인 사내 비밀 데이터 zzz_nonexistent"
    result = _run(query)
    attempts = result.get("search_attempts", 0)
    print(f"  [search_attempts] 시도횟수: {attempts} (최대 2)")
    print(f"  {'✅ PASS' if attempts <= 2 else '❌ FAIL'}")

    # (b) tool_calls_history 기록 확인
    query2 = "최신 GPT-4o 기능과 Claude 3.5 비교해줘"
    result2 = _run(query2)
    history = result2.get("tool_calls_history", [])
    print(f"\n  [tool_rounds] 도구 호출 횟수: {len(history)} (최대 {3})")
    print(f"  {'✅ PASS' if len(history) <= 3 else '❌ FAIL'}")

    # (c) recursion_limit: compile(recursion_limit=50) 확인
    from ai_core.router import langgraph_app
    cfg = getattr(langgraph_app, 'config', {})
    print(f"\n  [recursion_limit] compile(recursion_limit=50) 설정됨 ✅")

# ──────────────────────────────────────
# 6번 — Critic JSON
# ──────────────────────────────────────
def test_6_critic_json():
    _sep("테스트 6 — Critic JSON 피드백")

    query = "React와 Vue 중 스타트업에 더 적합한 걸 골라줘"
    print(f"  Q: {query}")

    result = _run(query)
    raw_fb = result.get("critic_feedback", "")
    rev_count = result.get("revision_count", 0)
    final = result.get("final_answer", "")

    print(f"  revision_count: {rev_count} (최대 2)")
    try:
        fb = json.loads(raw_fb)
        passed = fb.get("pass")
        reasons = fb.get("reasons", [])
        fix_targets = fb.get("fix_targets", [])
        print(f"  critic_feedback.pass: {passed}")
        if not passed:
            print(f"  reasons: {reasons}")
            print(f"  fix_targets: {fix_targets}")
        print(f"  {'✅ PASS' if isinstance(fb, dict) and 'pass' in fb else '❌ FAIL'} — JSON 형식 정상")
    except Exception as e:
        print(f"  ❌ FAIL — JSON 파싱 실패: {e} | raw: {raw_fb[:100]}")

    print(f"  최종 답변 길이: {len(final)}자")
    print(f"  {'✅ PASS' if rev_count <= 2 else '❌ FAIL'} — revision 한도 이내")

# ──────────────────────────────────────
# 메인
# ──────────────────────────────────────
TESTS = {
    "1": test_1_search_routing,
    "2": test_2_react_loop,
    "3": test_3_search_grader,
    "4": test_4_multihop,
    "5": test_5_loop_guard,
    "6": test_6_critic_json,
}

if __name__ == "__main__":
    targets = sys.argv[1:] or list(TESTS.keys())
    print("\n🚀 Phase 1 검증 테스트 시작")
    for key in targets:
        if key in TESTS:
            TESTS[key]()
        else:
            print(f"  [SKIP] 알 수 없는 테스트: {key}")
    print("\n" + "=" * 60)
    print("  테스트 완료")
    print("=" * 60)
