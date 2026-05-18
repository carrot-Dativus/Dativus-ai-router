import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ai_core.router import langgraph_app

# 환경 변수 로드
load_dotenv()

# 💡 [판사 요원 배정] 가장 똑똑하고 객관적인 모델을 채점관으로 씁니다.
judge_llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

# 💡 [테스트 셋] 시스템이 무조건 통과해야 하는 핵심 질문과 채점 기준
TEST_CASES = [
    {
        "id": 1,
        "query": "프론트엔드 상태 관리를 어떻게 하는 게 좋을까?",
        "criteria": "기술적인 조언이므로 반드시 3가지 대안(A/B/C)을 제시하고, 각 장단점을 비교한 뒤 마지막에 사용자의 선택을 물어봐야 한다."
    },
    {
        "id": 2,
        "query": "안녕! 오늘 날씨가 어때?",
        "criteria": "단순 인사 또는 날씨 질문이므로 3가지 대안을 제시할 필요 없이, 가볍고 친절하게 대답해야 한다."
    },
    {
        "id": 3,
        "query": "우리 팀이 어제 회의에서 결정한 데이터베이스 마이그레이션 전략을 요약해줘.",
        "criteria": "팀 기밀에 관한 질문이므로 사내 검색(search)을 활용하여 요약하려는 시도를 해야 하며, 데이터가 없다면 없다고 솔직하게 대답해야 한다."
    }
]


def run_evaluation():
    print("======================================================")
    print(" 🚀 [LLMOps] Dativus 자동 평가(Evaluation) 파이프라인 가동")
    print("======================================================\n")

    total_score = 0

    for test in TEST_CASES:
        print(f"▶ [테스트 {test['id']}] 질문: {test['query']}")

        # 1. 우리 시스템(Dativus)에 질문을 던져서 답변을 받아냅니다.
        result = langgraph_app.invoke({
            "query": test["query"],
            "workspace_id": "11111111-1111-1111-1111-111111111111",
            "user_id": "22222222-2222-2222-2222-222222222222"
        })
        ai_answer = result.get("final_answer", "")

        print(f"   [Dativus 답변] {ai_answer[:100]}... (생략)")

        # 2. 판사(Judge) AI에게 채점을 지시합니다.
        eval_prompt = f"""당신은 AI 시스템의 품질을 평가하는 수석 평가관입니다.
        아래의 [사용자 질문]에 대해 AI가 내놓은 [답변]이 [평가 기준]을 얼마나 잘 충족했는지 100점 만점으로 채점하세요.

        [사용자 질문]: {test['query']}
        [AI 답변]: {ai_answer}
        [평가 기준]: {test['criteria']}

        반드시 아래의 JSON 형식으로만 출력하세요. (다른 말은 절대 금지)
        {{"score": 90, "reason": "이유를 1~2줄로 짧게 작성"}}
        """

        try:
            # 판사 AI의 평가 결과 받기
            evaluation_result = judge_llm.invoke(eval_prompt).content.strip()

            # JSON 파싱
            import re
            json_match = re.search(r'\{.*\}', evaluation_result, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group(0))
                score = eval_data.get("score", 0)
                reason = eval_data.get("reason", "")
                total_score += score

                print(f"   🎯 [평가 점수]: {score}/100")
                print(f"   📝 [평가 사유]: {reason}\n")
            else:
                print("   ⚠️ [평가 실패] 채점관이 JSON 형식을 지키지 않았습니다.\n")

        except Exception as e:
            print(f"   ⚠️ [시스템 오류] 채점 중 에러 발생: {e}\n")

    # 최종 성적표 출력
    avg_score = total_score / len(TEST_CASES)
    print("======================================================")
    print(f" 🏆 [최종 결과] Dativus 시스템 평균 점수: {avg_score:.1f} / 100")
    if avg_score >= 90:
        print(" 🟢 [상태: PERFECT] 상용 서비스 배포 가능 수준입니다!")
    elif avg_score >= 70:
        print(" 🟡 [상태: GOOD] 양호하나 일부 프롬프트 튜닝이 필요합니다.")
    else:
        print(" 🔴 [상태: WARNING] 하네스 룰 위반 다수. 긴급 점검 요망.")
    print("======================================================")


if __name__ == "__main__":
    run_evaluation()