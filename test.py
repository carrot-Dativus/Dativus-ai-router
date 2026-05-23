# test.py
import asyncio
from ai_core.router import langgraph_app


async def run_test():
    print("🚀 [테스트 가동] AI 랭그래프 단독 구동 시작!")

    inputs = {
        "query": "파이썬으로 1부터 10까지 더하는 코드 짜줘",
        "workspace_id": None,
        "user_id": None,
        "history": []
    }

    try:
        # FastAPI 없이 AI만 쌩으로 돌려서 에러를 끄집어냅니다!
        async for event in langgraph_app.astream(inputs):
            print("\n✅ [작전 통과 완료 노드]:", event.keys())

        print("\n🎉 [테스트 성공] 아무 에러 없이 끝까지 도달했습니다!")

    except Exception as e:
        import traceback
        print("\n💥💥💥 [범인 검거! 정확한 에러 원인] 💥💥💥")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_test())