import uuid

from langchain_core.messages import HumanMessage
from src.graph.workflow import build_workflow

def new_config() -> dict:
    return {"configurable": {"thread_id": str(uuid.uuid4())}}

def main():
    app = build_workflow()
    config = new_config()

    print("생활법률 상담 챗봇 (/new, q)")

    while True:
        try:
            text = input("\n나> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not text:
            continue
        if text in ("q", "/quit"):
            break
        if text == "/new":
            config = new_config()
            print("새 대화를 시작합니다.")
            continue

        try:
            out = app.invoke(
                {
                    "messages": [HumanMessage(content=text)],
                    "origin_query": text,
                    "retry_count": 0,   # 재시도 횟수는 매 턴 초기화
                },
                config,
            )
            print(f"\n봇> {out['messages'][-1].text}")
        except Exception as e:
            print(f"\n[오류] {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()