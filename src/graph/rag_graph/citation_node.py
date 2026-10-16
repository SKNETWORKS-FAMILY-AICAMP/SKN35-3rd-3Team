from src.graph.state import State
from src.citation import citation
from langchain_core.messages import AIMessage
# 노드 입니다.

# state에 따라서 노드의 상태를 정의합니다.

def citation_node(state : State):
    # 현재는 인용 구조 검사만 실행합니다. Judge는 연결하지 않습니다.
    result = citation(state, verbose=False)
    messages = state.get("messages") or []
    # 답변이 보류되면 대화 기록에도 같은 안내가 남도록 기존 AI 메시지를 교체합니다.
    if messages and isinstance(messages[-1], AIMessage) and messages[-1].content != result["answer"]:
        result["messages"] = [AIMessage(content=result["answer"], id=messages[-1].id)]
    return result
