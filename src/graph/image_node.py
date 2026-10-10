from src.graph.state import State
from langchain_core.messages import HumanMessage
from prompt.image_prompt import IMAGE_PROMPT
from src.const.model import create_llm
# 노드 입니다.

llm = create_llm()

# 이미지를 판단하는 함수를 가져와서 실행

def to_image_part(img: str) -> dict:
    # url로 들어오면 그대로 사용
    url = img if img.startswith("http") else f"data:image/png;base64,{img}"
    return {"type": "image_url", "image_url": {"url": url}}

def image_node(state: State):
    query = state["origin_query"]

    # 텍스트 뒤에 이미지 데이터를 붙인다
    content = [{"type" : "text" , "text": f"{IMAGE_PROMPT}\n\n[사용자 질문]\n{query}"}]
    content += [to_image_part(img) for img in state["images"]]

    description = llm.invoke([HumanMessage(content=content)]).content

    return {"origin_query":f"{query}\n\n[첨부 이미지 설명]\n{description}", "images" : []}