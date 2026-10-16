"""사용자 질문의 사실관계를 보존하면서 검색용 질문 하나로 정리합니다."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence

from src.const.model import create_llm
from src.prompt.query_rewriter_prompt import QUERY_REWRITE_PROMPT



def query_rewriter(text, llm=None, history: str = ""):
    # 빈 질문은 실제 모델을 호출하기 전에 확인합니다.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("질문은 비어 있지 않은 문자열이어야 합니다.")

    # 이전 대화가 있으면 함께 전달하고, 없으면 기존처럼 질문만 전달합니다.
    human = "[이전 대화]\n{history}\n\n[현재 질문]\n{question}" if history else "{question}"
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_PROMPT),
        ("human", human),
    ])
    chain = prompt | (llm if llm is not None else create_llm()) | StrOutputParser()

    inputs = {"question": text.strip()}
    if history:
        inputs["history"] = history

    # 이 줄이 빠져 있으면 UnboundLocalError가 납니다.
    response = chain.invoke(inputs).strip()

    # 빈 응답이면 원본을 사용합니다. API 오류는 호출한 쪽으로 전달합니다.
    response = response or text.strip()
    print(f"[재작성 질문] {response}")
    return response
