"""사용자 질문의 사실관계를 보존하면서 검색용 질문 하나로 정리합니다."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.const.model import create_llm
from src.prompt.query_rewriter_prompt import QUERY_REWRITE_PROMPT



def query_rewriter(text):
    # 빈 질문은 실제 모델을 호출하기 전에 확인합니다.
    if not isinstance(text, str) or not text.strip():
        raise ValueError("질문은 비어 있지 않은 문자열이어야 합니다.")

    # 분리한 지침과 사용자 질문을 각각 시스템·사용자 메시지로 구성합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", QUERY_REWRITE_PROMPT),
        ("human", "{question}"),
    ])
    # 프롬프트 → model.py의 LLM → 응답 문자열 흐름을 하나의 체인으로 묶습니다.
    chain = prompt | create_llm() | StrOutputParser()
    # 실제 LLM에 질문을 전달하고, 응답 앞뒤의 공백을 제거합니다.
    response = chain.invoke({"question": text.strip()}).strip()

    # 빈 응답이면 원본을 사용합니다. API 오류는 호출한 쪽으로 전달합니다.
    response = response or text.strip()
    # 재작성 질문을 터미널에 출력하고, 호출한 노드에 문자열로 전달합니다.
    print(f"[재작성 질문] {response}")
    return response


if __name__ == "__main__":
    # 터미널에서 직접 실행한 경우에만 질문을 입력받아 재작성합니다.
    query_rewriter(input("질문을 입력하세요: "))
