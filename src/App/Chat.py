import base64
import uuid

import streamlit as st

from src.graph.workflow import build_workflow  # 실제 파일 이름에 맞게 수정


def to_b64(file_bytes: bytes) -> str:
    # 이미지 바이트를 base64 문자열로 변환
    return base64.b64encode(file_bytes).decode()


def init_session():
    # 그래프와 thread_id는 세션 동안 한 번만 생성 (입력마다 스크립트가 재실행되기 때문)
    if "graph" not in st.session_state:
        st.session_state.graph = build_workflow()
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []  # 화면 표시용 대화 기록


def render_history():
    # 이전 대화 표시
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            for img_bytes in msg.get("images", []):
                st.image(img_bytes, width=300)
            st.markdown(msg["content"])


def ask(text: str, img_bytes_list: list[bytes]) -> str:
    # 그래프를 실행해서 답변 문자열을 반환
    try:
        result = st.session_state.graph.invoke(
            {
                "origin_query": text,
                "retry_count": 0,
                "images": [to_b64(b) for b in img_bytes_list],  # 없으면 []
            },
            config={"configurable": {"thread_id": st.session_state.thread_id}},
        )
        return result.get("answer", "답변을 생성하지 못했어요.")
    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
            return "현재 AI 모델 사용량 한도에 도달했어요. 잠시 후 다시 시도해 주세요."
        raise  # 그 외 에러는 그대로


def render_chat():
    # 채팅 화면 전체를 그리는 함수
    st.title("⚖️ 생활법령 챗봇")

    init_session()
    render_history()

    # 입력창: 화면 아래 가운데, Enter로 전송, 이미지 첨부 가능
    prompt = st.chat_input(
        "질문을 입력하세요. 이미지를 첨부할 수도 있어요.",
        accept_file="multiple",
        file_type=["png", "jpg", "jpeg"],
    )
    if not prompt:
        return

    text = (prompt.text or "").strip()
    files = prompt.files or []

    if not text:
        st.warning("질문을 함께 입력해 주세요.")
        return

    img_bytes_list = [f.getvalue() for f in files]

    # 사용자 메시지 표시
    with st.chat_message("user"):
        for b in img_bytes_list:
            st.image(b, width=300)
        st.markdown(text)
    st.session_state.history.append(
        {"role": "user", "content": text, "images": img_bytes_list}
    )

    # 답변 생성 및 표시
    with st.chat_message("assistant"):
        with st.spinner("답변을 찾는 중..."):
            answer = ask(text, img_bytes_list)
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})