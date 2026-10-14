"""LAW:MATE Streamlit UI and workflow adapter."""
import base64
import logging
import uuid
import streamlit as st

def to_b64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode()


def init_session():
    for key, value in {"thread_id": str(uuid.uuid4()), "history": [], "view": "AI 법률상담", "consultations": {}, "result": {}}.items():
        st.session_state.setdefault(key, value)


def ask(text: str, img_bytes_list: list[bytes]) -> str:
    if "graph" not in st.session_state:
        from src.graph.workflow import build_workflow
        st.session_state.graph = build_workflow()
    result = st.session_state.graph.invoke(
        {"origin_query": text, "retry_count": 0, "images": [to_b64(b) for b in img_bytes_list],"answer": "",},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
    )
    st.session_state.result = result
    return result.get("answer") or "..."


def render_history():
    for index, msg in enumerate(st.session_state.history):
        with st.container(key=f"law_message_{msg['role']}_{index}"):
            with st.chat_message(msg["role"], avatar=":material/auto_awesome:" if msg["role"] == "assistant" else ":material/person:"):
                if msg["role"] == "assistant":
                    st.markdown("**LAW:MATE 답변**")
                for image in msg.get("images", []):
                    st.image(image, width=300, alt="상담에 첨부한 이미지")
                st.markdown(msg["content"])


def submit_question(text, images):
    st.session_state.result = {}
    st.session_state.history.append({"role": "user", "content": text, "images": images})
    with st.chat_message("user"):
        for image in images:
            st.image(image, width=300, alt="상담에 첨부한 이미지")
        st.markdown(text)
    with st.chat_message("assistant"):
        try:
            with st.spinner("질문을 분석하고 관련 법률 정보를 찾고 있어요..."):
                answer = ask(text, images)
        except Exception as exc:
            logging.getLogger(__name__).exception("Consultation failed")
            limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            answer = "현재 AI 모델 사용량 한도에 도달했어요. 잠시 후 다시 시도해 주세요." if limited else "상담 서비스에 연결하지 못했어요. 잠시 후 다시 시도해 주세요."
            st.session_state.result = {"answer": answer, "error_type": "service_error"}
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    first = next(m["content"] for m in st.session_state.history if m["role"] == "user")
    st.session_state.consultations[st.session_state.thread_id] = {"title": first[:30], "history": list(st.session_state.history), "result": st.session_state.result}
    st.rerun()


def render_chat():
    from src.app.views import render_app
    st.set_page_config(page_title="LAW:MATE | AI 법률 상담", page_icon="⚖️", layout="wide")
    init_session()
    render_app()
