"""LAW:MATE Streamlit UI and workflow adapter."""
import base64
import logging
import uuid
import streamlit as st
from src.app.progress import new_progress, advance_progress, finish_progress
from src.app.conversation_history import now_iso
from src.app.loading import LOADING_GRAPHIC

def to_b64(file_bytes: bytes) -> str:
    return base64.b64encode(file_bytes).decode()


def init_session():
    for key, value in {"thread_id": str(uuid.uuid4()), "history": [], "view": "AI 법률상담", "consultations": {}, "result": {}}.items():
        st.session_state.setdefault(key, value)


def ask(text: str, img_bytes_list: list[bytes], on_progress=None) -> str:
    progress = new_progress()
    st.session_state.rag_progress = progress
    if on_progress:
        on_progress()
    if "graph" not in st.session_state:
        from src.graph.workflow import build_workflow
        st.session_state.graph = build_workflow()
    result = {}
    for namespace, mode, event in st.session_state.graph.stream(
        {"origin_query": text, "retry_count": 0, "images": [to_b64(b) for b in img_bytes_list], "answer": "",
         "documents": [], "sources": [], "citations": [], "context_sources": [], "context_documents": [],
         "error_type": None, "generation_status": "", "citation_validation_status": "skipped"},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        stream_mode=["updates", "values"], subgraphs=True,
    ):
        if mode == "values" and not namespace:
            result = event
        elif mode == "updates":
            for node, update in event.items():
                if isinstance(update, dict):
                    advance_progress(progress, node, update)
            if on_progress:
                on_progress()
    finish_progress(progress, result)
    st.session_state.result = result
    if on_progress:
        on_progress()
    return result.get("answer") or "..."


def render_history(messages=None, start_index=0):
    for index, msg in enumerate(st.session_state.history if messages is None else messages, start=start_index):
        with st.container(key=f"law_message_{msg['role']}_{index}"):
            with st.chat_message(msg["role"], avatar=":material/auto_awesome:" if msg["role"] == "assistant" else ":material/person:"):
                if msg["role"] == "assistant":
                    st.markdown("**LAW:MATE 답변**")
                for image in msg.get("images", []):
                    st.image(image, width=300, alt="상담에 첨부한 이미지")
                st.markdown(msg["content"])


def submit_question(text, images, on_progress=None):
    st.session_state.result = {}
    st.session_state.history.append({"role": "user", "content": text, "images": images})
    render_history([st.session_state.history[-1]], start_index=len(st.session_state.history) - 1)
    with st.container(key=f"law_message_assistant_{len(st.session_state.history)}"), st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.markdown("**LAW:MATE 답변**")
        loading = st.empty()

        def refresh_loading():
            progress = st.session_state.get("rag_progress") or {}
            note = progress.get("note") or "질문과 첨부 자료를 살펴보고 있어요"
            count = len(progress.get("documents") or [])
            detail = f"검색된 근거 후보 {count}개를 함께 살펴보고 있어요" if count else "입력하신 상황에 맞는 안내를 준비하고 있어요"
            if progress.get("status") in {"complete", "limited", "error"}:
                loading.empty()
            else:
                with loading.container():
                    st.markdown(note)
                    st.caption(detail)
                    st.html(LOADING_GRAPHIC)
            if on_progress:
                on_progress()

        try:
            answer = ask(text, images, on_progress=refresh_loading)
        except Exception as exc:
            logging.getLogger(__name__).exception("Consultation failed")
            limited = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            answer = "현재 AI 모델 사용량 한도에 도달했어요. 잠시 후 다시 시도해 주세요." if limited else "상담 서비스에 연결하지 못했어요. 잠시 후 다시 시도해 주세요."
            st.session_state.result = {"answer": answer, "error_type": "service_error"}
            progress = st.session_state.get("rag_progress", new_progress())
            finish_progress(progress, st.session_state.result)
            st.session_state.rag_progress = progress
            refresh_loading()
        finally:
            loading.empty()
        st.markdown(answer)
    st.session_state.history.append({"role": "assistant", "content": answer})
    first = next(m["content"] for m in st.session_state.history if m["role"] == "user")
    previous = st.session_state.consultations.get(st.session_state.thread_id, {})
    updated_at = now_iso()
    st.session_state.consultations[st.session_state.thread_id] = {
        **previous, "title": previous["title"] if previous.get("custom_title") else first[:60],
        "history": list(st.session_state.history), "result": st.session_state.result,
        "progress": st.session_state.get("rag_progress"),
        "created_at": previous.get("created_at") or updated_at, "updated_at": updated_at,
    }
    st.rerun()


def render_chat():
    from src.app.views import render_app
    st.set_page_config(page_title="LAW:MATE | AI 법률 상담", page_icon="⚖️", layout="wide")
    init_session()
    render_app()
