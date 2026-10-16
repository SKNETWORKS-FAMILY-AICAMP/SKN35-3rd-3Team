"""Python translation of the supplied LAW:MATE Figma Make screens."""
import uuid
from urllib.parse import urlparse
import streamlit as st
from src.app.Chat import render_history, submit_question

CATEGORIES = [
    ("중고거래", "사기·환불·분쟁 해결", "storefront", "중고거래 사기를 당했는데 어떻게 대응해야 하나요?"),
    ("주택임대차", "보증금·계약·하자 분쟁", "apartment", "임대차 계약을 해지하고 보증금을 돌려받고 싶어요."),
    ("근로·노동", "임금·해고·직장 내 문제", "work", "퇴직금을 받지 못했는데 어떻게 해야 하나요?"),
    ("소비자 피해", "계약·환불·피해 보상", "shield", "구매한 상품의 환불을 거절당했어요."),
]


def go(view):
    st.session_state.view = view
    st.rerun()


def sources():
    result = st.session_state.result
    if result.get("citation_validation_status") not in {"passed", "passed_with_missing_metadata"}:
        return []
    return result.get("sources") or result.get("citations") or []


def header():
    with st.container(horizontal=True, vertical_alignment="center"):
        st.subheader("⚖️ LAW:MATE")
        for label in ["홈", "AI 법률상담", "판례·법령 검색", "이용 안내"]:
            active = st.session_state.view == label or (label == "판례·법령 검색" and st.session_state.view == "판례 상세")
            if st.button(label, key=f"nav_{label}", type="primary" if active else "tertiary"):
                go(label)
        if st.button("무료로 시작하기", type="primary", icon=":material/arrow_forward:"):
            go("AI 법률상담")
    st.divider()


def home():
    st.space("medium")
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.badge("법률 전문 인공지능 상담", icon=":material/auto_awesome:", color="violet")
        st.title("복잡한 법률 문제,\n이제 쉽게 해결하세요")
        st.markdown("AI가 법령과 판례를 분석해 이해하기 쉬운 답변을 제공합니다.\n\n누구나 이용할 수 있는 법률 상담을 경험해 보세요.")
        with st.form("home_question", border=True):
            question = st.text_area("무엇이 궁금하신가요?", placeholder="예) 중고거래 사기를 당했는데 돈을 돌려받을 수 있나요?")
            if st.form_submit_button("상담 시작하기", type="primary", icon=":material/arrow_forward:"):
                if question.strip():
                    st.session_state.pending_question = question.strip()
                    go("AI 법률상담")
                else:
                    st.warning("궁금한 내용을 입력해 주세요.")
        st.caption("이미지 첨부 가능 · 별도 가입 없이 시작")
    with right:
        st.space("large")
        with st.container(border=True):
            st.subheader("법률 분석 보고서", icon=":material/balance:")
            st.badge("상담 결과 미리보기", color="violet")
            st.markdown("### 내 질문에 맞춘 법률 정보")
            st.info("상담을 시작하면 AI 답변과 확인된 인용 출처를 살펴볼 수 있어요.", icon=":material/auto_awesome:")
            a, b = st.columns(2)
            with a.container(border=True):
                st.markdown("**관련 법령**")
                st.caption("판단의 기준이 되는 조항")
            with b.container(border=True):
                st.markdown("**인용 출처**")
                st.caption("답변의 근거가 된 자료")
            st.markdown("**상담 이용 순서**")
            st.caption("① 상황 입력 → ② AI 답변 확인 → ③ 근거 자료 살펴보기")
    st.space("large")
    st.caption("주요 상담 분야")
    st.header("어떤 법률 고민이 있으신가요?")
    st.caption("자주 찾는 분야를 선택하면 더 빠르게 상담을 시작할 수 있어요.")
    for col, (title, desc, icon, question) in zip(st.columns(4), CATEGORIES):
        with col.container(border=True):
            st.subheader(title, icon=f":material/{icon}:")
            st.caption(desc)
            if st.button("상담하기", key=f"category_{title}", width="stretch"):
                st.session_state.chat_draft = question
                go("AI 법률상담")


def sidebar():
    with st.sidebar:
        st.subheader("상담 기록", icon=":material/forum:")
        if st.button("새 상담 시작하기", icon=":material/add:", type="primary", width="stretch"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.history = []
            st.session_state.result = {}
            st.session_state.pending_question = None
            st.session_state.pop("chat_draft", None)
            go("AI 법률상담")
        st.caption("이번 세션의 상담")
        for thread_id, item in reversed(list(st.session_state.consultations.items())):
            if st.button(item["title"], key=f"history_{thread_id}", width="stretch"):
                st.session_state.thread_id = thread_id
                st.session_state.history = list(item["history"])
                st.session_state.result = item["result"]
                st.session_state.pending_question = None
                go("AI 법률상담")
        st.caption("기록은 현재 브라우저 세션에서만 유지됩니다.")


def chat():
    sidebar()
    conversation, panel = st.columns([3, 1], gap="large")
    with conversation:
        st.subheader("AI 법률 상담", icon=":material/auto_awesome:")
        st.caption("법률 데이터 기반 답변")
        render_history()
        if not st.session_state.history:
            st.markdown("### 어떤 법률 고민이 있으신가요?")
            st.caption("거래 과정, 날짜, 금액 등 상황을 구체적으로 알려 주세요.")
        with st.container(horizontal=True):
            for title, _, _, question in CATEGORIES[:3]:
                if st.button(title, key=f"example_{title}"):
                    st.session_state.chat_draft = question
                    st.rerun()
        prompt = st.chat_input("법률 고민을 자세히 입력해 주세요.", key="chat_draft", accept_file="multiple", file_type=["png", "jpg", "jpeg"], submit_mode="disable")
        st.caption("구체적인 사실관계에 따라 판단이 달라질 수 있으며, 답변은 법률 자문을 대체하지 않습니다.")
        if st.session_state.result and st.button("상세 법률 분석 결과 보기", icon=":material/description:", width="stretch"):
            go("판례·법령 검색")
    with panel:
        with st.container(border=True):
            st.subheader("답변 분석 과정", icon=":material/auto_awesome:")
            result = st.session_state.result
            st.caption("상담 오류" if result.get("error_type") else "답변 생성 완료" if result else "질문을 기다리고 있어요")
            st.markdown("**질문 분석 → 자료 검색 → 답변 생성**")
            if result:
                st.markdown(f"확인된 인용 출처 **{len(sources())}건**")
        with st.container(border=True):
            st.markdown("**답변의 근거 확인**")
            st.caption("분석 결과에서 실제 인용 자료와 원문을 확인하세요.")
    pending = st.session_state.pop("pending_question", None)
    if prompt:
        text = (prompt.text or "").strip()
        if not text:
            st.warning("이미지와 함께 질문을 입력해 주세요.")
            return
        with conversation:
            submit_question(text, [f.getvalue() for f in prompt.files or []])
    elif pending:
        with conversation:
            submit_question(pending, [])


def analysis():
    st.caption("홈 › AI 법률상담 › 분석 결과")
    st.title("법률 분석 보고서")
    result = st.session_state.result
    if not result:
        st.info("상담을 진행하면 답변과 인용 근거가 표시됩니다.")
        if st.button("상담 시작하기", type="primary"):
            go("AI 법률상담")
        return
    query = next((m["content"] for m in reversed(st.session_state.history) if m["role"] == "user"), "")
    st.subheader(query)
    left, right = st.columns([3, 1], gap="large")
    with left:
        with st.container(border=True):
            st.subheader("AI 분석 답변", icon=":material/auto_awesome:")
            st.markdown(result.get("answer") or "답변이 없습니다.")
        st.subheader("관련 법령·판례 및 인용 자료", icon=":material/menu_book:")
        if not sources():
            st.caption("이번 답변에는 확인된 인용 출처가 없습니다.")
        for index, source in enumerate(sources()):
            with st.container(border=True):
                st.markdown(f"**{source.get('source_name') or '법률 자료'}**")
                st.caption(str(source.get("article") or source.get("source_id") or ""))
                if source.get("article_title"):
                    st.markdown(source["article_title"])
                if st.button("근거 자료 상세 보기", key=f"source_{index}", icon=":material/chevron_right:"):
                    st.session_state.selected_source = index
                    go("판례 상세")
    with right:
        with st.container(border=True):
            st.subheader("분석 출처")
            st.metric("확인된 인용 자료", f"{len(sources())}건")
            st.caption("인용 구조 확인은 법률 판단의 정확성을 보증하지 않습니다.")
        with st.container(border=True):
            st.subheader("주의사항", icon=":material/info:")
            st.caption("사건별 사실관계에 따라 판단이 달라질 수 있습니다. 중요한 결정 전 전문가 상담을 권장합니다.")
        if st.button("후속 질문하기", type="primary", width="stretch"):
            go("AI 법률상담")


def detail():
    items = sources()
    index = st.session_state.get("selected_source", 0)
    if not items or index >= len(items):
        go("판례·법령 검색")
    source = items[index]
    st.caption("홈 › 분석 결과 › 근거 자료 상세")
    if st.button("분석 결과로 돌아가기", icon=":material/arrow_back:"):
        go("판례·법령 검색")
    st.title(source.get("source_name") or "근거 자료 상세")
    left, right = st.columns([3, 1], gap="large")
    with left.container(border=True):
        st.subheader(str(source.get("article_title") or source.get("article") or "인용 본문"))
        st.markdown(source.get("text") or source.get("context_text") or "제공된 본문이 없습니다.")
    with right.container(border=True):
        st.subheader("자료 정보")
        for label, field in [("인용 번호", "source_id"), ("시행일", "effective_date"), ("문서 ID", "document_id")]:
            if source.get(field):
                st.caption(label)
                st.text(str(source[field]))
        url = source.get("source_url") or ""
        if urlparse(url).scheme in {"http", "https"}:
            st.link_button("원문 보기", url, icon=":material/open_in_new:")


def guide():
    st.title("이용 안내")
    for title, description in [("1. 상황을 알려 주세요", "날짜, 금액, 계약 내용과 궁금한 점을 입력하세요. 상담 화면에서 PNG·JPG 이미지를 첨부할 수도 있어요."), ("2. 답변과 근거를 확인하세요", "상세 법률 분석 결과에서 AI 답변과 확인된 인용 자료를 살펴보세요."), ("3. 이어서 질문하세요", "같은 상담에서 후속 질문을 하거나 새 상담을 시작할 수 있어요.")]:
        with st.container(border=True):
            st.subheader(title)
            st.markdown(description)
    st.caption("상담 기록은 브라우저 세션에 저장됩니다. 주민등록번호 등 민감한 개인정보는 입력하지 마세요.")


def render_app():
    header()
    {"홈": home, "AI 법률상담": chat, "판례·법령 검색": analysis, "판례 상세": detail, "이용 안내": guide}[st.session_state.view]()
