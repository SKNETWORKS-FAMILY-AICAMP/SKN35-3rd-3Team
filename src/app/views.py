"""Python translation of the supplied LAW:MATE Figma Make screens."""
import uuid
import re
from pathlib import Path
from html import escape
from urllib.parse import urlparse
import streamlit as st
import streamlit.components.v1 as components
from src.app.Chat import render_history, submit_question
from src.app.conversation_history import group_consultations, first_question, category

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
    with st.container(key="law_header"):
        brand, navigation = st.columns([1, 3], vertical_alignment="center")
        with brand:
            st.markdown('<div class="law-brand"><span class="law-brand-mark">⚖</span><span>LAW<span class="law-brand-colon">:</span>MATE</span></div>', unsafe_allow_html=True)
        with navigation:
            with st.container(horizontal=True, vertical_alignment="center"):
                for label in ["AI 법률상담", "판례·법령 검색", "이용 안내"]:
                    active = st.session_state.view == label or (label == "판례·법령 검색" and st.session_state.view == "판례 상세")
                    if st.button(label, key=f"nav_{label}", type="primary" if active else "tertiary"):
                        go(label)


def reset_conversation():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.history = []
    st.session_state.result = {}
    for key in ("rag_progress", "chat_draft", "selected_source"):
        st.session_state.pop(key, None)


def sidebar():
    with st.container(key="law_history"):
        if st.button("새 상담 시작하기", icon=":material/add:", type="primary", width="stretch"):
            reset_conversation()
            go("AI 법률상담")
        mode = st.session_state.get("history_group_mode", "날짜별")
        st.html(f'<div class="law-history-heading">상담 기록<span>{len(st.session_state.consultations)}</span></div>')
        with st.container(key="law_history_list"):
            for label, items in group_consultations(st.session_state.consultations, mode).items():
                st.html(f'<div class="law-history-group">{label}<span>{len(items)}</span></div>')
                for thread_id, item in items:
                    active = thread_id == st.session_state.thread_id
                    full_question = first_question(item)
                    title = item.get("title") if item.get("custom_title") else full_question
                    with st.container(key=f"law_thread_row_{thread_id}"):
                        with st.container(horizontal=True, vertical_alignment="center", gap="xxsmall"):
                            if st.button(" ".join(str(title).split()), key=f"history_{thread_id}", icon=":material/chat_bubble_outline:", type="primary" if active else "tertiary", width="stretch", help=full_question):
                                st.session_state.thread_id = thread_id
                                st.session_state.history = list(item["history"])
                                st.session_state.result = item["result"]
                                st.session_state.rag_progress = item.get("progress")
                                st.session_state.pop("chat_draft", None)
                                st.session_state.pop("selected_source", None)
                                go("AI 법률상담")
                            with st.container(key=f"law_thread_actions_{thread_id}", width=34):
                                with st.popover("상담 관리", icon=":material/more_horiz:", width=34):
                                    st.markdown("**상담 관리**")
                                    with st.form(f"rename_form_{thread_id}", border=False):
                                        new_title = st.text_input("상담 이름", value=str(title)[:80], max_chars=80, key=f"rename_title_{thread_id}")
                                        if st.form_submit_button("이름 변경", icon=":material/edit:"):
                                            if new_title.strip():
                                                item.update(title=new_title.strip(), custom_title=True)
                                                st.rerun()
                                            else:
                                                st.warning("상담 이름을 입력해 주세요.")
                                    if st.button("목록에서 삭제", key=f"delete_{thread_id}", icon=":material/delete_outline:"):
                                        st.session_state.deleted_consultation = (thread_id, st.session_state.consultations.pop(thread_id))
                                        if active:
                                            reset_conversation()
                                        st.rerun()
                        st.html(f'<span class="law-history-tag">{category(item)}</span>')
            if not st.session_state.consultations:
                st.html('<div class="law-history-empty">아직 상담 기록이 없어요.<br><small>질문을 보내면 이곳에 정리됩니다.</small></div>')
        with st.container(key="law_history_footer"):
            if st.session_state.get("deleted_consultation"):
                st.caption("상담 1개를 목록에서 삭제했습니다.")
                if st.button("삭제 되돌리기", icon=":material/undo:", key="restore_consultation", width="stretch"):
                    thread_id, item = st.session_state.pop("deleted_consultation")
                    st.session_state.consultations[thread_id] = item
                    st.rerun()
            st.html('<div class="law-session-profile"><span>G</span><div><strong>게스트 세션</strong><small>현재 브라우저에서 이용 중</small></div></div>')
            with st.popover("목록 표시 설정", icon=":material/tune:", width="stretch"):
                st.radio("상담 기록 분류", ["날짜별", "분야별"], key="history_group_mode")
                st.caption("분야는 첫 질문의 키워드로 분류합니다. 기록은 현재 세션에서만 보관됩니다.")
            st.caption("법령 · 판례 · 공식 안내 기반\n\n기록은 영구 저장되지 않습니다.")


def process_panel(revision=0):
    progress = st.session_state.get("rag_progress") or {}
    steps = progress.get("steps", ["waiting"] * 4)
    state = progress.get("note", "질문을 입력하면 분석을 시작해요")
    with st.container(key=f"law_process_{revision}", border=True):
        st.markdown('<div class="law-process-title"><span class="law-ai-mark">✦</span><div><strong>답변 분석 과정</strong><small>' + state + '</small></div></div>', unsafe_allow_html=True)
        for (title, desc), style in zip([("질문 분석", "입력한 상황과 첨부 자료 확인"), ("관련 법령 검색", "벡터 검색 · 근거 후보 재정렬"), ("인용 근거 확인", "검색 문서의 본문·구조 확인"), ("답변 생성", "검색된 근거를 바탕으로 답변 작성")], steps):
            symbol = {"done": "✓", "active": "", "waiting": "·", "skipped": "–", "error": "!"}[style]
            st.markdown(f'<div class="law-process-step"><span class="law-process-dot {style}">{symbol}</span><div><strong>{title}</strong><small>{desc}</small></div></div>', unsafe_allow_html=True)
        st.progress(sum(step in {"done", "skipped"} for step in steps) / 4)
        st.caption(state if progress else "질문 대기 중")
    with st.container(key=f"law_source_note_{revision}", border=True):
        st.markdown("**:material/manage_search: 검색된 근거 미리보기**")
        documents = progress.get("documents", [])
        st.caption(f"검색 후보 {len(documents)}개 · 답변에 인용된 자료와 다를 수 있어요" if documents else "검색이 완료되면 실제 검색 문서를 표시합니다.")
        for index, document in enumerate(documents[:5]):
            meta = document.get("metadata") or {}
            title = " ".join(str(value) for value in [meta.get("law_name") or meta.get("source_name") or meta.get("title") or "근거 자료", meta.get("article_label") or meta.get("article_number") or meta.get("case_number") or ""] if value)
            text = str(document.get("text") or "")
            with st.expander(title):
                st.caption({"law": "법령", "precedent": "판례"}.get(meta.get("document_type"), "안내 자료"))
                st.write(" ".join(text.split())[:160] + ("…" if len(text) > 160 else ""))
                with st.popover("원문 스니펫", icon=":material/article:", key=f"snippet_{revision}_{index}"):
                    st.markdown(f"**{escape(title)}**")
                    st.text(text[:3000])
                    if len(text) > 3000:
                        st.caption("긴 검색 청크의 앞 3,000자를 표시합니다.")
                url = meta.get("source_url") or ""
                if urlparse(url).scheme in {"http", "https"}:
                    st.html(f'<a href="{escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">출처 원문 ↗</a>')
        if progress.get("status") == "complete" and steps[1] == "skipped":
            st.caption("이번 응답은 법령 검색을 사용하지 않았습니다.")
        st.caption("문서 구조 확인은 법률 판단의 정확성이나 인용의 의미적 일치를 보증하지 않습니다.")


def chat():
    with st.container(key="law_chat_shell"):
        history, conversation, panel = st.columns([1.05, 3.8, 1.25], gap="small")
        with history:
            sidebar()
        with panel:
            process_slot = st.empty()
        revision = 0

        def refresh_process():
            nonlocal revision
            revision += 1
            with process_slot.container():
                process_panel(revision)

        refresh_process()
        with conversation:
            with st.container(key="law_conversation"):
                st.markdown('<div class="law-chat-heading"><span class="law-ai-mark">✦</span><div><strong>AI 법률 상담</strong><small><i></i> 법률 데이터 기반 상담</small></div><span class="law-data-badge">♧ 법률 데이터 기반 답변</span></div>', unsafe_allow_html=True)
                st.markdown('<div class="law-date-divider"><span>현재 상담</span></div>', unsafe_allow_html=True)
                with st.container(key="law_chat_messages", height=460, border=False, autoscroll=True):
                    render_history()
                    # Reserve a position above the composer for in-flight messages.
                    pending_messages = st.empty()
                    welcome = st.empty()
                    if not st.session_state.history:
                        with welcome.container(), st.container(key="law_chat_empty"):
                            components.html(
                                """<!doctype html><html lang="ko"><head><meta charset="utf-8">
                                <style>
                                body{margin:0;background:transparent;font-family:"Noto Sans KR","Malgun Gothic",sans-serif;color:#1e293b;}
                                h3{margin:0;text-align:center;font-size:clamp(24px,3.5vw,34px);font-weight:700;line-height:1.5;letter-spacing:-1px;}
                                .cursor{display:inline-block;width:2px;height:1em;margin-left:4px;vertical-align:-.1em;background:#e2be7a;animation:blink .7s step-end infinite;}
                                @keyframes blink{50%{opacity:0;}}
                                </style></head><body>
                                <h3 aria-label="어떤 법률 고민이 있으신가요?"><span id="title" aria-hidden="true"></span><span class="cursor" aria-hidden="true"></span></h3>
                                <script>
                                const text = Array.from("어떤 법률 고민이 있으신가요?");
                                const title = document.getElementById("title");
                                const cursor = document.querySelector(".cursor");
                                let index = 0;
                                function typeNext(){
                                    if(index < text.length){
                                        title.textContent += text[index++];
                                        window.setTimeout(typeNext, 100);
                                    }else{
                                        window.setTimeout(() => {cursor.style.display = "none";}, 700);
                                    }
                                }
                                if(window.matchMedia("(prefers-reduced-motion: reduce)").matches){
                                    title.textContent = text.join("");
                                    cursor.style.display = "none";
                                }else{window.setTimeout(typeNext, 350);}
                                </script></body></html>""",
                                height=110,
                                scrolling=False,
                            )
                            st.caption("거래 과정, 날짜, 금액 등 상황을 구체적으로 알려 주세요.")
                if st.session_state.result:
                    if st.button("상세 법률 분석 결과 보기", key="law_analysis_link", icon=":material/description:", width="stretch"):
                        go("판례·법령 검색")
                    st.caption(f"확인된 인용 출처 {len(sources())}건")
                with st.container(horizontal=True):
                    for title, _, _, question in CATEGORIES[:3]:
                        if st.button(title, key=f"example_{title}"):
                            st.session_state.chat_draft = question
                            st.rerun()
                prompt = st.chat_input("법률 고민을 자세히 입력해 주세요.", key="chat_draft", accept_file="multiple", file_type=["png", "jpg", "jpeg", "pdf"], submit_mode="disable")
                st.caption("구체적인 사실관계에 따라 판단이 달라질 수 있으며, 본 답변은 법률 자문을 대체하지 않습니다.")
                if prompt:
                    text = (prompt.text or "").strip()
                    if text:
                        try:
                            images = []
                            for file in prompt.files or []:
                                if file.name.lower().endswith(".pdf"):
                                    from src.app.pdf_utils import pdf_to_images
                                    images.extend(pdf_to_images(file.getvalue()))
                                else:
                                    images.append(file.getvalue())
                        except Exception:
                            st.error("첨부파일을 읽지 못했습니다. PDF가 손상되거나 암호로 보호되어 있는지 확인해 주세요.")
                        else:
                            welcome.empty()
                            with pending_messages.container():
                                    submit_question(text, images, on_progress=refresh_process)
                    else:
                        st.warning("이미지와 함께 질문을 입력해 주세요.")


def report_groups(items):
    """Group chunks for display while retaining their original detail indices."""
    groups = {"law": {}, "precedent": {}, "guide": {}}
    for index, source in enumerate(items):
        kind = source.get("document_type") or ""
        category = kind if kind in {"law", "precedent"} else "guide"
        identity = source.get("document_id") or source.get("source_url") or source.get("source_id") or str(index)
        if category == "law":
            identity = (identity, source.get("article"), source.get("version_id"))
        groups[category].setdefault(identity, []).append((index, source))
    return groups


def report_heading(icon, kicker, title, count=None):
    badge = f'<span class="law-count">{count}건</span>' if count is not None else ""
    st.markdown(f'<div class="law-section-heading"><span class="law-section-icon">{icon}</span><div><small>{escape(kicker)}</small><strong>{escape(title)}</strong></div>{badge}</div>', unsafe_allow_html=True)


def report_sources(title, kind, groups):
    with st.container(key=f"law_report_{kind}"):
        report_heading("§" if kind == "law" else "⚖" if kind == "precedent" else "▤", "판단의 기준이 된" if kind == "law" else "함께 확인할 근거 자료", title, len(groups))
        if not groups:
            st.caption(f"이번 답변에 인용된 {title}가 없습니다.")
        for entries in groups.values():
            index, source = entries[0]
            name = source.get("source_name") or "법률 자료"
            article = source.get("article") or source.get("article_title") or ""
            ids = " · ".join(str(item.get("source_id") or "") for _, item in entries)
            with st.container(key=f"law_evidence_{index}"):
                excerpt = " ".join(str(source.get("context_text") or source.get("text") or "").split())
                st.html(f'<div class="law-evidence-copy"><div class="law-evidence-title"><span class="law-type">{escape({"law": "법령", "precedent": "판례", "guide": "안내"}[kind])}</span><strong>{escape(name)} {escape(str(article))}</strong></div><p class="law-evidence-excerpt">{escape(excerpt[:135])}{"…" if len(excerpt) > 135 else ""}</p><p class="law-evidence-meta">인용 {escape(ids)}</p></div>')
                if st.button("근거 자료 상세 보기", key=f"source_{index}", icon=":material/chevron_right:", width="stretch"):
                    st.session_state.selected_source = index
                    go("판례 상세")
                if len(entries) > 1:
                    with st.expander(f"이 자료의 인용 {len(entries)}개"):
                        for detail_index, item in entries:
                            if st.button(f"{item.get('source_id', '')} 인용 본문 보기", key=f"chunk_{detail_index}"):
                                st.session_state.selected_source = detail_index
                                go("판례 상세")


def analysis():
    with st.container(key="law_analysis"):
        render_analysis_report()


def render_analysis_report():
    st.caption("AI 법률상담 › 분석 결과")
    result = st.session_state.result
    if not result:
        st.info("상담을 진행하면 답변과 인용 근거가 표시됩니다.")
        if st.button("상담 시작하기", type="primary"):
            go("AI 법률상담")
        return
    query = next((m["content"] for m in reversed(st.session_state.history) if m["role"] == "user"), "")
    items = sources()
    groups = report_groups(items)
    passed = result.get("citation_validation_status") in {"passed", "passed_with_missing_metadata"}
    status = "인용 구조 확인" if passed else "서비스 연결 오류" if result.get("error_type") else "추가 확인 필요"
    with st.container(key="law_report_banner"):
        st.markdown(f'<div class="law-banner-content"><div><span class="law-report-label">법률 분석 보고서</span><h1>{escape(query or "상담 분석 결과")}</h1><p>상담 내용과 검색된 근거를 바탕으로 작성한 분석 결과</p></div><div class="law-report-status"><strong>{len(items)}<small>개 인용</small></strong><span>{status}</span></div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="law-report-badges"><span class="law-status-badge">{status}</span><span>법령 {len(groups["law"])}건</span><span class="violet">판례 {len(groups["precedent"])}건</span><span>안내 자료 {len(groups["guide"])}건</span></div>', unsafe_allow_html=True)
    left, right = st.columns([2.5, 1], gap="medium")
    with left:
        with st.container(key="law_report_summary"):
            report_heading("✧", "한눈에 보는", "핵심 요약")
            answer = result.get("answer") or "답변이 없습니다."
            paragraphs = re.split(r"\n\s*\n", answer.strip())
            with st.container(key="law_summary_highlight"):
                st.markdown(paragraphs[0])
            st.caption("답변의 첫 문단입니다. 전체 답변과 인용된 근거를 함께 확인해 주세요.")
            with st.expander("AI 분석 답변 전체 보기"):
                st.markdown(answer)
        report_sources("관련 법령", "law", groups["law"])
        report_sources("유사 판례", "precedent", groups["precedent"])
        if groups["guide"]:
            report_sources("관련 안내 자료", "guide", groups["guide"])
    with right:
        with st.container(key="law_report_actions"):
            report_heading("→", "다음 단계", "권장 대응 방향")
            for number, (title, desc) in enumerate([
                ("상담 사실관계 정리", "날짜, 금액, 계약 내용과 상대방 답변을 정리하세요."),
                ("관련 자료 보관", "계약서, 거래 내역과 대화 기록을 보관하세요."),
                ("분석 근거 확인", "아래 출처의 본문과 적용 조건을 확인하세요."),
                ("추가 질문으로 구체화", "빠진 사실관계를 알려 주시면 이어서 상담할 수 있어요."),
            ], 1):
                st.markdown(f'<div class="law-action-step"><span>{number}</span><div><strong>{title}</strong><small>{desc}</small></div></div>', unsafe_allow_html=True)
            if st.button("후속 질문하기", type="primary", icon=":material/arrow_forward:", width="stretch"):
                go("AI 법률상담")
        with st.container(key="law_report_warning"):
            report_heading("△", "확인해 주세요", "주의사항")
            st.caption("사건별 사실관계에 따라 판단이 달라질 수 있습니다. 중요한 결정 전 전문가 상담을 권장합니다.")
            st.caption("인용 구조 확인은 법률 판단의 정확성이나 법령의 최신성을 보증하지 않습니다.")
        with st.container(key="law_report_origins"):
            report_heading("◇", "근거 원문", "분석 출처")
            seen = set()
            for source in items:
                url = source.get("source_url") or ""
                name = source.get("source_name") or "법률 자료"
                if (name, url) in seen:
                    continue
                seen.add((name, url))
                if urlparse(url).scheme in {"http", "https"}:
                    st.link_button(name, url, icon=":material/open_in_new:", width="stretch")
                else:
                    st.caption(name)
            if not items:
                st.caption("확인된 인용 출처가 없습니다.")


def detail():
    with st.container(key="law_detail"):
        render_source_detail()


def render_source_detail():
    items = sources()
    index = st.session_state.get("selected_source", 0)
    if not items or index >= len(items):
        go("판례·법령 검색")
    source = items[index]
    st.caption("분석 결과 › 근거 자료 상세")
    if st.button("분석 결과로 돌아가기", icon=":material/arrow_back:"):
        go("판례·법령 검색")
    st.title(source.get("source_name") or "근거 자료 상세")
    left, right = st.columns([3, 1], gap="large")
    with left.container(key="law_detail_body", border=True):
        st.subheader(str(source.get("article_title") or source.get("article") or "인용 본문"))
        text = str(source.get("text") or source.get("context_text") or "제공된 본문이 없습니다.")
        paragraphs = re.split(r"\n\s*\n|(?<=다\.)[ \t]+", text.strip())
        st.html('<div class="law-source-reading">' + ''.join(f'<p>{escape(paragraph)}</p>' for paragraph in paragraphs if paragraph) + '</div>')
        with st.expander("원문 텍스트 그대로 보기"):
            st.text(text)
    with right.container(key="law_detail_meta", border=True):
        st.subheader("자료 정보")
        for label, field in [("인용 번호", "source_id"), ("시행일", "effective_date"), ("문서 ID", "document_id")]:
            if source.get(field):
                st.caption(label)
                st.text(str(source[field]))
        url = source.get("source_url") or ""
        if urlparse(url).scheme in {"http", "https"}:
            st.link_button("원문 보기", url, icon=":material/open_in_new:")


def close_other_faq(opened_key):
    """Expander callbacks run before rendering, so sibling states can be reset."""
    if st.session_state.get(opened_key):
        for key in tuple(st.session_state):
            if key.startswith("guide_faq_") and key != opened_key:
                st.session_state[key] = False


def guide():
    with st.container(key="law_guide"):
        with st.container(key="law_guide_intro"):
            st.markdown('<span class="law-report-label">LAW:MATE 이용 안내</span><h1>생활 속 법률 고민,<br>근거와 함께 살펴보세요.</h1><p>LAW:MATE는 법령·판례·공식 안내 자료를 검색하고,<br>찾은 근거를 바탕으로 생활분쟁에 필요한 정보를 안내합니다.</p>', unsafe_allow_html=True)
            if st.button("AI 법률상담 시작하기", type="primary", icon=":material/arrow_forward:", key="guide_start"):
                go("AI 법률상담")

        st.subheader("어떤 고민을 이야기할 수 있나요?")
        for col, (title, description, icon, _) in zip(st.columns(4, gap="small"), CATEGORIES):
            with col.container(border=True, key=f"law_guide_category_{icon}"):
                st.markdown(f":material/{icon}:")
                st.markdown(f"**{title}**")
                st.caption(description)

        st.subheader("상담부터 근거 확인까지, 이렇게 이용하세요")
        steps = [
            ("01", "상황과 질문 입력", "AI 법률상담에서 언제, 누구와, 어떤 일이 있었는지 알려 주세요. 금액·계약 내용·상대방의 답변과 원하는 해결 방향을 함께 적으면 도움이 됩니다."),
            ("02", "이미지 함께 첨부", "입력창의 첨부 버튼으로 PNG·JPG·JPEG 이미지를 추가할 수 있습니다. 계약서 사진이나 대화 캡처는 질문과 함께 보내 주세요."),
            ("03", "답변과 분석 결과 확인", "답변을 받은 뒤 ‘상세 법률 분석 결과 보기’ 또는 상단 ‘판례·법령 검색’을 누르세요. 핵심 요약, 전체 답변과 인용된 자료를 확인할 수 있습니다."),
            ("04", "원문 확인 후 이어서 질문", "자료 카드의 ‘근거 자료 상세 보기’에서 본문을 읽고, 원문 링크가 있으면 출처도 확인하세요. ‘후속 질문하기’로 돌아가 추가 상황을 설명할 수 있습니다."),
        ]
        for number, title, description in steps:
            with st.container(key=f"law_guide_step_{number}", border=True):
                st.markdown(
                    '<div class="law-step-layout">'
                    f'<div class="law-step-number">{escape(number)}</div>'
                    '<div class="law-step-copy">'
                    f'<div class="law-step-title">{escape(title)}</div>'
                    f'<div class="law-step-description">{escape(description)}</div>'
                    '</div></div>', unsafe_allow_html=True,
                )

        left, right = st.columns(2, gap="medium")
        with left.container(key="law_guide_example", border=True):
            st.subheader("질문은 이렇게 적어 보세요", icon=":material/edit_note:")
            st.markdown("지난주 중고거래로 물건을 구매하고 15만 원을 송금했어요. 판매자가 발송을 미루다가 연락을 끊었습니다. 송금 내역과 대화 캡처가 있는데, 어떤 대응 방법을 확인하면 좋을까요?")
            st.caption("발생 시점 · 거래 금액 · 진행 상황 · 보유 자료 · 궁금한 점")
        with right.container(key="law_guide_evidence", border=True):
            st.subheader("답변의 근거 읽는 법", icon=":material/menu_book:")
            st.markdown("**[S1], [S2]**는 답변에서 참조한 자료의 인용 번호입니다. 분석 결과에서 같은 번호의 근거를 찾아보세요.")
            st.caption("같은 문서의 여러 인용은 한 카드로 묶입니다. 법령·판례·안내 자료는 구분해 표시하며, 인용된 자료가 없는 항목은 비어 있을 수 있습니다.")

        st.subheader("자주 묻는 질문")
        for index, (title, description) in enumerate([
            ("판례·법령 검색 탭에서 직접 검색할 수 있나요?", "현재는 진행 중인 상담의 분석 결과를 보여주는 탭입니다. 별도의 독립 검색 입력은 제공하지 않습니다. 먼저 AI 법률상담에 상황을 입력해 주세요."),
            ("인용이 확인되면 답변이 정확하다는 뜻인가요?", "인용 구조 검사는 출처와 본문 등의 연결을 확인합니다. 법적 결론의 정확성이나 법령의 최신성을 보증하지 않으므로, 원문의 적용 시점과 조건을 함께 확인해 주세요."),
            ("상담 기록은 계속 저장되나요?", "기록은 현재 브라우저 세션에서 확인할 수 있으며 영구 저장을 제공하지 않습니다. ‘새 상담 시작하기’를 누르면 별도 상담을 시작합니다."),
            ("모든 법률 분야를 상담할 수 있나요?", "중고거래, 주택임대차, 근로·노동, 소비자 피해 등 생활분쟁을 중심으로 구성되어 있습니다. 형사 사건, 이혼, 상속, 세금 등은 지원 범위 밖이며 개별 사건에 대한 전문 상담을 대신하지 않습니다."),
        ]):
            faq_key = f"guide_faq_{index}"
            with st.expander(title, key=faq_key, on_change=close_other_faq, args=(faq_key,)):
                st.write(description)
        with st.container(key="law_guide_notice"):
            st.markdown("**:material/privacy_tip: 상담 전 확인해 주세요**")
            st.caption("질문과 첨부 이미지에서 주민등록번호, 계좌번호, 연락처 등 불필요한 개인정보를 가려 주세요. 자료가 부족하면 추가 질문이나 답변 보류 안내가 나올 수 있습니다. 중요한 결정 전에는 전문가와 상담해 주세요.")


def render_app():
    if st.session_state.view == "홈":
        st.session_state.view = "AI 법률상담"
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    if st.session_state.view == "AI 법률상담":
        css += Path(__file__).with_name("chat.css").read_text(encoding="utf-8")
    css += Path(__file__).with_name("palette.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    header()
    {"AI 법률상담": chat, "판례·법령 검색": analysis, "판례 상세": detail, "이용 안내": guide}[st.session_state.view]()
