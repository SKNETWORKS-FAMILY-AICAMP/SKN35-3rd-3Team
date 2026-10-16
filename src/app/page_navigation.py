"""페이지 스크롤용 Top 버튼. 채팅 내부 스크롤과는 별도로 동작한다."""

from pathlib import Path

import streamlit as st

_SCRIPT = Path(__file__).with_suffix(".js").read_text(encoding="utf-8")


def render_page_navigation():
    # rerun 시 기존 이벤트를 정리하고, DOM 마운트 후 새 버튼에 연결한다(JS에서 처리).
    st.markdown(
        '<button id="law-back-to-top" class="law-top-button" type="button" '
        'aria-label="페이지 맨 위로 이동" title="맨 위로 이동" hidden>'
        '<span class="law-top-arrow" aria-hidden="true">↑</span>'
        '<span class="law-top-label">Top</span></button>',
        unsafe_allow_html=True,
    )
    st.html(f"<script>{_SCRIPT}</script>", unsafe_allow_javascript=True)
