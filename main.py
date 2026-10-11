import streamlit as st
from src.App.Chat import render_chat

def main():
    # set_page_config는 첫 번째 Streamlit 명령이어야 해서 여기서 호출
    st.set_page_config(page_title="생활법령 챗봇", page_icon="⚖️")

    render_chat()


if __name__ == "__main__":
    main()
