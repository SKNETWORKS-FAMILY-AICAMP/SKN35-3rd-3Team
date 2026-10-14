import subprocess
import sys
import streamlit as st
from src.app.Chat import render_chat


CHECK_COMMAND = "check-all"
CHECK_STEPS = (
    (
        "전처리 테스트용 tmp 준비",
        ("powershell.exe", "-NoProfile", "-Command", "New-Item -ItemType Directory -Force -Path .\\tmp | Out-Null"),
    ),
    (
        "전처리 테스트 163건",
        ("uv", "run", "python", "-m", "unittest", "discover", "-s", "src/scripts/tests", "-t", ".", "-p", "test*_pmh.py"),
    ),
    ("Qdrant 연결·모델 확인", (sys.executable, "-m", "src.scripts.qdrant_setup", "check")),
    ("청크 수량 확인", (sys.executable, "-m", "src.scripts.qdrant_ingest", "check-input")),
    ("MCP Tool 4종 확인", (sys.executable, "-m", "src.scripts.mcp_server", "--check-tools")),
)


def run_project_check() -> int:
    """기존 검증 명령을 순서대로 실행합니다."""

    for title, arguments in CHECK_STEPS:
        print(f"\n[{title}]", flush=True)
        result = subprocess.run(arguments, check=False)
        if result.returncode:
            print(f"검증 중단: {title}")
            return result.returncode

    print("\n[최종 결과] 전처리 → Qdrant → 청크 → MCP 전체 통과")
    return 0


def main() -> int:
    if sys.argv[1:] == [CHECK_COMMAND]:
        check_result = run_project_check()
        if check_result:
            return check_result
        return subprocess.run(
            [sys.executable, "-m", "streamlit", "run", __file__],
            check=False,
        ).returncode

    st.set_page_config(page_title="생활법령 챗봇", page_icon="⚖️")
    render_chat()
    return 0


if __name__ == "__main__":
    main_exit_code = main()
    if main_exit_code:
        raise SystemExit(main_exit_code)
