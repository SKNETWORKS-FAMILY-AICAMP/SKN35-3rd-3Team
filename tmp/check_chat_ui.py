import ast
from pathlib import Path
from streamlit.testing.v1 import AppTest
root = Path(__file__).resolve().parents[1]
for name in ["src/app/views.py", "src/app/Chat.py"]:
    ast.parse((root / name).read_text(encoding="utf-8"))
app = AppTest.from_file(str(root / "main.py")).run(timeout=30)
next(b for b in app.button if b.label == "AI 법률상담").click().run()
assert not app.exception, str(app.exception)
assert len(app.chat_input) == 1
app.session_state["history"] = [{"role": "user", "content": "테스트 질문"}, {"role": "assistant", "content": "테스트 답변"}]
app.session_state["result"] = {"answer": "테스트 답변", "citation_validation_status": "passed", "sources": []}
app.run()
assert not app.exception, str(app.exception)
assert len(app.chat_message) == 2
next(b for b in app.button if b.label == "상세 법률 분석 결과 보기").click().run()
assert app.session_state["view"] == "판례·법령 검색"
next(b for b in app.button if b.label == "AI 법률상담").click().run()
next(b for b in app.button if b.label == "새 상담 시작하기").click().run()
assert app.session_state["history"] == []
assert not app.exception, str(app.exception)
print("Chat UI, analysis navigation and new consultation OK")
