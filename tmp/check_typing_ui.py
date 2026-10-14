import ast
from pathlib import Path
from streamlit.testing.v1 import AppTest
root = Path(__file__).resolve().parents[1]
ast.parse((root / "src/app/views.py").read_text(encoding="utf-8"))
app = AppTest.from_file(str(root / "main.py")).run(timeout=30)
next(b for b in app.button if b.label == "AI 법률상담").click().run()
assert not app.exception, str(app.exception)
frames = app.get("iframe")
assert len(frames) == 1
html = frames[0].proto.srcdoc
assert 'window.setTimeout(typeNext, 100)' in html
assert '어떤 법률 고민이 있으신가요?' in html
app.session_state["history"] = [{"role": "user", "content": "질문"}]
app.run()
assert len(app.get("iframe")) == 0
assert not app.exception, str(app.exception)
print("Typing component HTML and empty conversation condition OK")
