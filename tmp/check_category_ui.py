from pathlib import Path
from streamlit.testing.v1 import AppTest
root = Path(__file__).resolve().parents[1]
app = AppTest.from_file(str(root / "main.py")).run(timeout=30)
assert not app.exception, str(app.exception)
assert sum('class="law-category-visual"' in m.value for m in app.markdown) == 4
assert not any(b.label == "상담하기" for b in app.button)
next(b for b in app.button if b.label == "주택임대차 상담 시작").click().run()
assert not app.exception, str(app.exception)
assert app.session_state["view"] == "AI 법률상담"
print("Four horizontal cards and consultation navigation OK")
