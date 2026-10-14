import ast
from pathlib import Path
for name in ["main.py", "src/app/views.py"]:
    ast.parse(Path(name).read_text(encoding="utf-8"))
from streamlit.testing.v1 import AppTest
app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run(timeout=30)
assert not app.exception, str(app.exception)
assert any("복잡한 법률" in item.value for item in app.markdown)
for label in ["AI 법률상담", "판례·법령 검색", "이용 안내", "홈"]:
    next(b for b in app.button if b.label == label).click().run(timeout=30)
    assert not app.exception, str(app.exception)
print("Syntax and all four views OK")
