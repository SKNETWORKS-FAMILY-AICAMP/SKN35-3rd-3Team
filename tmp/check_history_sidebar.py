from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from streamlit.testing.v1 import AppTest
from src.app.conversation_history import KST, group_consultations, category

now = datetime.now(KST)
def item(question, days=0, **extra):
    return {"title": question, "history": [{"role": "user", "content": question}],
            "result": {"answer": "기존 답변"}, "updated_at": (now-timedelta(days=days)).isoformat(), **extra}

records = {"a": item("퇴직금이 궁금해요"), "b": item("전세 보증금 문의", 3),
           "c": {"title": "오래된 기록", "history": [], "result": {}}}
groups = group_consultations(records, now=now)
assert list(groups) == ["오늘", "지난 7일", "이전 기록"]
assert category(records["a"]) == "노동/근로"
assert category(records["b"]) == "부동산"

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run(timeout=30)
app.session_state["consultations"] = records
app.session_state["thread_id"] = "a"
app.session_state["history"] = records["a"]["history"].copy()
app.run(timeout=30)
assert not app.exception, str(app.exception)
next(t for t in app.text_input if t.key == "rename_title_a").set_value("나의 퇴직금 상담")
next(b for b in app.button if b.label == "이름 변경").click().run(timeout=30)
assert not app.exception, str(app.exception)
assert app.session_state["consultations"]["a"]["custom_title"]
assert next(b for b in app.button if b.key == "history_a").label == "나의 퇴직금 상담"

def stream(*args, **kwargs):
    yield (), "updates", {"general": {"answer": "새 답변"}}
    yield (), "values", {"answer": "새 답변"}
app.session_state["graph"] = SimpleNamespace(stream=stream)
app.chat_input[0].set_value("추가 질문").run(timeout=30)
assert not app.exception, str(app.exception)
assert app.session_state["consultations"]["a"]["title"] == "나의 퇴직금 상담"
assert app.session_state["consultations"]["a"]["created_at"]
app.run(timeout=30)
next(b for b in app.button if b.key == "delete_a").click().run(timeout=30)
assert not app.exception, str(app.exception)
assert "a" not in app.session_state["consultations"]
assert app.session_state["history"] == []
assert app.session_state["result"] == {}
assert app.session_state["thread_id"] != "a"
next(b for b in app.button if b.key == "restore_consultation").click().run(timeout=30)
assert "a" in app.session_state["consultations"]
next(r for r in app.radio if r.key == "history_group_mode").set_value("분야별").run(timeout=30)
assert not app.exception, str(app.exception)
assert app.session_state["history_group_mode"] == "분야별"
next(b for b in app.button if b.key == "history_a").click().run(timeout=30)
assert not app.exception, str(app.exception)
assert app.session_state["thread_id"] == "a"
assert app.session_state["result"]["answer"] == "새 답변"
print("Date/category grouping, rename persistence, active delete, undo and restore OK")
