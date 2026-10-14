from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.app.progress import new_progress, advance_progress, finish_progress
import streamlit as st

doc = {"text": "Actual retrieved snippet", "metadata": {"source_name": "Test law", "article_label": "Article 1", "document_type": "law"}}
observed_steps = []

def stream(*args, **kwargs):
    assert kwargs["subgraphs"] is True
    assert args[0]["sources"] == []
    for name, value in [
        ("query_rewrite", {}), ("rrf", {"documents": [doc]}),
        ("rerank", {"documents": [doc]}), ("validation", {"is_valid": True}),
        ("generator", {"answer": "Mock answer", "generation_status": "generated_unverified"}),
    ]:
        yield ("rag:test",), "updates", {name: value}
        observed_steps.append(tuple(st.session_state.rag_progress["steps"]))
    yield (), "values", {"answer": "Mock answer", "documents": [doc], "generation_status": "generated_unverified"}

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run(timeout=30)
assert not app.exception, str(app.exception)
app.session_state["graph"] = SimpleNamespace(stream=stream)
with patch.object(st, "caption", wraps=st.caption) as captions, patch.object(st, "html", wraps=st.html) as html:
    app.chat_input[0].set_value("Test question").run(timeout=30)
    assert any("검색된 근거 후보 1개" in str(call.args[0]) for call in captions.call_args_list)
    assert any("law-loading-art" in str(call.args[0]) and "<style>" in str(call.args[0]) for call in html.call_args_list)
assert not app.exception, str(app.exception)
app.run(timeout=30)
assert not app.exception, str(app.exception)
assert app.session_state["rag_progress"]["steps"] == ["done"] * 4
assert app.session_state["result"]["answer"] == "Mock answer"
assert any("Test law" in exp.label for exp in app.expander)
assert any("Actual retrieved snippet" in text.value for text in app.text)
assert ("done", "active", "waiting", "waiting") in observed_steps
assert ("done", "done", "done", "active") in observed_steps
assert not any('class="law-loading"' in str(getattr(element, "proto", "")) for element in app.get("html"))

p = new_progress()
advance_progress(p, "general", {})
finish_progress(p, {"answer": "Hello"})
assert p["steps"] == ["done", "skipped", "skipped", "done"]
p = new_progress()
advance_progress(p, "rerank", {"documents": [doc]})
advance_progress(p, "validation", {"is_valid": False})
assert p["documents"] == [] and p["steps"][1] == "active"
finish_progress(p, {"error_type": "service_error"})
assert p["status"] == "error" and "active" not in p["steps"]
print("Live progress, source popup, retry and failure states OK")

# Verify the real installed LangGraph emits the nested tuple format used by ask.
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
class Sample(TypedDict):
    value: int
inner = StateGraph(Sample)
inner.add_node("child", lambda s: {"value": s["value"] + 1})
inner.add_edge(START, "child")
inner.add_edge("child", END)
outer = StateGraph(Sample)
outer.add_node("nested", inner.compile())
outer.add_edge(START, "nested")
outer.add_edge("nested", END)
events = list(outer.compile().stream({"value": 0}, stream_mode=["updates", "values"], subgraphs=True))
assert all(len(event) == 3 for event in events)
assert any(ns and mode == "updates" for ns, mode, data in events)
assert [data for ns, mode, data in events if not ns and mode == "values"][-1]["value"] == 1
print("Real LangGraph nested stream format OK")
