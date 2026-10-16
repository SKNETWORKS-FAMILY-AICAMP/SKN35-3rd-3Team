from pathlib import Path
from streamlit.testing.v1 import AppTest

app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run()
app.session_state["view"] = "판례·법령 검색"
app.session_state["history"] = [{"role": "user", "content": "중고거래 문제를 어떻게 해결하나요?"}]
app.session_state["result"] = {
    "answer": "첫 번째 요약 문단입니다. [S1]\n\n두 번째 상세 설명입니다. [S2]",
    "citation_validation_status": "passed_with_missing_metadata",
    "sources": [
        {"source_id": "S1", "document_id": "guide1", "document_type": "guide", "source_name": "안내 자료", "text": "첫 번째 본문", "source_url": "https://example.com/guide"},
        {"source_id": "S2", "document_id": "guide1", "document_type": "guide", "source_name": "안내 자료", "text": "두 번째 본문", "source_url": "https://example.com/guide"},
        {"source_id": "S3", "document_id": "law1", "document_type": "law", "source_name": "법령 자료", "article": "제1조", "text": "법령 본문"},
        {"source_id": "S4", "document_id": "case1", "document_type": "precedent", "source_name": "판례 자료", "text": "판례 본문"},
    ],
}
app.run()
assert not app.exception, str(app.exception)
assert len([b for b in app.button if b.label == "근거 자료 상세 보기"]) == 3
assert any("안내 자료 1건" in m.value for m in app.markdown)
next(b for b in app.button if b.key == "chunk_1").click().run()
assert not app.exception, str(app.exception)
assert any("두 번째 본문" in item.value for item in app.text)
next(b for b in app.button if b.label == "분석 결과로 돌아가기").click().run()
next(b for b in app.button if b.label == "후속 질문하기").click().run()
assert app.session_state["view"] == "AI 법률상담"
assert not app.exception, str(app.exception)
app.session_state["view"] = "판례·법령 검색"
app.session_state["result"] = {"answer": "서비스 연결 오류", "error_type": "service_error"}
app.run()
assert not app.exception, str(app.exception)
assert not [b for b in app.button if b.label == "근거 자료 상세 보기"]
print("Report grouping, per-chunk detail, follow-up and error rendering OK")
