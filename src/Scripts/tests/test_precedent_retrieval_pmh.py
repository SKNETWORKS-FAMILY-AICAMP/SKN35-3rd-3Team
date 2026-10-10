"""27건 판례의 로컬 문자 n-gram BM25 검색 기준 평가. 임베딩/LLM RAG는 아님.

uv run python -m src.Scripts.tests.test_precedent_retrieval_pmh
"""

import json
import math
import re
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CHUNKS_DIR = ROOT / "data/chunk/precedents"
EVALUATION_DIR = ROOT / "data/processed/retrieval_evaluation"
# 소규모 수동 정의 질문. expected_ids는 최소 한 건의 관련 판례 기준이며 모든 정답의 목록은 아니다.
QUESTIONS = [
    ("rent_guarantee", "전세계약이 끝났는데 보증금을 못 받았어요. 주택도시보증공사에 지급을 청구했는데 늦어지면 지연손해금을 받을 수 있나요?", ["622247", "622837", "622971"]),
    ("rent_priority", "전세보증금 반환채권을 주택도시보증공사에 양도했는데 집주인이 바뀌었어요. 경매에서 우선변제권을 행사할 수 있나요?", ["616251"]),
    ("broker", "다가구주택 계약 때 공인중개사가 선순위 보증금을 제대로 설명하지 않아서 손해를 봤어요.", ["623049"]),
    ("dismissal_return", "부당해고 구제신청 후 회사가 해고를 취소하고 복직하라고 했어요. 금전보상명령을 받을 수 있나요?", ["605067"]),
    ("rehire", "경영상 이유로 해고당했는데 회사에서 새 직원을 뽑았어요. 우선 재고용의무가 있나요?", ["614661"]),
    ("freelancer", "프리랜서 쇼핑호스트로 일했는데 해고됐어요. 근로기준법상 근로자로 인정될까요?", ["606973"]),
    ("wage", "퇴직했는데 회사가 임금과 퇴직금을 지급하지 않았어요. 근로기준법 위반으로 처벌할 수 있나요?", ["234477", "227759", "241141"]),
    ("used_non_delivery", "중고나라에서 미개봉 물건을 산다고 돈을 송금했는데 판매자가 물건을 보내지 않았어요. 사기인가요?", ["605583"]),
    ("used_third_party", "중고 목걸이를 팔았는데 보이스피싱 피해자 돈이 제 계좌에 들어왔어요. 부당이득으로 반환해야 하나요?", ["240967", "241143"]),
    ("music_cancel", "음악 이용권을 해지하려는데 앱에서 중도해지와 환불 방법을 알려주지 않아요. 계약해지 방해인가요?", ["613301"]),
]
UNCOVERED = ["헬스장 이용권을 중도해지하면 환불은 얼마나 받나요?",
             "인터넷 쇼핑몰에서 옷을 샀는데 단순변심 반품 배송비는 누가 부담하나요?",
             "택배가 분실됐는데 누가 배상해야 하나요?"]


def make_chunks(document, size=900, overlap=120):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("청크 크기와 겹침이 유효하지 않습니다.")
    text = document["retrieval_text"]
    chunks = []
    segments = document.get("retrieval_segments") or [{"start": 0, "end": len(text)}]
    if document.get("selected_spans") and document["retrieval_scope"] == "selected_spans" and not document.get("retrieval_segments"):
        raise ValueError("부분 사용 판례를 다시 전처리해 역할 구간을 생성해주세요.")
    for segment in segments:
        start = segment["start"]
        stop = segment["end"]
        while start < stop:
            end = min(start + size, stop)
            if end < stop:
                boundary = text.rfind("\n", start + size // 2, end)
                if boundary >= 0:
                    end = boundary + 1
            chunk = {"chunk_id": f"{document['precedent_id']}_{len(chunks):04d}",
                     "precedent_id": document["precedent_id"], "text": text[start:end],
                     "start": start, "end": end, "offset_basis": "retrieval_text",
                     "case_name": document["case_name"], "case_number": document["case_number"],
                     "court": document["court"], "judgment_date": document["judgment_date"],
                     "domain": document["domain"], "source_url": document["source_url"],
                     "retrieval_scope": document["retrieval_scope"], "warnings": document["warnings"]}
            chunk["role"] = segment.get("role", "full_document")
            chunk["section_heading"] = segment.get("heading", "판례 검색 본문")
            chunk["context_header"] = segment.get("context_header", "")
            chunk["embedding_text"] = (chunk["context_header"] + "\n" + chunk["text"]).strip()
            chunks.append(chunk)
            if end == stop:
                break
            start = max(start + 1, end - overlap)
    return chunks


def tokens(text):
    text = re.sub(r"[^가-힣a-z0-9]", "", text.lower())
    return [text[i:i+n] for n in (2, 3) for i in range(len(text)-n+1)]


class BM25:
    def __init__(self, chunks):
        self.chunks = chunks
        self.counts = [Counter(tokens(c.get("embedding_text", c["text"]))) for c in chunks]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(1, len(chunks)) or 1
        df = Counter(term for counts in self.counts for term in counts)
        self.idf = {term: math.log(1 + (len(chunks)-count+0.5)/(count+0.5)) for term, count in df.items()}

    def search(self, query, top_k=3):
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0:
            raise ValueError("top_k는 0 이상의 정수여야 합니다.")
        if top_k == 0:
            return []
        terms = set(tokens(query))
        scored = []
        for chunk, counts, length in zip(self.chunks, self.counts, self.lengths):
            score = sum(self.idf.get(t, 0) * counts[t] * 2.5 /
                        (counts[t] + 1.5*(0.25+0.75*length/self.average)) for t in terms if counts[t])
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]["chunk_id"]))
        results, seen = [], set()
        for score, chunk in scored:
            if chunk["precedent_id"] in seen:
                continue
            seen.add(chunk["precedent_id"])
            results.append({**chunk, "score": round(score, 5)})
            if len(results) == top_k:
                break
        return results


def evaluate():
    manifest = json.loads((ROOT / "data/processed/precedent_preprocessing_report.json").read_text(encoding="utf-8"))
    documents = []
    for identifier in manifest["saved"]:
        doc = json.loads((ROOT / "data/processed/precedents" / f"{identifier}.json").read_text(encoding="utf-8"))
        if doc["embedding_candidate"] and not doc["needs_manual_review"]:
            documents.append(doc)
    chunks = [c for d in documents for c in make_chunks(d)]
    index = BM25(chunks)
    results = []
    for identifier, query, expected in QUESTIONS:
        hits = index.search(query)
        rank = next((i for i,h in enumerate(hits, 1) if h["precedent_id"] in expected), None)
        results.append({"question_id": identifier, "query": query, "expected_ids": expected,
                        "hit_at_3": rank is not None, "reciprocal_rank_at_3": 1/rank if rank else 0,
                        "results": hits})
    coverage = [{"query": q, "expected_status": "판례 데이터셋에서 직접 답변할 근거 미확보",
                 "results": index.search(q)} for q in UNCOVERED]
    report = {"evaluation_type": "offline_char_ngram_bm25_baseline", "document_count": len(documents),
              "chunk_count": len(chunks), "chunk_size": 900, "overlap": 120,
              "hit_at_3": sum(r["hit_at_3"] for r in results)/len(results),
              "mrr_at_3": sum(r["reciprocal_rank_at_3"] for r in results)/len(results),
              "questions": results, "coverage_gap_questions": coverage,
              "limitations": ["수동 정의 10문항의 소규모 기준 평가. 미사용 질문에 대한 일반화 성능이 아님.",
                  "LLM 답변·임베딩·Qdrant·자동 답변거절 기능은 평가하지 않음.",
                  "BM25 점수는 신뢰확률이 아님. 근거 미확보 질문도 결과가 나올 수 있음."]}
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    (CHUNKS_DIR / "chunks.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False)+"\n" for c in chunks), encoding="utf-8")
    (EVALUATION_DIR / "baseline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 판례 검색 기준 평가", "", f"대상 {len(documents)}건 / 청크 {len(chunks)}개", "",
             f"Hit@3: {report['hit_at_3']:.0%} / MRR@3: {report['mrr_at_3']:.3f}", "",
             "문자 n-gram BM25 로컬 평가이며 전체 RAG 검증은 아닙니다.", "",
             "| 질문 | 기대 판례 | 검색 TOP 3 | 적중 |", "|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['query']} | {', '.join(r['expected_ids'])} | {', '.join(h['precedent_id'] for h in r['results'])} | {r['hit_at_3']} |")
    lines += ["", "## 근거 부족 질문", ""] + ["- "+q for q in UNCOVERED]
    lines += ["", "## 평가 한계", ""] + ["- "+x for x in report["limitations"]]
    (EVALUATION_DIR / "baseline_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"평가 완료: 판례 {len(documents)}건, 청크 {len(chunks)}개, Hit@3={report['hit_at_3']:.0%}, MRR@3={report['mrr_at_3']:.3f}")
    return report


class RetrievalTests(unittest.TestCase):
    def test_chunks_cover_text_without_loss_and_keep_provenance(self):
        doc = {"retrieval_text": "가나다 문단\n"*400, "precedent_id": "1", "case_name": "사건",
               "case_number": "2020다1", "court": "법원", "judgment_date": "2020-01-01",
               "domain": "분야", "source_url": "https://www.law.go.kr", "retrieval_scope": "selected_spans", "warnings": []}
        chunks = make_chunks(doc)
        covered = set()
        for c in chunks:
            self.assertEqual(c["text"], doc["retrieval_text"][c["start"]:c["end"]])
            covered.update(range(c["start"], c["end"]))
            self.assertEqual(c["case_number"], "2020다1")
        self.assertEqual(len(covered), len(doc["retrieval_text"]))

    def test_retrieval_deduplicates_documents_and_ignores_unmatched_terms(self):
        index = BM25([{"chunk_id": "a1", "precedent_id": "a", "text": "보증금 반환 보증금"},
                      {"chunk_id": "a2", "precedent_id": "a", "text": "보증금"},
                      {"chunk_id": "b1", "precedent_id": "b", "text": "근로자 해고"}])
        self.assertEqual([h["precedent_id"] for h in index.search("보증금 반환")], ["a"])
        self.assertEqual(index.search("zzzz"), [])
        self.assertEqual(index.search("보증금", top_k=0), [])
        for invalid in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                index.search("보증금", top_k=invalid)

    def test_partial_chunks_keep_roles_and_do_not_cross_spans(self):
        for identifier in ("227759", "234477", "605583"):
            doc = json.loads((ROOT / "data/processed/precedents" / f"{identifier}.json").read_text(encoding="utf-8"))
            for chunk in make_chunks(doc):
                segment = next(s for s in doc["retrieval_segments"] if s["start"] <= chunk["start"] < s["end"])
                self.assertLessEqual(chunk["end"], segment["end"])
                self.assertEqual(chunk["role"], segment["role"])
                self.assertTrue(chunk["embedding_text"].startswith(segment["context_header"]))
                self.assertEqual(chunk["text"], doc["retrieval_text"][chunk["start"]:chunk["end"]])


if __name__ == "__main__":
    evaluate()
