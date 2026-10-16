"""RAG 품질·속도 개선 규칙의 외부 서비스 없는 회귀 테스트."""

import unittest

from src.graph.rag_graph.validation_node import validation_node
from src.rag.multi_query import search_rankings
from src.rag.rerank import _preserve_primary_laws, _rerank_question


class FakeSearcher:
    def __init__(self):
        self.calls = []

    def search_documents(self, query, *, top_k, filters):
        self.calls.append((query, top_k, filters))
        return (f"{query}:{filters}",)


def document(
    name,
    score,
    document_type,
    article_number=None,
    legal_rank=None,
    precedent_rank=None,
):
    metadata = {"document_type": document_type}
    if article_number is not None:
        metadata["article_number"] = article_number
    rrf_sources = {}
    if legal_rank is not None:
        rrf_sources["legal_primary_source"] = {"rank": legal_rank}
    if precedent_rank is not None:
        rrf_sources["precedent_source"] = {"rank": precedent_rank}
    if rrf_sources:
        metadata["rrf_sources"] = rrf_sources
    return {
        "point_id": name,
        "text": name,
        "score": score,
        "metadata": metadata,
    }


class RagPerformanceImprovementTests(unittest.TestCase):
    def test_primary_law_and_precedent_searches_are_added_for_unfiltered_query(self):
        searcher = FakeSearcher()
        rankings = search_rankings(searcher, ["자연어", "법률 핵심어"], candidate_k=20)

        self.assertEqual(
            list(rankings),
            ["query_0", "query_1", "legal_primary_source", "precedent_source"],
        )
        self.assertEqual(searcher.calls[-2], ("법률 핵심어", 20, {"document_type": "law"}))
        self.assertEqual(searcher.calls[-1], ("자연어", 20, {"document_type": "precedent"}))

    def test_explicit_document_type_is_respected(self):
        searcher = FakeSearcher()
        search_rankings(
            searcher,
            ["질문", "핵심어"],
            filters={"document_type": "official_guide"},
        )

        self.assertEqual(len(searcher.calls), 2)
        self.assertTrue(all(call[2] == {"document_type": "official_guide"} for call in searcher.calls))

    def test_rerank_question_includes_legal_keyword_query(self):
        question = _rerank_question({
            "origin_query": "원문",
            "rewrite_query": "알기 쉬운 질문",
            "queries": ["알기 쉬운 질문", "관련 질문", "민법 차임 감액청구"],
        })

        self.assertEqual(question, "알기 쉬운 질문\n검색 핵심어: 민법 차임 감액청구")

    def test_final_evidence_keeps_two_primary_laws(self):
        ranked = [
            document("guide-1", .9, "official_guide"),
            document("case-1", .8, "mediation_case"),
            document("guide-2", .7, "official_guide"),
            document("case-2", .6, "mediation_case"),
            document("guide-3", .5, "official_guide"),
            document("law-1", .4, "law"),
            document("law-2", .3, "law"),
        ]

        selected = _preserve_primary_laws(ranked, top_k=5)

        self.assertEqual(len(selected), 5)
        self.assertEqual(sum(doc["metadata"]["document_type"] == "law" for doc in selected), 2)
        self.assertTrue(all(
            doc["metadata"].get("selection_reason") == "primary_law_source_coverage"
            for doc in selected if doc["metadata"]["document_type"] == "law"
        ))

    def test_referenced_articles_are_kept_for_multi_issue_question(self):
        ranked = [
            document("guide-1", .9, "official_guide"),
            document("guide-2", .8, "official_guide"),
            document("wrong-law", .7, "law", "22", 1),
            document("guide-3", .6, "official_guide"),
            document("guide-4", .5, "official_guide"),
            document("article-17", .4, "law", "17", 8),
            document("article-18", .3, "law", "18", 2),
        ]

        selected = _preserve_primary_laws(
            ranked, 5, "전자상거래법 제17조 청약철회 제18조 대금 환급",
        )

        selected_ids = {doc["point_id"] for doc in selected}
        self.assertIn("article-17", selected_ids)
        self.assertIn("article-18", selected_ids)

    def test_final_evidence_keeps_primary_laws_and_one_precedent(self):
        ranked = [
            document("guide-1", .9, "official_guide"),
            document("guide-2", .8, "official_guide"),
            document("guide-3", .7, "official_guide"),
            document("guide-4", .6, "official_guide"),
            document("guide-5", .5, "official_guide"),
            document("law-1", .4, "law", legal_rank=1),
            document("law-2", .3, "law", legal_rank=2),
            document("precedent-1", .2, "precedent", precedent_rank=1),
        ]

        selected = _preserve_primary_laws(ranked, top_k=5)
        selected_types = [doc["metadata"]["document_type"] for doc in selected]

        self.assertEqual(selected_types.count("law"), 2)
        self.assertEqual(selected_types.count("precedent"), 1)
        precedent = next(doc for doc in selected if doc["point_id"] == "precedent-1")
        self.assertEqual(precedent["metadata"]["selection_reason"], "precedent_coverage")

    def test_validation_accepts_low_scored_evidence_without_retry(self):
        result = validation_node({
            "documents": [document("law", .01, "law")],
            "score": .01,
            "retry_count": 1,
        })

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["validation_reason"], "usable_evidence_found")

    def test_empty_evidence_retries_only_once(self):
        first = validation_node({"documents": [], "score": None, "retry_count": 1})
        second = validation_node({"documents": [], "score": None, "retry_count": 2})

        self.assertFalse(first["is_valid"])
        self.assertTrue(second["is_valid"])
        self.assertEqual(second["validation_reason"], "empty_evidence_retry_exhausted")


if __name__ == "__main__":
    unittest.main()
