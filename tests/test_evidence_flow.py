"""Generator → Citation → 검색 자료 표시 연결을 외부 API 없이 검증합니다."""

import copy
import json
import unittest
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.app.evidence import display_sources, verified_sources
from src.citation import citation
from src.generator import generator
from src.graph import workflow
from src.graph.mcp_lookup_document import create_mcp_lookup_document_graph_node
from src.mcp_tools.contracts import serialize_tool_response


def documents():
    return [
        {"point_id": "law-1", "text": "신청은 관할 기관에 접수합니다.", "metadata": {
            "document_id": "law:test", "chunk_id": "article-1", "document_type": "law",
            "source_name": "검사 법령", "source_url": "https://example.org/law", "article_label": "제1조",
        }},
        {"point_id": "case-1", "text": "신청 사실이 확인된 사례입니다.", "metadata": {
            "document_id": "precedent:test", "chunk_id": "case-1", "document_type": "precedent",
            "source_name": "검사 판례", "source_url": "https://example.org/case",
        }},
    ]


def output(answer="신청은 관할 기관에 접수합니다.", ids=None):
    return {"status": "answered", "answer": answer, "claims": [
        {"statement": "신청은 관할 기관에 접수합니다.", "source_ids": ids or ["S1"]},
    ]}


class Replies:
    def __init__(self, *outputs):
        self.outputs = outputs
        self.calls = 0

    def invoke(self, messages):
        value = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        content = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        return AIMessage(content=content, usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})


class GeneratorEvidenceTests(unittest.TestCase):
    def run_generation(self, model, docs=None):
        self.state = {"origin_query": "신청 방법을 알려주세요.", "documents": documents() if docs is None else docs}
        return generator(self.state, llm=model, verbose=False)

    def test_missing_inline_marker_is_rendered_from_existing_claim_only(self):
        result = self.run_generation(Replies(output()))
        self.assertEqual(result["generation_status"], "generated_unverified")
        self.assertIn("[S1]", result["answer"])
        self.assertIn("https://example.org/law", result["answer"])
        self.assertNotIn("[S2]", result["answer"])
        checked = citation({**self.state, **result}, verbose=False)
        self.assertEqual(checked["citation_validation_status"], "passed")
        self.assertEqual(len(checked["sources"]), 1)

    def test_pdf_without_url_is_supported_without_fake_link(self):
        docs = documents()[:1]
        docs[0]["metadata"].update(source_url=None, document_type="official_guide")
        result = self.run_generation(Replies(output()), docs)
        checked = citation({**self.state, **result}, verbose=False)
        self.assertEqual(checked["citation_validation_status"], "passed_with_missing_metadata")
        self.assertIn("[S1]", checked["answer"])
        self.assertNotIn("https://", checked["answer"])

    def test_broken_pdf_link_is_replaced_with_real_plain_reference(self):
        docs = documents()[:1]
        docs[0]["metadata"].update(source_url=None, document_type="official_guide")
        result = self.run_generation(Replies(output(answer="신청 안내 [가이드](자료명 [S1])")), docs)
        self.assertEqual(result["generation_status"], "generated_unverified")
        self.assertNotIn("](자료명", result["answer"])
        self.assertIn("검사 법령", result["answer"])
        self.assertIn("[S1]", result["answer"])

    def test_inline_reference_uses_actual_source_name_and_url(self):
        result = self.run_generation(Replies(output(answer="안내 [다른 제목](https://example.org/wrong '[S1]')")))
        self.assertIn("https://example.org/law", result["answer"])
        self.assertNotIn("https://example.org/wrong", result["answer"])
        self.assertNotIn("다른 제목", result["answer"])

    def test_missing_claims_gets_one_model_repair_not_invented_mapping(self):
        model = Replies({"status": "answered", "answer": "누락 응답"}, output())
        result = self.run_generation(model)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["generation_attempts"], 2)
        self.assertEqual(result["token_usage"]["total_tokens"], 40)
        self.assertEqual(result["generation_status"], "generated_unverified")

    def test_repeated_missing_claims_stays_withheld(self):
        model = Replies({"status": "answered", "answer": "누락 응답"})
        result = self.run_generation(model)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["generation_status"], "invalid_output")
        self.assertEqual(result["sources"], [])
        self.assertFalse(result["source_validation_passed"])

    def test_invalid_json_can_be_repaired_once(self):
        model = Replies("not-json", output())
        self.assertEqual(self.run_generation(model)["generation_status"], "generated_unverified")
        self.assertEqual(model.calls, 2)

    def test_unknown_source_is_rejected_without_inventing_or_retrying(self):
        model = Replies(output(ids=["S999"]))
        result = self.run_generation(model)
        self.assertEqual(result["generation_status"], "invalid_output")
        self.assertEqual(model.calls, 1)
        self.assertEqual(result["sources"], [])

    def test_unmapped_numeric_citation_and_unclaimed_source_stay_invalid(self):
        for marker in ("[999]", "[S999]", "[S2]"):
            result = self.run_generation(Replies(output(answer=f"신청 안내 {marker}")))
            self.assertEqual(result["generation_status"], "invalid_output", marker)

    def test_known_numeric_marker_is_normalized(self):
        result = self.run_generation(Replies(output(answer="신청 안내 [ 1 ]")))
        self.assertEqual(result["generation_status"], "generated_unverified")
        self.assertTrue(result["answer"].endswith("[S1]"))

    def test_body_and_claim_id_mismatch_is_repaired_by_model_once(self):
        model = Replies(output(answer="신청 안내 [S2]"), output(answer="신청 안내 [S1]"))
        result = self.run_generation(model)
        self.assertEqual(model.calls, 2)
        self.assertEqual(result["generation_status"], "generated_unverified")
        self.assertNotIn("[S2]", result["answer"])


class EvidenceDisplayTests(unittest.TestCase):
    def test_retrieved_law_and_precedent_survive_failed_generation_as_candidates(self):
        result = {"documents": documents(), "citation_validation_status": "skipped", "sources": []}
        items = display_sources(result)
        self.assertEqual({s["document_type"] for s in items}, {"law", "precedent"})
        self.assertFalse(any(s["is_cited"] for s in items))
        self.assertEqual(verified_sources(result), [])
        self.assertTrue(all(s["text"] for s in items))

    def test_verified_sources_are_deduplicated_without_promoting_other_hits(self):
        state = {"origin_query": "신청 방법", "documents": documents()}
        generated = generator(state, llm=Replies(output()), verbose=False)
        result = {**state, **generated}
        result.update(citation(result, verbose=False))
        before = copy.deepcopy(result)
        items = display_sources(result)
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0]["is_cited"])
        self.assertFalse(items[1]["is_cited"])
        self.assertNotIn("source_id", items[1])
        self.assertEqual(len(verified_sources(result)), 1)
        self.assertEqual(result, before)

    def test_stale_sources_cannot_pass_failed_citation(self):
        result = {"documents": documents(), "sources": [{"source_id": "S99"}],
                  "citation_validation_status": "failed"}
        self.assertEqual(verified_sources(result), [])
        self.assertFalse(any(s["is_cited"] for s in display_sources(result)))


class WorkflowEvidenceTests(unittest.TestCase):
    def test_both_branches_reach_shared_citation_and_general_skips_it(self):
        def generate(state):
            return generator({**state, "documents": documents()}, llm=Replies(output()), verbose=False)

        for route in ("rag", "mcp", "general"):
            with patch.object(workflow, "route_image", new=lambda state, selected=route: selected), \
                 patch.object(workflow, "build_rag_graph", return_value=lambda state: {**generate(state), "documents": documents()}), \
                 patch.object(workflow, "build_mcp_graph", return_value=lambda state: {"documents": documents()}), \
                 patch.object(workflow, "generator_node", new=generate), \
                 patch.object(workflow, "general_agent", new=lambda state: {"answer": "서비스 안내"}):
                graph = workflow.build_workflow()
                result = graph.invoke({"origin_query": "신청 방법", "retry_count": 0, "messages": []},
                                      config={"configurable": {"thread_id": str(uuid.uuid4())}})
            if route == "general":
                self.assertNotIn("citation_validation_status", result)
            else:
                self.assertEqual(result["citation_validation_status"], "passed", route)
                self.assertEqual(len(verified_sources(result)), 1)

    def test_mcp_lookup_preserves_relevant_search_chunk_and_deduplicates(self):
        searched = documents()[0]
        unrelated = documents()[1]
        node = create_mcp_lookup_document_graph_node(lambda *args, **kwargs: serialize_tool_response(
            "success", documents=[unrelated, searched]))
        result = node({"mcp_document_id": "law:test", "mcp_search_documents": [searched]})
        self.assertEqual(result["documents"], [searched, unrelated])


class EvidenceScreenTests(unittest.TestCase):
    def test_report_and_candidate_detail_render_without_live_api(self):
        from streamlit.testing.v1 import AppTest

        state = {"origin_query": "신청 방법", "documents": documents()}
        state.update(generator(state, llm=Replies(output()), verbose=False))
        state.update(citation(state, verbose=False))
        script = (
            "import streamlit as st\n"
            "from unittest.mock import patch\n"
            "from src.app import views\n"
            f"st.session_state.setdefault('result', {state!r})\n"
            "st.session_state.setdefault('history', [{'role': 'user', 'content': '신청 방법'}])\n"
            "st.session_state.setdefault('view', '판례·법령 검색')\n"
            "with patch.object(views, 'report_summary', return_value={'text': '검증 요약', 'method': 'extracted'}):\n"
            "    if st.session_state.view == '판례 상세':\n"
            "        views.render_source_detail()\n"
            "    else:\n"
            "        views.render_analysis_report()\n"
        )
        app = AppTest.from_string(script, default_timeout=30).run()
        self.assertFalse(app.exception)
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("법령 1건", rendered)
        self.assertIn("판례 1건", rendered)
        app.button(key="source_1").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("검색된 후보 자료입니다." in item.value for item in app.caption))


if __name__ == "__main__":
    unittest.main()
