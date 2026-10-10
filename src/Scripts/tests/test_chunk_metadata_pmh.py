"""공통 메타데이터, 보존성, 실제 저장 데이터 및 생성 코드의 일관성 검증."""
import copy
import json
import unittest
from pathlib import Path

from src.Scripts.chunk_metadata import normalize_chunk, validate_chunk, normalize_date
from src.Scripts import chunk_laws, chunk_guides, chunk_consumer_sources
from src.Scripts.tests.test_precedent_retrieval_pmh import make_chunks as precedent_chunks

ROOT = Path(__file__).resolve().parents[3]


def corpus():
    result = []
    for folder in ("laws", "precedents", "consumer_sources"):
        path = ROOT / "data/chunk" / folder / "chunks.jsonl"
        result.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    root = ROOT / "data/chunk/guides"
    manifest = json.loads((root / "guide_chunking_report.json").read_text(encoding="utf-8"))
    for entry in manifest["saved"]:
        result.extend(json.loads((root / entry["file"]).read_text(encoding="utf-8"))["chunks"])
    return result


class MetadataTests(unittest.TestCase):
    def precedent(self):
        return {"chunk_id": "123_0000", "precedent_id": "123", "case_name": "사건명",
                "domain": "해고·임금", "source_url": "https://www.law.go.kr",
                "judgment_date": "20240123", "text": "본문", "embedding_text": "경고\n본문",
                "warnings": ["원문 이상"], "calculation_restricted_clause": "단위 미확인"}

    def test_preserves_original_metadata_text_and_cautions_without_mutation(self):
        original = self.precedent()
        before = copy.deepcopy(original)
        normalized = normalize_chunk(original, {"source_file": "data/raw/precedents/123.json"})
        self.assertEqual(original, before)
        for key, value in original.items():
            self.assertEqual(normalized[key], value)
        self.assertEqual(normalized["document_type"], "precedent")
        self.assertEqual(normalized["domains"], ["labor_dismissal", "labor_wage"])
        self.assertEqual(normalized["dates"]["judgment_date"], "2024-01-23")
        self.assertIsNone(normalized["dates"]["effective_date"])
        self.assertEqual(normalized["source_path"], "data/raw/precedents/123.json")
        self.assertEqual(normalize_chunk(normalized), normalized)

    def test_unknown_domain_and_missing_source_are_not_invented(self):
        value = self.precedent()
        value["domain"] = "미분류 분야"
        normalized = normalize_chunk(value)
        self.assertEqual(normalized["domains"], [])
        self.assertEqual(normalized["domain"], "미분류 분야")
        self.assertEqual(normalized["domain_classification"], "unmapped")
        self.assertIsNone(normalized["source_path"])

    def test_broad_laws_are_not_assigned_every_dispute_domain(self):
        value = {"chunk_id": "law1", "document_type": "law", "version_id": "v1",
                 "law_name": "민법", "source_file": "v1.json", "text": "목적", "embedding_text": "목적"}
        normalized = normalize_chunk(value)
        self.assertEqual(normalized["domains"], ["civil_general"])
        self.assertEqual(normalized["source_path"], "data/raw/laws/v1.json")
        self.assertEqual(normalized["document_id"], "law:v1")

    def test_invalid_dates_and_unsafe_paths_fail(self):
        for value in ("20240230", "2024", "어제", 20240123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_date(value)
        for value in ("../secret.json", "C:/secret.json", "/secret.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_chunk(self.precedent(), {"source_file": value})

    def test_empty_text_and_duplicate_domains_fail_validation(self):
        normalized = normalize_chunk(self.precedent())
        normalized["text"] = " "
        with self.assertRaises(ValueError):
            validate_chunk(normalized)
        normalized = normalize_chunk(self.precedent())
        normalized["domains"] *= 2
        with self.assertRaises(ValueError):
            validate_chunk(normalized)

    def test_saved_corpus_has_valid_metadata_unique_ids_and_existing_sources(self):
        chunks = corpus()
        self.assertTrue(chunks)
        self.assertEqual(len(chunks), len({c["chunk_id"] for c in chunks}))
        for chunk in chunks:
            with self.subTest(chunk_id=chunk["chunk_id"]):
                validate_chunk(chunk)
                self.assertEqual(normalize_chunk(chunk), chunk)
                self.assertTrue(chunk["domains"], "현재 확보한 자료의 분류 매핑 누락")
                self.assertIsNotNone(chunk["source_path"])
                self.assertTrue((ROOT / chunk["source_path"]).is_file())

    def test_saved_chunks_match_all_four_generators(self):
        stored = {c["chunk_id"]: c for c in corpus()}
        generated = []
        manifest = json.loads((ROOT / "data/processed/laws/law_preprocessing_report.json").read_text(encoding="utf-8"))
        for entry in manifest["saved"]:
            doc = json.loads((ROOT / "data/processed/laws" / entry["file"]).read_text(encoding="utf-8"))
            generated.extend(c for a in doc["articles"] for c in chunk_laws.make_chunks(doc, a))
            generated.extend(c for u in doc["supplements"] + doc["appendices"] for c in chunk_laws.make_aux_chunks(doc, u))
        manifest = json.loads((ROOT / "data/processed/guides/guide_preprocessing_report.json").read_text(encoding="utf-8"))
        for entry in manifest["saved"]:
            doc = json.loads((ROOT / "data/processed/guides" / entry["file"]).read_text(encoding="utf-8"))
            generated.extend(chunk_guides.make_chunks(doc))
        manifest = json.loads((ROOT / "data/processed/consumer_sources/consumer_preprocessing_report.json").read_text(encoding="utf-8"))
        for entry in manifest["saved"]:
            doc = json.loads((ROOT / "data/processed/consumer_sources" / entry["file"]).read_text(encoding="utf-8"))
            generated.extend(chunk_consumer_sources.make_chunks(doc))
        manifest = json.loads((ROOT / "data/processed/precedent_preprocessing_report.json").read_text(encoding="utf-8"))
        for identifier in manifest["saved"]:
            doc = json.loads((ROOT / "data/processed/precedents" / f"{identifier}.json").read_text(encoding="utf-8"))
            if doc["embedding_candidate"] and not doc["needs_manual_review"]:
                generated.extend(precedent_chunks(doc))
        self.assertEqual(len(generated), len(stored))
        for chunk in generated:
            with self.subTest(chunk_id=chunk["chunk_id"]):
                self.assertEqual(chunk, stored[chunk["chunk_id"]])
