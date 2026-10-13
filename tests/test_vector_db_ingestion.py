"""Vector DB 임베딩 배치 흐름의 순서와 기존 벡터 재사용을 검증합니다."""

from __future__ import annotations

import unittest

from src.const.qdrant_config import EmbeddingSettings
from src.const.vector_data_contract import PreparedChunk
from src.vector_db.ingestion import generate_document_vectors


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        return [[float(len(self.calls)), 0.0] for _ in texts]


def _chunk(
    chunk_id: str,
    text: str,
    *,
    vector: tuple[float, ...] | None = None,
) -> PreparedChunk:
    return PreparedChunk(
        document_id="document:1",
        chunk_id=chunk_id,
        text=text,
        vector=vector,
        source_name="테스트 문서",
    )


class GenerateDocumentVectorsTests(unittest.TestCase):
    def test_embeds_only_missing_vectors_and_preserves_record_order(self):
        embedder = _RecordingEmbedder()
        settings = EmbeddingSettings(
            provider="openai",
            model="text-embedding-3-small",
            dimension=2,
            normalized=True,
        )
        records = (
            _chunk("chunk-1", "첫 번째"),
            _chunk("chunk-2", "기존 벡터", vector=(0.5, 0.5)),
            _chunk("chunk-3", "세 번째"),
        )

        result = generate_document_vectors(
            records,
            embedder=embedder,
            settings=settings,
            batch_size=1,
        )

        self.assertEqual(embedder.calls, [["첫 번째"], ["세 번째"]])
        self.assertEqual([record.chunk_id for record in result], [
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ])
        self.assertEqual(result[0].vector, (1.0, 0.0))
        self.assertEqual(result[1].vector, (0.5, 0.5))
        self.assertEqual(result[2].vector, (2.0, 0.0))
        self.assertEqual(result[0].embedding_model, settings.model)
        self.assertIsNone(result[1].embedding_model)


if __name__ == "__main__":
    unittest.main()
