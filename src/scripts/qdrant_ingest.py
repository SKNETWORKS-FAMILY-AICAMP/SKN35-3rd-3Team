"""전처리 청크의 사전 점검과 Qdrant 적재를 분리해 실행합니다.

사용 예시:
    uv run python -m src.scripts.qdrant_ingest check-input
    uv run python -m src.scripts.qdrant_ingest ingest
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.const.config import PROJECT_PATH
from src.const.model import env_loader
from src.const.qdrant_config import load_qdrant_settings
from src.embedder import create_embedding_provider
from src.vector_db.chunk_loader import LoadedChunkSet, load_chunk_tree
from src.vector_db.ingestion import (
    generate_document_vectors,
    prepare_upsert_points,
)
from src.vector_db.qdrant_store import QdrantVectorStore
from src.vector_db.sparse_embedder import create_sparse_embedding_provider
from src.vector_db.validation import validate_qdrant_ingestion
from src.vectorstore import VectorStoreError


DEFAULT_CHUNK_ROOT = PROJECT_PATH / "data" / "chunk"
DEFAULT_INPUT_REPORT = (
    PROJECT_PATH / "tests" / "report" / "qdrant_input_validation.md"
)
DEFAULT_INGESTION_REPORT = (
    PROJECT_PATH / "tests" / "report" / "qdrant_ingestion_validation.md"
)


def _print_json(title: str, value: object) -> None:
    print(title)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def _load_and_report(
    data_root: Path,
    report_path: Path,
) -> tuple[LoadedChunkSet, bool]:
    loaded = load_chunk_tree(data_root)
    saved_path = loaded.report.write_markdown(report_path)
    _print_json("[전처리 청크 사전 점검]", loaded.report.to_dict())
    print(f"점검 보고서: {saved_path}")
    return loaded, loaded.report.passed


def check_input(args: argparse.Namespace) -> int:
    """외부 API 호출 없이 전처리 입력만 점검합니다."""

    _, passed = _load_and_report(args.data_root, args.input_report)
    return 0 if passed else 2


def ingest(args: argparse.Namespace) -> int:
    """검증된 청크를 임베딩하고 기존 Qdrant 컬렉션에 적재합니다."""

    loaded, passed = _load_and_report(args.data_root, args.input_report)
    if not passed:
        print(
            "입력 점검에 실패해 임베딩과 Qdrant 적재를 중단했습니다.",
            file=sys.stderr,
        )
        return 2

    env_loader()
    settings = load_qdrant_settings()
    settings.validate_for_vector_work()
    if settings.collection.search_mode != "hybrid":
        print(
            "팀 컬렉션 적재는 Dense와 BM25 Sparse Vector를 함께 저장하도록 "
            "SEARCH_MODE=hybrid에서만 실행할 수 있습니다. Dense 단독 모드는 "
            "적재 완료 후 검색 비교용으로 사용하세요.",
            file=sys.stderr,
        )
        return 2

    embedder = create_embedding_provider(settings.embedding)
    store = QdrantVectorStore.from_settings(settings)
    collection_info = store.validate_collection_contract()

    embedded_records = generate_document_vectors(
        loaded.records,
        embedder=embedder,
        settings=settings.embedding,
        batch_size=args.embedding_batch_size,
    )
    assert settings.collection.sparse_model is not None
    sparse_embedder = create_sparse_embedding_provider(
        settings.collection.sparse_model
    )
    sparse_vectors = sparse_embedder.embed_documents(
        [record.embedding_input_text for record in embedded_records],
        batch_size=args.embedding_batch_size,
    )
    points = prepare_upsert_points(
        embedded_records,
        expected_dimension=collection_info.vector_size,
        sparse_vector_name=collection_info.sparse_vector_name,
        sparse_vectors=sparse_vectors,
    )
    saved_count = store.upsert_points(
        points,
        batch_size=args.upsert_batch_size,
        parallel=args.parallel,
        max_retries=args.max_retries,
    )
    report = validate_qdrant_ingestion(store, points)
    report_path = report.write_markdown(args.ingestion_report)

    _print_json(
        "[Qdrant 적재 완료]",
        {
            "collection": collection_info.name,
            "embedding": settings.embedding.safe_summary(),
            "input_count": len(loaded.records),
            "saved_count": saved_count,
            "validation_passed": report.passed,
            "validation_report": str(report_path),
        },
    )
    return 0 if report.passed else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="data/chunk 입력 점검과 Qdrant 적재",
    )
    parser.add_argument(
        "command",
        choices=("check-input", "ingest"),
        help="check-input: 로컬 입력만 점검, ingest: 임베딩·적재·검증",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_CHUNK_ROOT,
        help="laws, guides, precedents, consumer_sources가 있는 chunk 경로",
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        default=DEFAULT_INPUT_REPORT,
    )
    parser.add_argument(
        "--ingestion-report",
        type=Path,
        default=DEFAULT_INGESTION_REPORT,
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--upsert-batch-size", type=int, default=64)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "check-input":
            return check_input(args)
        return ingest(args)
    except VectorStoreError as error:
        print(
            f"Qdrant 오류 [{error.code}]: {error.safe_message}",
            file=sys.stderr,
        )
        return 1
    except (OSError, TypeError, ValueError, RuntimeError) as error:
        print(f"실행 오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

