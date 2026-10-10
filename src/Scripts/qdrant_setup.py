"""팀 Qdrant 연결 확인 및 최초 컬렉션 준비 명령.

사용 예시:
    uv run python -m src.Scripts.qdrant_setup check
    uv run python -m src.Scripts.qdrant_setup init

``init``은 컬렉션이 없을 때만 생성하며 기존 컬렉션을 삭제하지 않습니다.
"""

from __future__ import annotations

import argparse
import json
import sys

from src.const.env_loader import load_project_env
from src.const.qdrant_config import load_qdrant_settings
from src.vector_db.qdrant_store import QdrantVectorStore
from src.vectorstore import VectorStoreError


def _print_json(title: str, value: object) -> None:
    print(title)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def check_connection() -> int:
    """키를 노출하지 않고 연결 및 컬렉션 규격을 확인합니다."""

    env_path = load_project_env()
    settings = load_qdrant_settings()
    settings.validate_for_vector_work()
    _print_json("[공용 설정 확인]", settings.safe_summary())
    print(f".env 확인 경로: {env_path}")

    store = QdrantVectorStore.from_settings(settings)
    if not store.health_check():
        print("대상 컬렉션이 없습니다. `init` 명령으로 생성할 수 있습니다.")
        return 2

    info = store.validate_collection_contract()
    _print_json("[Qdrant 연결·컬렉션 확인]", info.to_dict())
    return 0


def initialize_collection() -> int:
    """컬렉션이 없을 때만 생성하고 최종 규격을 검증합니다."""

    env_path = load_project_env()
    settings = load_qdrant_settings()
    settings.validate_for_vector_work()
    print(f".env 확인 경로: {env_path}")

    store = QdrantVectorStore.from_settings(settings)
    info, created = store.ensure_collection()
    _print_json(
        "[Qdrant 컬렉션 준비 완료]",
        {"created": created, **info.to_dict()},
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="팀 Qdrant 연결과 컬렉션 규격을 안전하게 준비합니다.",
    )
    parser.add_argument(
        "command",
        choices=("check", "init"),
        help="check: 읽기 확인, init: 없을 때만 컬렉션 생성",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "check":
            return check_connection()
        return initialize_collection()
    except VectorStoreError as error:
        print(
            f"Qdrant 오류 [{error.code}]: {error.safe_message}",
            file=sys.stderr,
        )
        return 1
    except (ValueError, RuntimeError) as error:
        print(f"설정 오류: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
