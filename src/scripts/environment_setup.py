"""팀원의 로컬 ``.env`` 준비 상태를 안전하게 점검합니다.

이 명령은 ``.env``가 없을 때만 ``.env.example``을 복사합니다. 기존 파일과
비밀값은 수정하거나 출력하지 않습니다.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

from src.const.config import DEFAULT_ENV_PATH, PROJECT_PATH

ENV_EXAMPLE_PATH = PROJECT_PATH / ".env.example"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def prepare_env_file(
    env_path: Path = DEFAULT_ENV_PATH,
    example_path: Path = ENV_EXAMPLE_PATH,
) -> bool:
    """누락된 ``.env``를 예제에서 만들고 생성 여부를 반환합니다."""

    if env_path.exists():
        return False
    if not example_path.is_file():
        raise FileNotFoundError(f"환경 예제 파일이 없습니다: {example_path}")

    env_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(example_path, env_path)
    return True


def _configured(values: dict[str, str | None], name: str) -> bool:
    value = values.get(name)
    return isinstance(value, str) and bool(value.strip())


def missing_required_names(
    env_path: Path = DEFAULT_ENV_PATH,
) -> tuple[str, ...]:
    """현재 제공자에서 입력이 필요한 설정 이름만 반환합니다."""

    values = dict(dotenv_values(env_path))
    missing: list[str] = []
    qdrant_url = str(values.get("QDRANT_URL") or "").strip()
    qdrant_host = urlparse(qdrant_url).hostname if qdrant_url else None
    if qdrant_host not in LOOPBACK_HOSTS and not _configured(
        values,
        "QDRANT_API_KEY",
    ):
        missing.append("QDRANT_API_KEY")

    provider = str(values.get("EMBEDDING_PROVIDER") or "").strip().lower()
    common_key = _configured(values, "EMBEDDING_API_KEY")
    if provider in {"local_huggingface", "sentence_transformers"}:
        return tuple(missing)
    if provider == "nvidia":
        if not common_key and not _configured(values, "NVIDIA_API_KEY"):
            missing.append("NVIDIA_API_KEY")
        return tuple(missing)
    if provider == "openai":
        if not common_key and not _configured(values, "OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY")
        return tuple(missing)
    if provider == "openai_compatible":
        if not common_key:
            missing.append("EMBEDDING_API_KEY")
        return tuple(missing)

    openai_key = _configured(values, "OPENAI_API_KEY")
    nvidia_key = _configured(values, "NVIDIA_API_KEY")
    if openai_key and nvidia_key:
        missing.append("EMBEDDING_PROVIDER")
    elif not common_key and not openai_key and not nvidia_key:
        missing.append("OPENAI_API_KEY 또는 NVIDIA_API_KEY")
    return tuple(missing)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=".env를 만들고 비밀값 설정 여부만 확인합니다.",
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    parser.add_argument(
        "--example-file",
        type=Path,
        default=ENV_EXAMPLE_PATH,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="필수 설정이 비어 있으면 종료 코드 2를 반환합니다.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    env_path = args.env_file.resolve()
    created = prepare_env_file(env_path, args.example_file.resolve())

    print(f".env 경로: {env_path}")
    print("생성 여부:", created)
    if not created:
        print("기존 .env를 유지했습니다.")

    missing = missing_required_names(env_path)
    if missing:
        print("입력이 필요한 항목:", ", ".join(missing))
        return 2 if args.check else 0

    print("필수 비밀값 설정 확인: 완료")
    return 0
