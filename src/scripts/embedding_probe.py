"""세 질문 임베딩 실행 프로필의 실제 준비 상태를 점검합니다."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from dataclasses import asdict, dataclass

from src.const.qdrant_config import EmbeddingSettings, get_api_embedding_profile
from src.embedder import create_embedding_provider

DEFAULT_QUERY = "주택 임대차보호법에서 계약 갱신은 어떻게 하나요?"


@dataclass(frozen=True)
class ProbeResult:
    provider: str
    status: str
    model: str | None
    expected_dimension: int | None
    actual_dimension: int | None = None
    vector_norm: float | None = None
    reason: str | None = None


def _configured(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def _api_settings(provider: str) -> EmbeddingSettings:
    profile = get_api_embedding_profile(provider)
    return EmbeddingSettings(
        provider=profile.provider,
        model=profile.model,
        dimension=profile.dimension,
        normalized=profile.normalized,
    )


def _local_settings() -> EmbeddingSettings | None:
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    raw_dimension = os.getenv("EMBEDDING_DIMENSION", "").strip()
    if not model or not raw_dimension:
        return None
    try:
        dimension = int(raw_dimension)
    except ValueError:
        return None
    return EmbeddingSettings(
        provider="local_huggingface",
        model=model,
        dimension=dimension,
        normalized=True,
    )


def _readiness_failure(provider: str) -> ProbeResult | None:
    if provider == "openai" and not _configured("OPENAI_API_KEY"):
        profile = get_api_embedding_profile(provider)
        return ProbeResult(
            provider=provider,
            status="skipped",
            model=profile.model,
            expected_dimension=profile.dimension,
            reason="OPENAI_API_KEY가 설정되지 않았습니다.",
        )
    if provider == "nvidia" and not _configured("NVIDIA_API_KEY"):
        profile = get_api_embedding_profile(provider)
        return ProbeResult(
            provider=provider,
            status="skipped",
            model=profile.model,
            expected_dimension=profile.dimension,
            reason="NVIDIA_API_KEY가 설정되지 않았습니다.",
        )
    if provider == "local_huggingface":
        settings = _local_settings()
        if settings is None:
            return ProbeResult(
                provider=provider,
                status="skipped",
                model=None,
                expected_dimension=None,
                reason=(
                    "EMBEDDING_MODEL과 EMBEDDING_DIMENSION이 필요합니다."
                ),
            )
        missing = [
            name
            for name in ("torch", "sentence_transformers")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            return ProbeResult(
                provider=provider,
                status="skipped",
                model=settings.model,
                expected_dimension=settings.dimension,
                reason="필수 패키지가 없습니다: " + ", ".join(missing),
            )
    return None


def _settings(provider: str) -> EmbeddingSettings:
    if provider in {"openai", "nvidia"}:
        return _api_settings(provider)
    local_settings = _local_settings()
    if local_settings is None:
        raise RuntimeError("로컬 임베딩 설정을 읽을 수 없습니다.")
    return local_settings


def _probe(provider: str, query: str) -> ProbeResult:
    readiness_failure = _readiness_failure(provider)
    if readiness_failure is not None:
        return readiness_failure

    settings = _settings(provider)
    try:
        vector = list(
            map(
                float,
                create_embedding_provider(settings).embed_query(query),
            )
        )
    except Exception as error:  # noqa: BLE001 - 키나 응답 전문을 노출하지 않음
        return ProbeResult(
            provider=provider,
            status="error",
            model=settings.model,
            expected_dimension=settings.dimension,
            reason=f"실제 호출 실패: {type(error).__name__}",
        )

    vector_norm = math.sqrt(sum(value * value for value in vector))
    return ProbeResult(
        provider=provider,
        status="success",
        model=settings.model,
        expected_dimension=settings.dimension,
        actual_dimension=len(vector),
        vector_norm=round(vector_norm, 6),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI·NVIDIA·개인 GPU 질문 임베딩을 점검합니다.",
    )
    parser.add_argument(
        "--provider",
        choices=("all", "openai", "nvidia", "local_huggingface"),
        default="all",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    providers = (
        ("openai", "nvidia", "local_huggingface")
        if args.provider == "all"
        else (args.provider,)
    )
    results = [_probe(provider, args.query) for provider in providers]
    print(
        json.dumps(
            [asdict(result) for result in results],
            ensure_ascii=False,
            indent=2,
        )
    )
