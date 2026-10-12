''' 경로, 변하지 않는 상수들 여기에 정의하고 다른 파일에서 import해서 사용'''

from __future__ import annotations

import os
from pathlib import Path


PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
# print(PROJECT_PATH) # 확인용

DATA_PATH = PROJECT_PATH / "data"

# Chunk된 데이터 경로
CHUNK_DATA_PATH = DATA_PATH / "chunk"

# 정제된 데이터 경로
CLEAN_DATA_PATH = DATA_PATH / "processed"

# 원본 데이터 경로
RAW_DATA_PATH = DATA_PATH / "raw"
CONSUMER_SOURCES_DATA_PATH = RAW_DATA_PATH / "consumer_sources"

# 테스트 결과 경로
TEST_PATH = PROJECT_PATH / "tests"
TEST_REPORT_PATH = TEST_PATH / "report"

# 환경 파일 경로
ENV_PATH = PROJECT_PATH / ".env"
ENV_EXAMPLE_PATH = PROJECT_PATH / ".env.example"


def load_project_environment(env_path: str | Path | None = None) -> Path:
    """프로젝트 ``.env``를 읽고 확인한 경로를 반환합니다.

    운영체제나 실행기가 이미 주입한 환경변수는 덮어쓰지 않습니다.
    """

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise RuntimeError(
            ".env 로딩에는 python-dotenv 설치가 필요합니다. "
            "프로젝트 루트에서 `uv sync`를 실행하세요."
        ) from error

    resolved_path = Path(env_path) if env_path is not None else ENV_PATH
    load_dotenv(dotenv_path=resolved_path, override=False)
    return resolved_path


# 모델 이름 여기에 정의
# 신규 코드는 .env에 정의된 모델명을 사용합니다.
# 아래 상수는 기존 호출부 호환을 위해 유지합니다.
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.8-flash")
EMBED_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
)
