"""프로젝트 루트의 ``.env``를 안전하게 읽는 공통 함수."""

from __future__ import annotations

from pathlib import Path

from src.const.config import ENV_PATH, PROJECT_PATH, load_project_environment


PROJECT_ROOT = PROJECT_PATH
DEFAULT_ENV_PATH = ENV_PATH


def load_project_env(env_path: str | Path | None = None) -> Path:
    """환경변수를 덮어쓰지 않고 프로젝트 ``.env``를 로드합니다.

    반환값은 실제로 확인한 경로입니다. 키의 값은 출력하지 않습니다.
    """

    return load_project_environment(env_path)
