"""프로젝트 루트의 ``.env``를 안전하게 읽는 공통 함수."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env(env_path: str | Path | None = None) -> Path:
    """환경변수를 덮어쓰지 않고 프로젝트 ``.env``를 로드합니다.

    반환값은 실제로 확인한 경로입니다. 키의 값은 출력하지 않습니다.
    """

    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError as error:
        raise RuntimeError(
            ".env 로딩에는 python-dotenv 설치가 필요합니다. "
            "프로젝트 루트에서 `uv sync`를 실행하세요."
        ) from error

    resolved_path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    load_dotenv(dotenv_path=resolved_path, override=False)
    return resolved_path
