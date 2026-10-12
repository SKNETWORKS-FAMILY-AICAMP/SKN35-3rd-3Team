import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.const.config import load_project_environment


def _required_environment(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name}가 설정되지 않았습니다.")
    return value


def _model_name(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


# 임베딩 모델 정의
def create_embedding():
    load_project_environment()
    return OpenAIEmbeddings(
        model=_model_name(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        api_key=_required_environment("OPENAI_API_KEY"),
    )

# 임베딩 모델 다른 곳에서 사용하는 법
# from src.const.model import create_embedding
# emb = create_embedding()


# Vector DB용 임베딩 모델 정의
def create_vector_embedding(settings=None):
    """Vector DB용 제공자별 임베딩 구현체를 생성합니다.

    OpenAI·NVIDIA·로컬 GPU 선택과 차원 검증은 기존 Vector DB 설정 및
    임베딩 구현에 위임합니다.
    """

    from src.const.qdrant_config import load_qdrant_settings
    from src.embedder import create_embedding_provider

    if settings is None:
        load_project_environment()
        settings = load_qdrant_settings().embedding
    embedding_settings = settings
    return create_embedding_provider(embedding_settings)


# 필요한 다른 모델도 정의하고 위에 예시처럼 사용

# 챗 모델 정의
def create_llm(temperature: float = 0):
    load_project_environment()
    return ChatOpenAI(
        model=_model_name("OPENAI_LLM_MODEL", "gpt-4o-mini"),
        temperature=temperature,
        api_key=_required_environment("OPENAI_API_KEY"),
    )

# ============================================================================================================

# 테스트 용 GEMINI VER

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


def create_embedding_gmn():
    load_project_environment()
    return GoogleGenerativeAIEmbeddings(
        model=_model_name(
            "GOOGLE_EMBEDDING_MODEL",
            "models/gemini-embedding-001",
        ),
        google_api_key=_required_environment("GOOGLE_API_KEY"),
    )


def create_llm_gmn(temperature: float = 0):
    load_project_environment()
    return ChatGoogleGenerativeAI(
        model=_model_name("LLM_MODEL", "gemini-3.8-flash"),
        temperature=temperature,
        google_api_key=_required_environment("GOOGLE_API_KEY"),
    )
