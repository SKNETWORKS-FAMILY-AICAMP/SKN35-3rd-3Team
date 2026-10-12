"""공통 .env 로딩과 공급자별 모델 생성 함수입니다."""
import os
from functools import lru_cache
from dotenv import load_dotenv
from src.const.config import ENV_PATH, RERANK_CACHE_PATH


def load_project_env(env_path=None):
    # 실행 위치와 관계없이 config에 정의한 프로젝트 .env를 읽습니다.
    path = ENV_PATH if env_path is None else env_path
    load_dotenv(dotenv_path=path, override=False)
    return path


def create_llm(temperature=0, model=None):
    # 모델명과 API 키를 .env에서 읽어 OpenAI 모델을 생성합니다.
    from langchain_openai import ChatOpenAI
    load_project_env()
    return ChatOpenAI(
        model=model or os.environ["LLM_MODEL"],
        api_key=os.environ["OPENAI_API_KEY"],
        temperature=temperature, timeout=30, max_retries=0,
    )


def create_embedding():
    # 검색 문서와 같은 임베딩 모델을 사용해야 합니다.
    from langchain_openai import OpenAIEmbeddings
    load_project_env()
    return OpenAIEmbeddings(
        model=os.environ["EMBEDDING_MODEL"], api_key=os.environ["OPENAI_API_KEY"],
    )


def create_llm_gmn(temperature=0):
    from langchain_google_genai import ChatGoogleGenerativeAI
    load_project_env()
    return ChatGoogleGenerativeAI(
        model=os.environ["GOOGLE_LLM_MODEL"],
        google_api_key=os.environ["GOOGLE_API_KEY"], temperature=temperature,
    )


def create_embedding_gmn():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    load_project_env()
    return GoogleGenerativeAIEmbeddings(
        model=os.environ["GOOGLE_EMBEDDING_MODEL"],
        google_api_key=os.environ["GOOGLE_API_KEY"],
    )


# 환경변수의 재정렬 모델을 한 번 로딩하고 이후 호출에서 재사용합니다.
@lru_cache(maxsize=1)
def create_rerank_model():
    # 선택 의존성이라 모델을 사용하는 시점에만 로딩합니다.
    from sentence_transformers import CrossEncoder
    import torch
    load_project_env()
    torch.set_num_threads(4)
    return CrossEncoder(
        os.environ["RERANK_MODEL"], device="cpu", max_length=512,
        cache_folder=str(RERANK_CACHE_PATH), trust_remote_code=False,
    )
