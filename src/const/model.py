import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from src.const import config

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

OPENAI_LLM_MODEL =os.getenv("OPENAI_LLM_MODEL")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL")

GEMINI_LLM_MODEL = os.getenv("GEMINI_LLM_MODEL")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL")

LAW_API = os.getenv("LAW_API_OC")

# 임베딩 모델 정의
def create_embedding():
    return OpenAIEmbeddings(
        model=OPENAI_EMBED_MODEL,
        dimensions=config.EMBED_DIMENSION,
        api_key=OPENAI_API_KEY,
    )

# 임베딩 모델 다른 곳에서 사용하는 법
# from src.const.model import create_embedding
# emb = create_embedding()

# 필요한 다른 모델도 정의하고 위에 예시처럼 사용

# 챗 모델 정의
def create_llm(temperature : float = 0):
    return ChatOpenAI(
        model=OPENAI_LLM_MODEL,
        temperature=temperature,
        api_key=OPENAI_API_KEY
    )

def create_rerank_model():
    # 선택 의존성이라 모델을 사용하는 시점에만 로딩합니다.
    from sentence_transformers import CrossEncoder
    import torch
    torch.set_num_threads(4)
    return CrossEncoder(
        os.environ["RERANK_MODEL"], device="cpu", max_length=512,
        cache_folder=str(RERANK_CACHE_PATH), trust_remote_code=False,
    )

# ============================================================================================================

# 테스트 용 GENIMI VER

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def create_embedding_gmn():
    return GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

def create_llm_gmn(temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model=GEMINI_LLM_MODEL,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY,
    )