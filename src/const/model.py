import os
import openai
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings , ChatOpenAI
from src.const import config

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

# 임베딩 모델 정의
def create_embedding():
    return OpenAIEmbeddings(
        model=config.EMBED_MODEL,
        api_key=API_KEY,
    )

# 임베딩 모델 다른 곳에서 사용하는 법
# from src.const.model import create_embedding
# emb = create_embedding()

# 필요한 다른 모델도 정의하고 위에 예시처럼 사용

# 챗 모델 정의
def create_llm(temperature : float = 0):
    return ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=temperature,
        api_key=API_KEY
    )

# ============================================================================================================

# 테스트 용 GENIMI VER

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

def create_embedding_gmn():
    return GoogleGenerativeAIEmbeddings(
        model=config.EMBED_MODEL,
        google_api_key=API_KEY,
    )

def create_llm_gmn(temperature: float = 0):
    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        temperature=temperature,
        google_api_key=API_KEY,
    )