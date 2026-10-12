''' 경로, 변하지 않는 상수들 여기에 정의하고 다른 파일에서 import해서 사용'''

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
# print(PROJECT_PATH) # 확인용
DATA_PATH = PROJECT_PATH / "data"

# Chunk된 데이터 경로
CHUNK_DATA_PATH = DATA_PATH / "chunk"
CHUNK_CONSUMER_SOURCES_DATA_PATH = CHUNK_DATA_PATH / "consumer_sources"
CHUNK_GUIDES_DATA_PATH = CHUNK_DATA_PATH / "guides"
CHUNK_LAWS_DATA_PATH = CHUNK_DATA_PATH / "laws"
CHUNK_PRECEDENTS_DATA_PATH = CHUNK_DATA_PATH / "precedents"

# 정제된 데이터 경로
PROCESSED_DATA_PATH = DATA_PATH / "processed"
PROCESSED_CONSUMER_SOURCES_DATA_PATH = PROCESSED_DATA_PATH / "consumer_sources"
PROCESSED_GUIDES_DATA_PATH = PROCESSED_DATA_PATH / "guides"
PROCESSED_LAWS_DATA_PATH = PROCESSED_DATA_PATH / "laws"
PROCESSED_PRECEDENTS_DATA_PATH = PROCESSED_DATA_PATH / "precedents"
PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH = PROCESSED_DATA_PATH / "retrieval_evaluation"

# 원본 데이터 경로
RAW_DATA_PATH = DATA_PATH / "raw"
RAW_CONSUMER_SOURCES_DATA_PATH = RAW_DATA_PATH / "consumer_sources"
RAW_GUIDES_DATA_PATH = RAW_DATA_PATH / "guides"
RAW_LAW_APPENDICES_DATA_PATH = RAW_DATA_PATH / "law_appendices"
RAW_LAW_HISTORY_DATA_PATH = RAW_DATA_PATH / "law_history"
RAW_LAWS_DATA_PATH = RAW_DATA_PATH / "laws"
RAW_PRECEDENTS_DATA_PATH = RAW_DATA_PATH / "precedents" 


# 모델 이름 여기에 정의

EMBED_MODEL = "text-embedding-3-small"

# 모델 이름 여기에 정의
# LLM_MODEL = "gpt-4o-mini"
LLM_MODEL = "gemini-3.8-flash"
# EMBED_MODEL = "text-embedding-3-small"
EMBED_MODEL = "gemini-embedding-001"