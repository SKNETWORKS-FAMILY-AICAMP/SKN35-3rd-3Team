''' 경로, 변하지 않는 상수들 여기에 정의하고 다른 파일에서 import해서 사용'''

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
# print(PROJECT_PATH) # 확인용

DATA_PATH = PROJECT_PATH / "data"
# print(DATA_PATH) # 확인용

# Chunk된 데이터 경로
CHUNK_DATA_PATH = DATA_PATH / "chunk" / "파일 이름 넣어주세요"
# print(CHUNK_DATA_PATH) # 확인용

# 정제된 데이터 경로
CLEAN_DATA_PATH = DATA_PATH / "processed" / "파일 이름 넣어주세요"
# print(CLEAN_DATA_PATH) # 확인용

# 원본 데이터 경로
RAW_DATA_PATH = DATA_PATH / "raw" / "파일 이름 넣어주세요"
# print(RAW_DATA_PATH) # 확인용


# 모델 이름 여기에 정의
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"