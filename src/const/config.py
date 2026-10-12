''' 경로, 변하지 않는 상수들 여기에 정의하고 다른 파일에서 import해서 사용'''

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


ENV_PATH = PROJECT_PATH / ".env"

# 재정렬 모델 다운로드 파일을 보관하는 공통 경로입니다.
RERANK_CACHE_PATH = PROJECT_PATH / ".cache" / "rerank"
