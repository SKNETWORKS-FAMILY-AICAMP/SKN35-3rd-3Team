''' 경로, 변하지 않는 상수들 여기에 정의하고 다른 파일에서 import해서 사용'''

from pathlib import Path

PROJECT_PATH = Path(__file__).resolve().parent.parent.parent

DATA_PATH = PROJECT_PATH / "data"

# Chunk된 데이터 경로
CHUNK_DATA_PATH = DATA_PATH / "chunk"

# 정제된 데이터 경로
CLEAN_DATA_PATH = DATA_PATH / "processed"

# 원본 데이터 경로
RAW_DATA_PATH = DATA_PATH / "raw"


# 모델 이름 여기에 정의

EMBED_MODEL = "text-embedding-3-small"
