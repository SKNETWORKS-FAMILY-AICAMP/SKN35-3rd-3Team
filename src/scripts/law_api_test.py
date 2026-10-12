
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import PROJECT_PATH


# .env 파일에 저장된 환경변수 불러오기
load_dotenv(PROJECT_PATH / ".env")

# .env 파일에 저장된 LAW_API_OC 키 
OC = os.environ["LAW_API_OC"]

url = "https://www.law.go.kr/DRF/lawSearch.do"

params = {
    "OC": OC,
    "target": "prec",
    "type": "XML",
    "search": 2,
    "query": "보증금반환",
    "display": 20,
    "page": 1
}

response = requests.get(url, params=params, timeout=30)
response.raise_for_status()

print("요청 URL:", response.url.replace(OC, "***"))
print("HTTP 상태:", response.status_code)
print("응답 일부:")
print(response.text[:1500])
