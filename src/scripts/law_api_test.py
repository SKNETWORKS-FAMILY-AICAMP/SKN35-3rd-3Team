
import os
import requests
from dotenv import load_dotenv


# .env 파일에 저장된 환경변수 불러오기
load_dotenv()

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
