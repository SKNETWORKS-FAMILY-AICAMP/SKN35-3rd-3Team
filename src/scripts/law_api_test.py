"""국가법령정보센터 API 연결을 확인하는 재사용 가능한 요청 함수와 CLI."""
import argparse
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import PROJECT_PATH

API_URL = "https://www.law.go.kr/DRF/lawSearch.do"


def request_precedent_search(query="보증금반환", *, oc=None, display=20, page=1,
                             env_path=None, session=None):
    """판례 목록 응답을 반환한다. 파일 저장이나 출력은 하지 않는다.

    oc가 없으면 프로젝트 .env와 환경변수에서 LAW_API_OC를 읽는다.
    requests 통신/HTTP 예외는 호출한 쪽으로 전달한다.
    session에는 requests와 같은 get() 인터페이스를 가진 클라이언트를 전달한다.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query는 비어 있지 않은 문자열이어야 합니다.")
    for name, value in (("display", display), ("page", page)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name}는 양의 정수여야 합니다.")
    if oc is None:
        load_dotenv(Path(env_path) if env_path is not None else PROJECT_PATH / ".env")
        oc = os.getenv("LAW_API_OC", "")
    if not isinstance(oc, str) or not oc.strip():
        raise ValueError("LAW_API_OC 인증값이 필요합니다.")
    params = {"OC": oc.strip(), "target": "prec", "type": "XML", "search": 2,
              "query": query.strip(), "display": display, "page": page}
    client = session if session is not None else requests
    response = client.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    return response


def main(argv=None):
    """명령행 인자를 처리하고 인증값을 제외한 상태와 응답 일부를 출력한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="보증금반환")
    parser.add_argument("--display", type=int, default=20)
    parser.add_argument("--page", type=int, default=1)
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        response = request_precedent_search(args.query, display=args.display,
                                           page=args.page, env_path=args.env_file)
    except ValueError as error:
        print(f"요청 설정 오류: {error}", file=sys.stderr)
        return 1
    except requests.RequestException:
        # 요청 예외 문자열에는 인증값이 담긴 URL이 들어갈 수 있다.
        print("API 요청에 실패했습니다. 네트워크와 인증 설정을 확인하세요.", file=sys.stderr)
        return 1
    print("요청 주소:", API_URL)
    print("HTTP 상태:", response.status_code)
    print("응답 일부:")
    print(response.text[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
