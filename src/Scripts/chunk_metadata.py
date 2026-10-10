"""자료별 원래 필드를 유지하면서 검색 청크의 공통 메타데이터를 추가한다.

domains는 문서/기존 분류 수준의 후보 태그이며 개별 조문의 관련성 판정이 아니다.
임베딩 본문·헤더·경고·원래 날짜 및 domain 값은 변경하지 않는다.
"""
from datetime import date
from pathlib import PurePosixPath

METADATA_VERSION = 1
DOCUMENT_TYPES = frozenset({"law", "precedent", "official_guide",
                            "administrative_rule", "mediation_case", "standard_terms"})
DOMAIN_MAP = {
    "주택임대차": ["housing_lease"], "임대차": ["housing_lease"],
    "해고": ["labor_dismissal"], "임금": ["labor_wage"],
    "임금체불": ["labor_wage"], "해고·임금": ["labor_dismissal", "labor_wage"],
    "중고거래": ["used_trade"], "온라인 소비자 서비스": ["online_consumer_service"],
    "consumer": ["consumer_general"], "fitness_refund": ["fitness_refund"],
    "parcel_loss": ["parcel_loss"], "shopping_return": ["shopping_return"],
}
LAW_DOMAINS = {
    "주택임대차보호법": ["housing_lease"], "공인중개사법": ["housing_lease"],
    "민법": ["civil_general"], "형법": ["criminal_general"],
    "근로기준법": ["labor_general"], "근로기준법 시행령": ["labor_general"],
    "근로자퇴직급여 보장법": ["labor_wage"],
    "전자상거래 등에서의 소비자보호에 관한 법률": ["shopping_return", "consumer_general"],
}
GUIDE_DOMAINS = {
    "[pdf]주택임대차.pdf": ["housing_lease"],
    "[pdf]중고거래 피해자.pdf": ["used_trade"],
    "[pdf]해고근로자.pdf": ["labor_dismissal"],
    "[pdf]인터넷 쇼핑.pdf": ["shopping_return"],
    "[pdf]택배.pdf": ["parcel_loss"],
}
ALLOWED_DOMAINS = frozenset(d for mapping in (DOMAIN_MAP, LAW_DOMAINS, GUIDE_DOMAINS)
                          for domains in mapping.values() for d in domains)
DATE_FIELDS = ("effective_date", "article_effective_date", "unit_effective_date",
               "promulgation_date", "judgment_date", "decision_date", "updated_date",
               "terms_revision_date", "information_as_of", "as_of")
COMMON_FIELDS = ("metadata_version", "document_type", "document_id", "title", "domains",
                 "domain_classification", "source_url", "source_path", "dates")


def normalize_date(value):
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise ValueError("날짜는 문자열 또는 null이어야 합니다.")
    if len(value) == 8 and value.isdigit():
        value = f"{value[:4]}-{value[4:6]}-{value[6:]}"
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"확인할 수 없는 날짜 형식: {value}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"확인할 수 없는 날짜 형식: {value}")
    return value


def normalize_chunk(chunk, document=None):
    """원문 문서는 판례처럼 청크에 없는 원본 경로를 보충할 때만 사용한다."""
    document = document or {}
    result = dict(chunk)
    kind = chunk.get("document_type")
    if not kind and chunk.get("precedent_id"):
        kind = "precedent"
    if kind not in DOCUMENT_TYPES:
        raise ValueError(f"지원하지 않는 자료 유형: {kind}")
    id_key = {"law": "version_id", "precedent": "precedent_id",
              "official_guide": "guide_id"}.get(kind, "source_id")
    identifier = chunk.get(id_key)
    if not identifier:
        raise ValueError(f"문서 식별자 누락: {id_key}")
    title = chunk.get("title") or chunk.get("law_name") or chunk.get("case_name")
    if not title:
        raise ValueError("문서 제목 누락")
    if kind == "law":
        domains = LAW_DOMAINS.get(chunk.get("law_name"), [])
        basis = "law_document"
    elif kind == "official_guide":
        domains = GUIDE_DOMAINS.get(chunk.get("source_file"), [])
        basis = "guide_document"
    else:
        domains = DOMAIN_MAP.get(chunk.get("domain"), [])
        basis = "existing_domain"
    source_path = chunk.get("source_path") or chunk.get("raw_file")
    if not source_path:
        filename = document.get("source_file") or chunk.get("source_file")
        if filename:
            filename = str(filename).replace("\\", "/")
            if kind == "law" and PurePosixPath(filename).name == filename:
                filename = "data/raw/laws/" + filename
            source_path = filename
    if source_path:
        source_path = str(source_path).replace("\\", "/")
        path = PurePosixPath(source_path)
        if path.is_absolute() or ":" in source_path or ".." in path.parts:
            raise ValueError("출처 경로는 프로젝트 상대 경로여야 합니다.")
    result.update(metadata_version=METADATA_VERSION, document_type=kind,
                  document_id=f"{kind}:{identifier}", title=title,
                  domains=list(domains), domain_classification=basis if domains else "unmapped",
                  source_url=chunk.get("source_url"), source_path=source_path,
                  dates={key: normalize_date(chunk.get(key)) for key in DATE_FIELDS})
    validate_chunk(result)
    return result


def validate_chunk(chunk):
    if any(key not in chunk for key in COMMON_FIELDS):
        raise ValueError("공통 메타데이터 필드 누락")
    if chunk["metadata_version"] != METADATA_VERSION or chunk["document_type"] not in DOCUMENT_TYPES:
        raise ValueError("메타데이터 버전 또는 자료 유형 오류")
    for key in ("chunk_id", "document_id", "title", "text", "embedding_text"):
        if not isinstance(chunk.get(key), str) or not chunk[key].strip():
            raise ValueError(f"비어 있거나 유효하지 않은 필드: {key}")
    domains = chunk["domains"]
    if (not isinstance(domains, list) or any(not isinstance(d, str) or d not in ALLOWED_DOMAINS for d in domains)
            or len(set(domains)) != len(domains)):
        raise ValueError("분야 분류값 오류")
    if set(chunk["dates"]) != set(DATE_FIELDS):
        raise ValueError("시점 필드 누락")
    for value in chunk["dates"].values():
        if normalize_date(value) != value:
            raise ValueError("공통 날짜는 ISO 형식이어야 합니다.")
