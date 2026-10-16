"""상담 기록의 질문·분야·날짜 분류. 모델 호출이나 저장소 접근 없이 동작한다."""

from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
OLDEST_TIMESTAMP = datetime.min.replace(tzinfo=KST)
CATEGORY_KEYWORDS = {
    "노동/근로": ("임금", "퇴직", "해고", "근로", "급여", "직장", "야근"),
    "부동산": ("임대", "전세", "월세", "보증금", "집주인", "부동산"),
    "중고거래": ("중고", "당근", "번개장터"),
    "소비자 피해": ("환불", "환급", "택배", "배송", "쇼핑", "헬스", "필라테스"),
}


def now_iso():
    return datetime.now(KST).isoformat()


def first_question(item):
    questions = (
        str(message.get("content", ""))
        for message in item.get("history", [])
        if message.get("role") == "user"
    )
    return next(questions, str(item.get("title", "상담")))


def category(item):
    question = first_question(item)
    # 여러 분야의 키워드가 겹치면 사전에 정의한 순서대로 첫 분야를 사용한다.
    for label, words in CATEGORY_KEYWORDS.items():
        if any(word in question for word in words):
            return label
    return "기타"


def timestamp(item):
    """기존 기록에 시간이 없거나 형식이 잘못됐으면 날짜를 추측하지 않는다."""
    try:
        value = datetime.fromisoformat(item.get("updated_at") or item.get("created_at") or "")
        return value.astimezone(KST) if value.tzinfo else value.replace(tzinfo=KST)
    except (ValueError, TypeError):
        return None


def _date_group(stamp, today):
    if stamp is None:
        return "이전 기록"
    age = (today - stamp.date()).days
    if age == 0:
        return "오늘"
    if 1 <= age <= 7:
        return "지난 7일"
    return "이전 기록"


def group_consultations(consultations, mode="날짜별", now=None):
    now = now or datetime.now(KST)
    by_category = mode == "분야별"
    labels = [*CATEGORY_KEYWORDS, "기타"] if by_category else ["오늘", "지난 7일", "이전 기록"]
    groups = {label: [] for label in labels}
    # 같은 시각의 기록은 최근 추가된 항목부터, 날짜가 없는 기록은 마지막에 둔다.
    ordered = sorted(
        reversed(list(consultations.items())),
        key=lambda pair: timestamp(pair[1]) or OLDEST_TIMESTAMP,
        reverse=True,
    )
    for thread_id, item in ordered:
        label = category(item) if by_category else _date_group(timestamp(item), now.date())
        groups[label].append((thread_id, item))
    return {label: items for label, items in groups.items() if items}
