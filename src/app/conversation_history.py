"""Session history presentation without model calls or invented timestamps."""
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def now_iso():
    return datetime.now(KST).isoformat()


def first_question(item):
    return next((str(message.get("content", "")) for message in item.get("history", [])
                 if message.get("role") == "user"), str(item.get("title", "상담")))


def category(item):
    question = first_question(item)
    for label, words in [
        ("노동/근로", ("임금", "퇴직", "해고", "근로", "급여", "직장", "야근")),
        ("부동산", ("임대", "전세", "월세", "보증금", "집주인", "부동산")),
        ("중고거래", ("중고", "당근", "번개장터")),
        ("소비자 피해", ("환불", "환급", "택배", "배송", "쇼핑", "헬스", "필라테스")),
    ]:
        if any(word in question for word in words):
            return label
    return "기타"


def timestamp(item):
    try:
        value = datetime.fromisoformat(item.get("updated_at") or item.get("created_at") or "")
        return value.astimezone(KST) if value.tzinfo else value.replace(tzinfo=KST)
    except (ValueError, TypeError):
        return None


def group_consultations(consultations, mode="날짜별", now=None):
    now = now or datetime.now(KST)
    labels = ["노동/근로", "부동산", "중고거래", "소비자 피해", "기타"] if mode == "분야별" else ["오늘", "지난 7일", "이전 기록"]
    groups = {label: [] for label in labels}
    ordered = sorted(reversed(list(consultations.items())),
                     key=lambda pair: timestamp(pair[1]) or datetime.min.replace(tzinfo=KST), reverse=True)
    for thread_id, item in ordered:
        stamp = timestamp(item)
        age = (now.date() - stamp.date()).days if stamp else None
        label = category(item) if mode == "분야별" else (
            "오늘" if age == 0 else "지난 7일" if age is not None and 1 <= age <= 7 else "이전 기록")
        groups[label].append((thread_id, item))
    return {label: items for label, items in groups.items() if items}
