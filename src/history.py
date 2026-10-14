"""messages에서 최근 대화를 프롬프트용 문자열로 만듭니다."""

IMAGE_MARKER = "\n\n[첨부 이미지 설명]"
ROLE_NAMES = {"human": "사용자", "ai": "상담사"}


def format_history(messages, max_turns: int = 3, max_chars: int = 400) -> str:
    recent = (messages or [])[-max_turns * 2:]
    lines = []
    for m in recent:
        role = ROLE_NAMES.get(m.type)
        if role is None:  # system, tool 메시지는 제외
            continue
        text = m.content if isinstance(m.content, str) else str(m.content)
        # image_node가 붙인 이미지 설명은 이력에서 제외 (길고 검색에 불필요)
        text = text.split(IMAGE_MARKER)[0].strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)