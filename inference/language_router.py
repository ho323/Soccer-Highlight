import re


HANGUL_RE = re.compile(r"[가-힣]")


def detect_language(text: str, default: str = "en") -> str:
    if not text:
        return default
    if HANGUL_RE.search(text):
        return "ko"
    return "en"


def resolve_language(language_option: str, signal_text: str = "") -> str:
    if language_option in ("ko", "en"):
        return language_option
    return detect_language(signal_text, default="en")
