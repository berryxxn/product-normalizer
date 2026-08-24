import re

_WHITESPACE_RE = re.compile(r"\s+")
_UNIT_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(gr|gram|g|kg|ml|l|liter|pcs|buah|btl|botol|sachet|box|dus|pack|packs)\b", re.IGNORECASE)
_PUNCT_RE = re.compile(r"[^\w\s]")


def clean_text(name: str) -> str:
    text = name.strip().lower()
    text = _UNIT_RE.sub(r"\1\2", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_batch(names: list[str]) -> list[str]:
    return [clean_text(name) for name in names]