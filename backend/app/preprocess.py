import re

_WHITESPACE_RE = re.compile(r"\s+")
_UNIT_RE = re.compile(r"(\d+)\s*(gr|gram|g|kg|ml|l|liter)\b")
_QUANTITY_RE = re.compile(r"(?:(\d+)\s*x\s*)?(\d+(?:\.\d+)?)\s*(gr|gram|g|kg|ml|l|liter)\b")
_UNIT_TO_BASE = {"gr": 1, "gram": 1, "g": 1, "kg": 1000, "ml": 1, "l": 1000, "liter": 1000}


def clean_text(name: str) -> str:
    text = name.strip().lower()
    text = _UNIT_RE.sub(r"\1\2", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def clean_batch(names: list[str]) -> list[str]:
    return [clean_text(name) for name in names]


def extract_quantity(cleaned_text: str) -> float | None:
    """Total pack quantity in a base unit (grams or ml), accounting for an
    'NxQTYunit' multiplier (e.g. '5x85gr' -> 425). Returns None if no
    quantity/unit is present in the text."""
    match = _QUANTITY_RE.search(cleaned_text)
    if not match:
        return None
    multiplier = float(match.group(1)) if match.group(1) else 1.0
    amount = float(match.group(2))
    unit = match.group(3)
    return multiplier * amount * _UNIT_TO_BASE[unit]
