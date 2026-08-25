import logging
import re

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_UNIT_RE = re.compile(r"(\d+)\s*(gr|gram|g|kg|ml|l|liter)\b")
_QUANTITY_RE = re.compile(r"(?:(\d+)\s*x\s*)?(\d+(?:\.\d+)?)\s*(gr|gram|g|kg|ml|l|liter)\b")
_UNIT_TO_BASE = {"gr": 1, "gram": 1, "g": 1, "kg": 1000, "ml": 1, "l": 1000, "liter": 1000}


def clean_text(name: str) -> str:
    text = name.strip().lower()
    text = _UNIT_RE.sub(r"\1\2", text)
    text = _WHITESPACE_RE.sub(" ", text)
    result = text.strip()
    logger.debug("preprocess.clean_text: %r -> %r", name, result)
    return result


def clean_batch(names: list[str]) -> list[str]:
    cleaned = [clean_text(name) for name in names]
    logger.info("preprocess.clean_batch: cleaned %d name(s) -> %s", len(names), cleaned)
    return cleaned


def extract_quantity(cleaned_text: str) -> float | None:
    match = _QUANTITY_RE.search(cleaned_text)
    if not match:
        logger.debug("preprocess.extract_quantity: %r -> None (no quantity found)", cleaned_text)
        return None
    multiplier = float(match.group(1)) if match.group(1) else 1.0
    amount = float(match.group(2))
    unit = match.group(3)
    quantity = multiplier * amount * _UNIT_TO_BASE[unit]
    logger.debug("preprocess.extract_quantity: %r -> %s (base unit)", cleaned_text, quantity)
    return quantity
