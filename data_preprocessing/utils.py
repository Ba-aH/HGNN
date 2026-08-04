import re
import hashlib


def make_paper_id(title: str) -> str:
    """
    Generates a stable unique ID from a normalised title.
    Deterministic — same title always produces the same ID.
    """
    normalized = title.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)  # remove punctuation
    normalized = re.sub(r'\s+', ' ', normalized)      # collapse whitespace
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def make_author_id(name: str) -> str:
    """Stable unique ID derived from a normalised author name."""
    normalized = name.lower().strip()
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
