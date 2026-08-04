import json
import threading

from config import OPENALEX_CACHE_FILE

# Guards concurrent writes to the shared cache dict across threads
cache_lock = threading.Lock()


def load_cache() -> dict:
    """Load the OpenAlex cache from disk. Returns an empty dict if not found."""
    try:
        with open(OPENALEX_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_cache(cache: dict) -> None:
    """Persist the cache dict to disk. Call once after all parallel work finishes."""
    with open(OPENALEX_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
