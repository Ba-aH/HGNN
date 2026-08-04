import os

# ── Paths ────────────────────────────────────────────────────────────────────
EXTRACTED_DIR    = r"C:\Users\behan\OneDrive\Desktop\Master_Internship\scrapping\data_prep\extracted"
REF_FILENAME     = "referenced_work.md"
RML_MAPPING_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mapping.rml.ttl")

# ── Local LLM ─────────────────────────────────────────────────────────────────
LOCAL_LLM_URL = "http://localhost:1234/api/v1/chat"
LOCAL_LLM_MODEL = "qwen/qwen3.6-35b-a3b"  
LINES_PER_CHUNK = 80

# ── OpenAlex ─────────────────────────────────────────────────────────────────
OPENALEX_CREDENTIALS = [
    {"key": "h2VQLBHlgD4wBNOuroywew",  "email": "behantous@gmail.com"},
    {"key": "ZmYuOfeVVSYxVLZRyL0SwZ",  "email": "bahahantous@gmail.com"},
    {"key": "zruTIcT4eZmLhFzIUZ69hZ",  "email": "brensm3allem@gmail.com"},
    {"key": "i8l5VxjlMHHE5FYLUuV5PN",  "email": "pinkyboan@gmail.com"},
    {"key": "vbL8IZElFROAwaRgJ7bUO2",  "email": "tousa.shop.contact@gmail.com"},
]
OPENALEX_CACHE_FILE = os.path.join(EXTRACTED_DIR, "_openalex_cache.json")

# ── Checkpoint ────────────────────────────────────────────────────────────────
CHECKPOINT_FILE = os.path.join(EXTRACTED_DIR, "_checkpoint.json")