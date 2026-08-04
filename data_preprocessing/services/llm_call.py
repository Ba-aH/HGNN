import json
import re
import requests
from config import LOCAL_LLM_URL, LOCAL_LLM_MODEL, LINES_PER_CHUNK


def split_into_chunks(text: str) -> list[str]:
    lines = text.splitlines()
    chunks = []
    for i in range(0, len(lines), LINES_PER_CHUNK):
        chunk = "\n".join(lines[i:i + LINES_PER_CHUNK]).strip()
        if chunk:
            chunks.append(chunk)
    return chunks if chunks else [text]


def call_llm(raw_content: str) -> list:
    payload = {
        "model": LOCAL_LLM_MODEL,
        "system_prompt": (
            "You are an academic reference extractor. "
            "Return ONLY a valid JSON array. No explanations, no markdown, no thinking."
        ),
        "input": (
            "Extract references into a JSON array. Each item must have:\n"
            "- title (string, required)\n"
            "- authors (array of strings)\n"
            "- year (number or null)\n"
            "- venue (string or null)\n\n"
            "Rules: split merged entries "
            f"Reference list:\n\n{raw_content}"
        ),
        "temperature": 0.0,
    }

    resp = requests.post(LOCAL_LLM_URL, json=payload, timeout=180)
    resp.raise_for_status()

    output_blocks = resp.json().get("output", [])

    # Per the API docs, the final answer block has type "message" (not "text")
    raw = ""
    for block in output_blocks:
        if block.get("type") == "message":
            raw = block.get("content", "").strip()
            if raw:
                break

    # Fallback to text block
    if not raw:
        for block in output_blocks:
            if block.get("type") == "text":
                raw = block.get("content", "").strip()
                if raw:
                    break

    # Strip any leaked <think> tags just in case
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()

    # Strip markdown fences
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw).strip()

    if not raw:
        raise ValueError(f"LLM returned no usable content. Blocks: {output_blocks}")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned non-JSON: {e}\nRaw: {raw[:300]}")

    if isinstance(parsed, dict):
        for key in ("references", "data", "items", "results"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        return [parsed]

    if isinstance(parsed, list):
        return parsed

    return [{"title": "Parsing failed", "authors": [], "raw": raw}]