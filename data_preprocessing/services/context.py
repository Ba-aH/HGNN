"""
context.py — Regex-Based Context Extraction Service for CiteKG Pipeline 2

Reads:
  <EXTRACTED_DIR>/<folder>/<folder>.md       — full paper text
  <EXTRACTED_DIR>/<folder>/references.json   — paper_id, reference list

Returns a dict structured as:
  {
    "paper_id": "",
    "global_context": { "title": "", "abstract": "" },
    "citations": [
      {
        "reference":       { "title": "" },
        "citation_marker": "[1]",
        "citation_type":   "single" | "multiple",
        "source_marker":   "1" | null,   ← null when citation_type == "single"
        "local_context":   { "before": "", "after": "" }
      }
    ],
    "total_citations": 0
  }

Failure logs
------------
  <EXTRACTED_DIR>/context_failures.json  — one record per folder that failed,
      with a "reason" tag so you can triage missing context.json in bulk.
  <EXTRACTED_DIR>/ocr_failures.json      — one record per image that OCR could
      not process, keyed by folder + image path.
"""

import json
import re
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from pix2tex.cli import LatexOCR

from config import EXTRACTED_DIR

GLOBAL_HEAD_CHARS = 3000    # title/abstract always sit at the top
CONTEXT_CHARS     = 150     # max chars for before / after windows

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# LatexOCR model — loaded ONCE at module import time.
# ─────────────────────────────────────────────────────────────────────────────

_latex_ocr: LatexOCR | None = None

def _get_ocr() -> LatexOCR:
    global _latex_ocr
    if _latex_ocr is None:
        logger.info("Loading LatexOCR model...")
        _latex_ocr = LatexOCR()
        logger.info("LatexOCR model ready.")
    return _latex_ocr


# ─────────────────────────────────────────────────────────────────────────────
# Failure logging  (written atomically; safe under multi-threading)
# ─────────────────────────────────────────────────────────────────────────────

_CONTEXT_FAILURES_PATH = Path(EXTRACTED_DIR) / "context_failures.json"
_OCR_FAILURES_PATH     = Path(EXTRACTED_DIR) / "ocr_failures.json"
_context_failures_lock = threading.Lock()
_ocr_failures_lock     = threading.Lock()


def _append_failure(path: Path, lock: threading.Lock, record: dict) -> None:
    """Atomically append *record* to a JSON-array failure log at *path*."""
    with lock:
        if path.exists():
            try:
                with path.open(encoding="utf-8") as f:
                    failures = json.load(f)
            except (json.JSONDecodeError, OSError):
                failures = []
        else:
            failures = []

        failures.append(record)

        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        tmp.replace(path)


def _log_context_failure(folder: str, reason: str, detail: str = "") -> None:
    """
    Append a context-extraction failure to context_failures.json.

    reason  — short machine-readable tag, e.g. "no_md_file", "bad_json",
              "no_references_json", "unexpected_error"
    detail  — human-readable elaboration (exception message, file list, …)
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folder":    folder,
        "reason":    reason,
        "detail":    detail,
    }
    _append_failure(_CONTEXT_FAILURES_PATH, _context_failures_lock, record)
    logger.error(f"[{folder}] context_failure reason={reason!r}: {detail}")


def _log_ocr_failure(folder: str, img_path: Path, error: str) -> None:
    """Append an OCR failure to ocr_failures.json."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folder":    folder,
        "image":     str(img_path),
        "error":     error,
    }
    _append_failure(_OCR_FAILURES_PATH, _ocr_failures_lock, record)
    logger.warning(f"[{folder}] ocr_failure image={img_path}: {error}")


# ─────────────────────────────────────────────────────────────────────────────
# Formula image replacement
# ─────────────────────────────────────────────────────────────────────────────

# Matches:  <img src="imgs/foo.jpg" ... />   or   <img src="imgs/foo.png" ... >
# src= may appear anywhere among other attributes (alt, width, style, etc.)
IMG_TAG_RE = re.compile(
    r'<img\b[^>]*\bsrc="([^"]+\.(?:jpg|jpeg|png))"[^>]*/?>',
    re.IGNORECASE,
)


def _replace_formula_images(text: str, paper_folder: Path, folder_name: str = "") -> str:
    """
    Find all <img src="..."> tags in *text*, run LatexOCR on each image,
    and replace the tag with <latex>$$ … $$</latex>.

    Image paths are resolved relative to *paper_folder*.
    Tags whose image cannot be found or OCR'd are left unchanged; every
    failure is logged to ocr_failures.json.
    """
    def _substitute(match: re.Match) -> str:
        src      = match.group(1)           # e.g. "imgs/img_foo.jpg"
        img_path = paper_folder / src

        if not img_path.exists():
            _log_ocr_failure(folder_name, img_path, "image file not found")
            return match.group(0)           # leave tag as-is

        try:
            # FIX: always convert to RGB — RGBA PNGs will crash LatexOCR
            image = Image.open(img_path).convert("RGB")
            latex = _get_ocr()(image)

            # FIX: LatexOCR can return None for very small / low-confidence images
            if not latex:
                _log_ocr_failure(folder_name, img_path,
                                 "LatexOCR returned None or empty string")
                return match.group(0)

            latex = latex.strip().replace("\\\\", "\\")
            return f"<latex>$$ {latex} $$</latex>"

        except Exception as e:
            _log_ocr_failure(folder_name, img_path, str(e))
            return match.group(0)           # leave tag as-is

    return IMG_TAG_RE.sub(_substitute, text)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Global context via regex
# ─────────────────────────────────────────────────────────────────────────────

def _extract_global_context(paper_text: str) -> dict:
    head = paper_text[:GLOBAL_HEAD_CHARS]

    # ── Title ──────────────────────────────────────────────────────────────
    title = ""
    if m := re.search(r'^#\s+(.+)', head, re.MULTILINE):
        title = m.group(1).strip()
    else:
        for line in head.splitlines():
            if line.strip():
                title = line.strip()
                break

    # ── Abstract ───────────────────────────────────────────────────────────
    abstract = ""
    abs_pattern = re.compile(
        r'(?:^|\n)'
        r'(?:#+\s*)?'
        r'Abstract[:\s]*\n+'
        r'(.*?)'
        r'(?=\n{2,}(?:[A-Z#]|\d+\.)|'
        r'\n#+\s|$)',
        re.IGNORECASE | re.DOTALL,
    )
    if m := abs_pattern.search(head):
        abstract = re.sub(r'\s+', ' ', m.group(1)).strip()

    return {"title": title, "abstract": abstract}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Citation marker detection via regex
# ─────────────────────────────────────────────────────────────────────────────

# Character class covering ASCII letters plus the full Latin-1 Supplement and
# Latin Extended-A/B blocks (handles é, è, ó, ñ, Á, Ā, ł, ž, etc.)
_SC = r"A-Za-zÀ-ÖØ-öø-ÿĀ-ɏ'-"   # surname characters (continue)
_SU = r"A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞ"  # surname uppercase starters

# Surname atom:  Hansson  /  Alchourrón  /  Fermé  /  van der Berg  are all covered.
# "et al." allows one or two trailing periods (et al. vs et al..)
_SURNAME = (
    rf"[{_SU}][{_SC}]+"
    rf"(?:\s+et\s+al\.{{1,2}}|\s+(?:and|&)\s+[{_SU}][{_SC}]+)?"
)

# Full author-year atom: "Parsons et al., 1998" or "Wang and Chang, 2016a"
_AY_ATOM = rf"{_SURNAME},\s*\d{{4}}[a-z]?"
_AY_ATOM_NOCOMMA = rf"{_SURNAME}\s+\d{{4}}[a-z]?"

CITATION_RE = re.compile(
    rf"""(?:
        # ── Numeric bracketed: [1]  [1,2]  [1–3]  [1; 2]  [1, 3-5, 7] ──
        \[(?:\d+(?:\s*[-\u2012\u2013\u2014,;]\s*\d+)*)\]

      |
        # ── Parenthetical author-year ─────────────────────────────────────
        # Optionally prefixed with "e.g.," or "i.e.,"
        \(
            (?:(?:e\.g\.|i\.e\.),?\s*)?
            (?:{_AY_ATOM}|{_AY_ATOM_NOCOMMA})
            (?:
                \s*;\s*
                (?:
                    (?:{_AY_ATOM}|{_AY_ATOM_NOCOMMA})   # full author-year continuation
                  |
                    \d{{4}}[a-z]?                        # year-only continuation
                )
            )*
        \)

      |
        # ── Inline author-year: "Author (Year)" or "Author et al.(Year)" ──
        # The opening parenthesis may be glued to the name (no space).
        {_SURNAME}[ \t]*\(\d{{4}}[a-z]?\)
    )""",
    re.VERBOSE | re.UNICODE,
)


def _detect_citations(paper_text: str) -> list[dict]:
    """
    Scan paper_text for all citation markers and return a list of dicts:
      [{"marker": "...", "before": "...", "after": "..."}, ...]
    """
    results = []
    for m in CITATION_RE.finditer(paper_text):
        start, end = m.start(), m.end()
        before = paper_text[max(0, start - CONTEXT_CHARS): start]
        after  = paper_text[end: end + CONTEXT_CHARS]

        before = re.sub(r'[ \t]+', ' ', before).strip()
        after  = re.sub(r'[ \t]+', ' ', after).strip()

        # Replace any citation markers nested inside the context windows
        before = CITATION_RE.sub("[cite]", before)
        after  = CITATION_RE.sub("[cite]", after)

        results.append({"marker": m.group(), "before": before, "after": after})
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Marker expansion and resolution
# ─────────────────────────────────────────────────────────────────────────────

def _expand_numeric_marker(marker: str) -> list[int]:
    """
    '[1, 3-5, 7]' → [1, 3, 4, 5, 7]
    """
    indices = []
    inner = marker.strip("[]")
    for part in re.split(r'[,;]\s*', inner):
        part = part.strip()
        if rng := re.match(r'(\d+)\s*[-\u2012\u2013\u2014]\s*(\d+)', part):
            indices.extend(range(int(rng.group(1)), int(rng.group(2)) + 1))
        elif part.isdigit():
            indices.append(int(part))
    return indices


def _resolve_numeric_marker(marker: str, references: list) -> list[tuple[dict, str]]:
    """
    '[1, 2]' → [(references[0], "1"), (references[1], "2")]
    """
    resolved = []
    for idx in _expand_numeric_marker(marker):
        i = idx - 1
        if 0 <= i < len(references):
            resolved.append((references[i], str(idx)))
    return resolved


# Matches a single author-year atom inside a parenthetical marker.
# Handles: "Hansson, 1999" / "Fermé et al., 2003" / "Wang and Chang, 2016a"
# Also handles double period: "Jennings et al.., 2001"
AUTHOR_YEAR_ATOM_RE = re.compile(
    rf"([{_SU}][{_SC}]+"
    rf"(?:\s+(?:et\s+al\.{{1,2}}|(?:and|&)\s+[{_SU}][{_SC}]+))?)"
    rf"[,\s]\s*(\d{{4}}[a-z]?)",
    re.UNICODE,
)


def _normalize_marker_for_resolution(marker: str) -> str:
    """
    Convert any citation marker to a consistent parenthetical form so that
    ``_resolve_author_year_marker`` can always use the same parsing logic.

    Examples
    --------
    'Rahwan et al. (2003)'           → '(Rahwan et al., 2003)'
    'Fermé et al.(2003)'             → '(Fermé et al., 2003)'
    '(e.g., Parsons et al., 1998)'   → '(Parsons et al., 1998)'
    '(Parsons et al., 1998)'         → '(Parsons et al., 1998)'  [unchanged]
    """
    m = marker.strip()

    if not m.startswith("(") and not m.startswith("["):
        inline = re.match(r'^(.+?)[ \t]*\((\d{4}[a-z]?)\)$', m, re.UNICODE)
        if inline:
            author = inline.group(1).rstrip(' \t')          # ← no period strip
            author = re.sub(r'\.{2,}$', '.', author)        # ← collapse al.. → al.
            return f"({author}, {inline.group(2)})"
        return m

    m = re.sub(r'^\((?:e\.g\.|i\.e\.),?\s*', '(', m, flags=re.UNICODE)
    return m


def _resolve_author_year_marker(marker: str, references: list) -> list[tuple[dict, str]]:
    resolved = []
    for atom in re.split(r';\s*', marker.strip("()")):
        atom = atom.strip()
        if not atom:
            continue
        m = AUTHOR_YEAR_ATOM_RE.match(atom)
        if not m:
            continue
        surname = m.group(1).split()[0].lower()
        year    = m.group(2)[:4]
        # Compile once; \b ensures "Li" doesn't match inside "Williams"
        surname_re = re.compile(rf'\b{re.escape(surname)}\b', re.IGNORECASE)
        for r in references:
            authors_str = str(r.get("authors", []))
            if str(r.get("year", "")) == year and surname_re.search(authors_str):
                resolved.append((r, atom))
                break
    return resolved


def _resolve_marker(marker: str, references: list) -> list[tuple[dict, str]]:
    """Dispatch to the appropriate resolver based on marker format."""
    if marker.startswith("["):
        return _resolve_numeric_marker(marker, references)
    # Author-year: normalise first so both inline and parenthetical are handled
    # identically by _resolve_author_year_marker.
    normalized = _normalize_marker_for_resolution(marker)
    if normalized.startswith("("):
        return _resolve_author_year_marker(normalized, references)
    return []


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

# .md filenames that are never the main paper body
_MD_EXCLUDE = {"referenced_work.md", "cleaned_references.md"}


def _find_paper_md(folder_path: Path, folder_name: str) -> Path:
    """
    Return the main paper .md inside folder_path.
    1. Tries <folder_name>.md first (canonical).
    2. Falls back to any .md not in _MD_EXCLUDE.
    Raises FileNotFoundError with a detailed diagnostic message if nothing
    qualifies (lists every .md actually present so callers can log it).
    """
    canonical = folder_path / f"{folder_name}.md"
    if canonical.exists():
        return canonical

    # List ALL .md files present for diagnostics
    all_mds = sorted(p.name for p in folder_path.glob("*.md"))

    candidates = [p for p in folder_path.glob("*.md") if p.name not in _MD_EXCLUDE]
    if candidates:
        chosen = candidates[0]
        logger.info(
            f"[{folder_name}] Canonical '{folder_name}.md' not found; "
            f"using fallback: '{chosen.name}' "
            f"(all .md files present: {all_mds})"
        )
        return chosen

    raise FileNotFoundError(
        f"No paper .md found in {folder_path}. "
        f"All .md files present: {all_mds}. "
        f"Expected canonical name: '{folder_name}.md' or any .md "
        f"not in {sorted(_MD_EXCLUDE)}."
    )


def extract_context_for_folder(folder_name: str) -> dict:
    """
    Extract citation context for *folder_name* and return the result dict.

    On any recoverable error the failure is logged to context_failures.json
    and the exception is re-raised so the caller can skip this folder.
    """
    path      = Path(EXTRACTED_DIR) / folder_name
    refs_path = path / "references.json"

    # ── Locate paper .md ───────────────────────────────────────────────────
    try:
        md_path = _find_paper_md(path, folder_name)
    except FileNotFoundError as e:
        _log_context_failure(folder_name, "no_md_file", str(e))
        raise

    # ── Load references.json ───────────────────────────────────────────────
    if not refs_path.exists():
        detail = f"references.json not found at {refs_path}"
        _log_context_failure(folder_name, "no_references_json", detail)
        raise FileNotFoundError(detail)

    try:
        with refs_path.open(encoding="utf-8") as f:
            refs_data = json.load(f)
    except json.JSONDecodeError as e:
        detail = f"references.json is malformed: {e}"
        _log_context_failure(folder_name, "bad_references_json", detail)
        raise

    references = refs_data.get("references", [])
    paper_id   = refs_data.get("paper", {}).get("paper_id", folder_name)
    logger.info(f"[{folder_name}] Loaded {len(references)} references from {md_path.name}")

    # ── Read paper text ────────────────────────────────────────────────────
    try:
        paper_text = md_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as e:
        detail = f"Could not read {md_path}: {e}"
        _log_context_failure(folder_name, "md_read_error", detail)
        raise

    # ── Step 1: Global context ─────────────────────────────────────────────
    logger.info(f"[{folder_name}] Extracting global context...")
    glob = _extract_global_context(paper_text)

    # ── Step 2: Detect all citation markers with surrounding context windows
    logger.info(f"[{folder_name}] Scanning for citation markers...")
    raw_detections = _detect_citations(paper_text)

    # ── Step 3: Expand, resolve, and replace formula images ───────────────
    seen, citations = set(), []
    _SNIPPET_LEN = 40

    try:
        for d in raw_detections:
            m, b, a = d["marker"], d["before"], d["after"]

            if IMG_TAG_RE.search(b) or IMG_TAG_RE.search(a):
                logger.info(
                    f"[{folder_name}] Running LatexOCR on formula images "
                    f"in context window of marker {m!r}..."
                )
                b = _replace_formula_images(b, path, folder_name)
                a = _replace_formula_images(a, path, folder_name)

            resolved    = _resolve_marker(m, references)
            is_multiple = len(resolved) > 1

            for ref, source_marker in resolved:
                title     = ref.get("title", "Untitled")
                dedup_key = (m, title, b[-_SNIPPET_LEN:])
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                citations.append({
                    "reference":       {"title": title},
                    "citation_marker": m,
                    "citation_type":   "multiple" if is_multiple else "single",
                    "source_marker":   source_marker if is_multiple else None,
                    "local_context":   {"before": b, "after": a},
                })

    except Exception as e:
        detail = f"Unexpected error during citation processing: {e}"
        _log_context_failure(folder_name, "unexpected_error", detail)
        raise

    return {
        "paper_id":        paper_id,
        "global_context":  glob,
        "citations":       citations,
        "total_citations": len(citations),
    }