import re
import urllib.parse
from rapidfuzz import fuzz
import unicodedata
import requests
from cache import cache_lock, load_cache, save_cache
from config import OPENALEX_CREDENTIALS
from enchant import Dict
import wordninja

_key_index = 0
en_dict    = Dict("en_US")

_BATCH_SIZE = 25   # safe OR-filter batch size for OpenAlex


# ── Credentials ───────────────────────────────────────────────────────────────

def get_openalex_credentials() -> dict:
    """Return the next (key, email) pair, rotating round-robin."""
    global _key_index
    creds = OPENALEX_CREDENTIALS[_key_index % len(OPENALEX_CREDENTIALS)]
    _key_index += 1
    return creds


# ── Text helpers ──────────────────────────────────────────────────────────────

def extract_main_title(title: str) -> str:
    """Strip subtitle — everything after ':', '|', em-dash, or en-dash."""
    if not title:
        return ""
    main = re.split(r'[:\|—–]', title)[0].strip()
    main = re.sub(r'[^\w\s]$', '', main).strip()
    return main if len(main) > 10 else title


def fix_merged_words(title: str) -> str:
    tokens = title.split()
    fixed_tokens = []
    for token in tokens:
        clean = re.sub(r"'s$", "", token.lower())
        clean = re.sub(r"[^a-z]", "", clean)
        if len(clean) > 9 and not en_dict.check(clean):
            split = wordninja.split(clean)
            if (
                len(split) > 1
                and all(len(w) > 2 for w in split)
                and all(en_dict.check(w) for w in split)
            ):
                fixed_tokens.append(" ".join(split))
                continue
        fixed_tokens.append(token)
    return " ".join(fixed_tokens)


def normalize_text(text: str) -> str:
    """Normalize text: lowercase, remove accents, clean punctuation."""
    if not text:
        return ""
    # NFKD + ascii folding
    nfkd = unicodedata.normalize('NFKD', text)
    text = nfkd.encode('ascii', 'ignore').decode('ascii')
    # Lowercase + remove non-word chars except spaces
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_search_query(title: str) -> str:
    """Clean title for OpenAlex title.search"""
    if not title:
        return ""
    title = extract_main_title(title)
    title = title.split(",")[0].strip()
    title = fix_merged_words(title)
    title = normalize_text(title)          
    return title

def normalize_author_name(name: str) -> str:
    """Clean author name: lowercase, remove accents, punctuation."""
    if not name:
        return ""
    nfkd = unicodedata.normalize('NFKD', name)
    text = nfkd.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def author_match(ref_authors: list, oa_authors: list, ref_year: int | None = None, oa_year: int | None = None) -> bool:
    """
    Lenient author matching with exact year requirement for strong rescue.
    """
    if not ref_authors or not oa_authors:
        return False

    # Exact year match flag (as requested - no tolerance)
    exact_year_match = (ref_year is not None and oa_year is not None and ref_year == oa_year)

    for ref_author in ref_authors:
        if not ref_author:
            continue
        ref_norm = normalize_author_name(ref_author)
        ref_parts = [p for p in ref_norm.split() if p]
        if not ref_parts:
            continue

        ref_last = ref_parts[-1]
        ref_full = ref_norm

        for oa_author in oa_authors:
            if not oa_author:
                continue
            oa_norm = normalize_author_name(oa_author)
            oa_parts = [p for p in oa_norm.split() if p]
            if not oa_parts:
                continue

            oa_last = oa_parts[-1]

            # 1. Last name - very lenient
            if len(ref_last) >= 3 and len(oa_last) >= 3:
                if fuzz.ratio(ref_last, oa_last) >= 75 or fuzz.token_sort_ratio(ref_last, oa_last) >= 80:  # was 32/42
                    return True

            # 2. Full name token matching
            if fuzz.token_set_ratio(ref_full, oa_norm) >= 85:
                return True
            if fuzz.partial_ratio(ref_full, oa_norm) >= 85:
                return True
            if fuzz.token_sort_ratio(ref_full, oa_norm) >= 80:
                return True

            # 3. Initials matching with year boost
            ref_initials = [w[0] for w in ref_parts if w and w[0].isalpha()]
            oa_initials = [w[0] for w in oa_parts if w and w[0].isalpha()]

            if ref_initials and oa_initials:
                if ref_initials == oa_initials:
                    return True
                # Last initial match + exact year = strong signal
                if ref_initials[-1] == oa_initials[-1] and exact_year_match:
                    return True

    return False

# ── Shared scoring logic ──────────────────────────────────────────────────────

def _score_and_build_result(ref: dict, match: dict) -> dict:
    
    title = ref.get("title", "")
    oa_title = match.get("title") or ""
    ref_year = ref.get("year")
    oa_year = match.get("publication_year")

    # Title similarity (normalized)
    norm_ref_title = normalize_text(extract_main_title(title))
    norm_oa_title = normalize_text(extract_main_title(oa_title))
    title_similarity = fuzz.token_set_ratio(norm_ref_title, norm_oa_title) / 100.0

    # Authors
    oa_authors = [
        a.get("author", {}).get("display_name")
        for a in match.get("authorships", [])
        if a.get("author", {}).get("display_name")
    ]
    ref_authors = [
        a.get("name", "") if isinstance(a, dict) else str(a)
        for a in ref.get("authors", [])
    ]

    authors_confirmed = author_match(ref_authors, oa_authors, ref_year, oa_year)
    exact_year_match = (ref_year is not None and oa_year is not None and ref_year == oa_year)

    score = title_similarity

    if score < 0.85:
        if exact_year_match and authors_confirmed and score >= 0.65:
            score = 1.0                              
        elif authors_confirmed and score >= 0.70:
            score = 0.90                             
        else:
            return {
                "openalex_found": False,
                "openalex_score": round(score * 100, 1)
            }


    oa_citation_count = match.get("cited_by_count")   # ← this is the correct field name in OpenAlex

    # Build successful result
    venue = (match.get("primary_location") or {}).get("source") or {}
    return {
        "title":          oa_title,
        "authors":        oa_authors or ref.get("authors", []),
        "year":           oa_year or ref_year,
        "doi":            match.get("doi") or ref.get("doi"),
        "venue":          venue.get("display_name") or ref.get("venue"),
        "openalex_id":    match.get("id"),
        "openalex_found": True,
        "openalex_score": round(score * 100, 1),
        "citation_count": oa_citation_count,
        "openalex_authors":  oa_authors,          
    }

# ── Low-level fetch helpers ───────────────────────────────────────────────────

def _get_headers() -> dict:
    creds = get_openalex_credentials()
    return {
        "User-Agent":    f"mailto:{creds['email']}",
        "Authorization": f"Bearer {creds['key']}",
    }


def _fetch_single(search_title: str, per_page: int = 5) -> list:
    """One title.search request — no fallback."""
    encoded = urllib.parse.quote(search_title)
    resp    = requests.get(
        f"https://api.openalex.org/works?filter=title.search:{encoded}&per_page={per_page}",
        timeout=10,
        headers=_get_headers(),
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


def _fetch_batch(search_titles: list[str]) -> list:
    """
    Fetch up to _BATCH_SIZE titles in ONE request using OR (|) filter syntax.
    Returns the combined list of works from OpenAlex.
    """
    filter_value = "|".join(urllib.parse.quote(t) for t in search_titles)
    per_page     = min(len(search_titles) * 5, 200)  # OpenAlex hard max is 200
    resp = requests.get(
        f"https://api.openalex.org/works?filter=title.search:{filter_value}&per_page={per_page}",
        timeout=15,
        headers=_get_headers(),
    )
    resp.raise_for_status()
    return resp.json().get("results", [])


# ── Public API ────────────────────────────────────────────────────────────────

def validate_with_openalex(ref: dict, cache: dict | None = None) -> dict:
    """
    Validate a single reference — kept for backwards-compat.
    Fires at most ONE OpenAlex request (no fallback).
    """
    _own_cache = cache is None
    if _own_cache:
        cache = load_cache()

    title = ref.get("title", "").strip()
    if not title:
        return ref

    title     = re.sub(r'\s+', ' ', title.replace("_", " ")).strip()
    cache_key = title.lower()

    if cache_key in cache:
        cached = cache[cache_key]
        # A score of 0.0 means a previous run failed to find the paper
        # (crowded out of results). Don't trust it — retry live.
        if cached.get("openalex_score", 0.0) > 0.0:
            ref.update(cached)
            ref["openalex_from_cache"] = True
            return ref

    try:
        results = _fetch_single(build_search_query(title))

        if not results:
            result = {"openalex_found": False, "openalex_score": 0.0}
        else:
            best = max(
                results,
                key=lambda r: fuzz.token_set_ratio(
                    normalize_text(extract_main_title(title)),           # ← fixed
                    normalize_text(extract_main_title(r.get("title") or "")),
                ),
            )
            result = _score_and_build_result(ref, best)

        # Only persist a definitive result; 0.0 means "not found in results",
        # which may change on the next run — don't lock it into the cache.
        if result.get("openalex_score", 0.0) > 0.0:
            with cache_lock:
                cache[cache_key] = result
        if _own_cache:
            save_cache(cache)
        ref.update(result)

    except Exception as e:
        ref["openalex_found"] = False
        ref["openalex_error"] = str(e)

    return ref


def validate_batch_with_openalex(refs: list[dict], cache: dict | None = None) -> list[dict]:
    """
    Validate a list of references with the fewest possible OpenAlex requests.

    Cost: ceil(unique_cache_miss_titles / _BATCH_SIZE) requests.
    For 30 refs → at most 2 requests (vs 30 with the old thread-pool approach).

    The caller is responsible for calling save_cache() after this returns
    when passing a pre-loaded cache dict.
    """
    _own_cache = cache is None
    if _own_cache:
        cache = load_cache()

    # ── Normalise + split cache hits from misses ──────────────────────────────
    pending: list[tuple[dict, str, str]] = []   # (ref, cache_key, search_title)

    for ref in refs:
        title = ref.get("title", "").strip()
        if not title:
            continue
        title     = re.sub(r'\s+', ' ', title.replace("_", " ")).strip()
        cache_key = title.lower()

        if cache_key in cache:
            cached = cache[cache_key]
            # Skip stale failures so they get retried live
            if cached.get("openalex_score", 0.0) > 0.0:
                ref.update(cached)
                ref["openalex_from_cache"] = True
                continue
        pending.append((ref, cache_key, build_search_query(title)))

    if not pending:
        if _own_cache:
            save_cache(cache)
        return refs

    # ── Deduplicate identical search titles ───────────────────────────────────
    # Multiple refs can share the same cleaned search title — only one API call needed.
    title_to_refs: dict[str, list[tuple[dict, str]]] = {}
    for ref, cache_key, search_title in pending:
        title_to_refs.setdefault(search_title, []).append((ref, cache_key))

    unique_titles = list(title_to_refs.keys())

    # ── Batch API calls ───────────────────────────────────────────────────────
    for chunk_start in range(0, len(unique_titles), _BATCH_SIZE):
        chunk = unique_titles[chunk_start: chunk_start + _BATCH_SIZE]

        try:
            works = _fetch_batch(chunk)
        except Exception as e:
            for st in chunk:
                for ref, _ in title_to_refs[st]:
                    ref["openalex_found"] = False
                    ref["openalex_error"] = str(e)
            continue

        # Score every ref against the returned works
        for search_title in chunk:
            for ref, cache_key in title_to_refs[search_title]:
                orig_title = ref.get("title", "")

                if not works:
                    result = {"openalex_found": False, "openalex_score": 0.0}
                else:
                    best = max(
                        works,
                        key=lambda w: fuzz.token_set_ratio(
                            normalize_text(extract_main_title(orig_title)),   
                            normalize_text(extract_main_title(w.get("title") or "")),
                        ),
                    )
                    result = _score_and_build_result(ref, best)

                # ── Fallback: score 0.0 means crowded out of batch results.
                #    Retry individually before giving up.
                if result.get("openalex_score", 0.0) == 0.0:
                    try:
                        single_results = _fetch_single(search_title, per_page=5)
                        if single_results:
                            best_single = max(
                                single_results,
                                key=lambda w: fuzz.token_set_ratio(
                                    normalize_text(extract_main_title(orig_title)),   # ← fixed
                                    normalize_text(extract_main_title(w.get("title") or "")),
                                ),
                            )
                            fallback = _score_and_build_result(ref, best_single)
                            if fallback.get("openalex_score", 0.0) > 0.0:
                                result = fallback
                    except Exception:
                        pass  # keep the 0.0 result; don't crash the batch

                # Only cache definitive results — not 0.0 failures
                if result.get("openalex_score", 0.0) > 0.0:
                    with cache_lock:
                        cache[cache_key] = result
                ref.update(result)

    if _own_cache:
        save_cache(cache)

    return refs