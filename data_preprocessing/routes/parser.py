import difflib
import json
import re

from flask import Blueprint, jsonify, request

bp = Blueprint("parser", __name__)


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r'^```[a-z]*\n?', '', text.strip())
    text = re.sub(r'\n?```$',       '', text.strip())
    return text


# ── POST /parse_llm_output ───────────────────────────────────────────────────
@bp.route("/parse_llm_output", methods=["POST"])
def parse_llm_output():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    # Accept either a raw llm response envelope or a plain content string
    llm_response = None
    if 'choices' in data and data['choices']:
        choice = data['choices'][0]
        if isinstance(choice, dict) and 'message' in choice:
            llm_response = choice['message'].get('content')
    if not llm_response and isinstance(data.get('content'), str):
        llm_response = data.get('content')

    if isinstance(llm_response, str):
        llm_response = _strip_markdown_fences(llm_response)
        try:
            ref = json.loads(llm_response)
            if isinstance(ref, dict):
                ref = [ref]
            elif not isinstance(ref, list):
                ref = []
        except Exception:
            ref = [{"title": "Parsing failed", "authors": [], "raw": llm_response}]
    else:
        ref = llm_response if isinstance(llm_response, list) else [llm_response]

    cleaned_ref = ref[0] if ref else {"title": "No reference found", "authors": []}
    folder      = data.get('folder') or data.get('original_folder') or "unknown"

    return jsonify({"originalRef": cleaned_ref, "folder": folder}), 200


# ── POST /fix_reference ───────────────────────────────────────────────────────
@bp.route("/fix_reference", methods=["POST"])
def fix_reference():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Missing JSON body"}), 400

    llm_ref = data.get('originalRef', {})
    folder   = data.get('folder') or llm_ref.get('folder') or "unknown"

    openalex_response = data.get('openalexResults') or data
    openalex_data     = None
    if isinstance(openalex_response, dict):
        results = openalex_response.get('results')
        if results and isinstance(results, list) and len(results) > 0:
            openalex_data = results[0]
        elif openalex_response.get('id'):
            openalex_data = openalex_response

    fixed            = dict(llm_ref)
    similarity_score = 0.0

    if openalex_data and openalex_data.get('title'):
        llm_title = str(llm_ref.get('title', '')).lower().strip()
        oa_title   = str(openalex_data.get('title', '')).lower().strip()
        if llm_title and oa_title:
            similarity_score = difflib.SequenceMatcher(None, llm_title, oa_title).ratio()
            if similarity_score >= 0.80:
                fixed['title'] = openalex_data.get('title')

    if openalex_data and openalex_data.get('authorships'):
        authors_list = [
            a.get('author', {}).get('display_name')
            for a in openalex_data['authorships']
            if a.get('author', {}).get('display_name')
        ]
        if authors_list:
            fixed['authors'] = authors_list

    if openalex_data:
        if not fixed.get('year') and openalex_data.get('publication_year'):
            fixed['year'] = openalex_data['publication_year']
        if not fixed.get('doi') and openalex_data.get('doi'):
            fixed['doi'] = openalex_data['doi']
        if not fixed.get('journal'):
            venue = openalex_data.get('host_venue') or openalex_data.get('primary_location')
            if venue and venue.get('display_name'):
                fixed['journal'] = venue['display_name']

    fixed['title'] = str(fixed.get('title') or llm_ref.get('title') or "Untitled Reference").strip()
    if not isinstance(fixed.get('authors'), list) or len(fixed.get('authors', [])) == 0:
        fixed['authors'] = llm_ref.get('authors') or ["Unknown Author"]

    fixed['folder']     = folder
    fixed['validation'] = {
        "fixedByOpenAlex":   bool(openalex_data) and similarity_score >= 0.80,
        "openalexId":        openalex_data.get('id') if openalex_data else None,
        "openalexFound":     bool(openalex_data),
        "titleSimilarity":   round(similarity_score * 100, 1),
        "usedOpenAlexTitle": similarity_score >= 0.80,
    }

    return jsonify(fixed), 200
