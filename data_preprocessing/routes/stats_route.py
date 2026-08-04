"""
stats_route.py — Aggregated statistics over all context.json files.

Endpoints:
    GET           → serves the dashboard HTML
    GET /stats/data     → returns aggregated JSON consumed by the dashboard
"""

import json
import os
import re
from pathlib import Path

from flask import Blueprint, jsonify, send_file

from config import EXTRACTED_DIR

bp = Blueprint("stats", __name__)

CONTEXT_FILENAME = "context_clean.json"

_LATEX_RE   = re.compile(r"<latex>")
_FAILOCR_RE = re.compile(r"<img[\s>]", re.IGNORECASE)
_NUMERIC_RE = re.compile(r"^\[\d")


def _count(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def _context_text(citation: dict) -> str:
    lc = citation.get("local_context", {})
    return (lc.get("before") or "") + " " + (lc.get("after") or "")


def _aggregate() -> dict:
    root = Path(EXTRACTED_DIR)

    papers  = []
    missing = []

    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        ctx_path = folder / CONTEXT_FILENAME
        if not ctx_path.exists():
            missing.append(folder.name)
            continue
        try:
            with ctx_path.open(encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            missing.append(folder.name)
            continue
        papers.append((folder.name, data))

    total_citations       = 0
    single_cit            = 0
    multi_cit             = 0
    numeric_markers       = 0
    authoryear_markers    = 0
    papers_with_abstract  = 0
    papers_with_title     = 0
    papers_with_latex     = 0
    papers_with_failocr   = 0
    total_latex           = 0
    total_failocr         = 0

    cit_per_paper = []
    hist_data     = {}
    top_failocr   = []

    BUCKET = 5

    for folder_name, data in papers:
        cits     = data.get("citations") or []
        gc       = data.get("global_context") or {}
        paper_id = data.get("paper_id") or folder_name
        title    = (gc.get("title") or "").strip()
        abstract = (gc.get("abstract") or "").strip()
        count    = data.get("total_citations") or len(cits)
        total_citations += count

        p_latex = p_failocr = 0

        for c in cits:
            if c.get("citation_type") == "single":
                single_cit += 1
            else:
                multi_cit += 1

            m = c.get("citation_marker") or ""
            if _NUMERIC_RE.match(m):
                numeric_markers += 1
            else:
                authoryear_markers += 1

            ctx = _context_text(c)
            lx  = _count(_LATEX_RE, ctx)
            fi  = _count(_FAILOCR_RE, ctx)
            p_latex   += lx
            p_failocr += fi

            if fi > 0 and len(top_failocr) < 200:
                top_failocr.append({"paper": paper_id, "marker": m, "count": fi})

        total_latex   += p_latex
        total_failocr += p_failocr
        if p_latex   > 0: papers_with_latex   += 1
        if p_failocr > 0: papers_with_failocr += 1
        if abstract:      papers_with_abstract += 1
        if title:         papers_with_title    += 1

        bucket = (count // BUCKET) * BUCKET
        hist_data[bucket] = hist_data.get(bucket, 0) + 1

        cit_per_paper.append({
            "id":           paper_id,
            "title":        title or "—",
            "count":        count,
            "latex":        p_latex,
            "failocr":      p_failocr,
            "has_abstract": bool(abstract),
        })

    n = len(papers)
    hist_sorted = sorted(hist_data.items())

    return {
        "papers_total":         n,
        "papers_missing":       len(missing),
        "missing_folders":      missing[:50],
        "total_citations":      total_citations,
        "avg_citations":        round(total_citations / n, 2) if n else 0,
        "single_citations":     single_cit,
        "multiple_citations":   multi_cit,
        "numeric_markers":      numeric_markers,
        "authoryear_markers":   authoryear_markers,
        "papers_with_abstract": papers_with_abstract,
        "papers_with_title":    papers_with_title,
        "papers_with_latex":    papers_with_latex,
        "papers_without_latex": n - papers_with_latex,
        "total_latex":          total_latex,
        "papers_with_failocr":  papers_with_failocr,
        "total_failocr":        total_failocr,
        "histogram": {
            "labels": [f"{k}–{k+BUCKET-1}" for k, _ in hist_sorted],
            "values": [v for _, v in hist_sorted],
        },
        "top20_by_citations": sorted(cit_per_paper, key=lambda p: p["count"], reverse=True)[:20],
        "top20_by_latex":     sorted(cit_per_paper, key=lambda p: p["latex"],  reverse=True)[:20],
        "top_failocr_items":  top_failocr,
    }


@bp.route("/stats/data", methods=["GET"])
def stats_data():
    return jsonify(_aggregate())


@bp.route("/stats", methods=["GET"])
def stats_dashboard():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_dashboard.html")
    return send_file(html_path, mimetype="text/html")
