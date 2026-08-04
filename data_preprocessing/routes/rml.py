import os
import shutil
import subprocess
import logging

from flask import Blueprint, jsonify, request

from config import EXTRACTED_DIR, RML_MAPPING_PATH

bp = Blueprint("rml", __name__)


def _count_triples(ttl_path: str) -> int:
    """Count triples using rdflib for accuracy."""
    try:
        from rdflib import Graph
        g = Graph()
        g.parse(ttl_path, format="turtle")
        return len(g)
    except Exception:
        # Fallback: heuristic count if rdflib not available
        if not os.path.isfile(ttl_path):
            return 0
        count = 0
        with open(ttl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("@") and stripped.endswith("."):
                    count += 1
        return count


# ── POST /rml/convert ─────────────────────────────────────────────────────────
@bp.route("/rml/convert", methods=["POST"])
def rml_convert():
    import logging

    body = request.get_json(silent=True) or {}

    json_path = body.get("json_path", "").strip()
    folder_name = body.get("folder", "").strip()

    logging.warning(f"[rml/convert] folder='{folder_name}'")
    logging.warning(f"[rml/convert] references exists: {os.path.isfile(os.path.join(EXTRACTED_DIR, folder_name, 'references.json'))}")
    logging.warning(f"[rml/convert] context_flat exists: {os.path.isfile(os.path.join(EXTRACTED_DIR, folder_name, 'context_flat.json'))}")

    if not json_path and folder_name:
        json_path = os.path.join(EXTRACTED_DIR, folder_name, "references.json")

    if not json_path:
        return jsonify({"success": False, "skipped": True,
                        "reason": "No folder or json_path provided"}), 200

    if not os.path.isfile(json_path):
        logging.warning(f"[rml/convert] SKIPPED: references.json not found at {json_path}")
        return jsonify({"success": False, "skipped": True,
                        "folder": folder_name,
                        "reason": f"references.json not found: {json_path}"}), 200

    paper_folder = os.path.dirname(json_path)

    context_flat_path = os.path.join(paper_folder, "context_flat.json")
    if not os.path.isfile(context_flat_path):
        logging.warning(f"[rml/convert] SKIPPED: context_flat.json not found at {context_flat_path}")
        return jsonify({"success": False, "skipped": True,
                        "folder": folder_name,
                        "reason": f"context_flat.json not found in: {paper_folder} — "
                                   "run flatten_citations_batch.py first"}), 200

    if not os.path.isfile(RML_MAPPING_PATH):
        return jsonify({"success": False,
                        "error": f"mapping.rml.ttl not found at: {RML_MAPPING_PATH}"}), 500

    shutil.copy2(RML_MAPPING_PATH, os.path.join(paper_folder, "mapping.rml.ttl"))

    docker_folder = paper_folder.replace("\\", "/")

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{docker_folder}:/data",
        "rmlio/rmlmapper-java",
        "-m", "/data/mapping.rml.ttl",
        "-o", "/data/paper-kg.ttl",
        "-s", "turtle",
        "--duplicates",
    ]

    logging.warning(f"[rml/convert] running docker cmd: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "RMLMapper timed out after 120s"}), 500
    except FileNotFoundError:
        return jsonify({"success": False,
                        "error": "docker command not found — is Docker Desktop running?"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

    logging.warning(f"[rml/convert] docker returncode: {result.returncode}")
    if result.returncode != 0:
        logging.warning(f"[rml/convert] stderr: {result.stderr[-500:]}")
        return jsonify({"success": False, "error": "RMLMapper exited with errors",
                        "stderr": result.stderr[-2000:]}), 500

    ttl_path = os.path.join(paper_folder, "paper-kg.ttl")
    logging.warning(f"[rml/convert] SUCCESS: {ttl_path}")
    return jsonify({
        "success":  True,
        "folder":   os.path.basename(paper_folder),
        "ttl_path": ttl_path,
        "triples":  _count_triples(ttl_path),
    }), 200

# ── POST /rml/merge ───────────────────────────────────────────────────────────
@bp.route("/rml/merge", methods=["POST"])
def rml_merge():
    ttl_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(EXTRACTED_DIR)
        for fname in files
        if fname == "paper-kg.ttl"
    ]

    if not ttl_files:
        return jsonify({"success": False, "error": "No paper-kg.ttl files found"}), 404

    # FIX #7: Prefix block now matches the ontologies actually used in mapping.rml.ttl
    prefix_block = (
        "@prefix rr:      <http://www.w3.org/ns/r2rml#> .\n"
        "@prefix rml:     <http://semweb.mmlab.be/ns/rml#> .\n"
        "@prefix ql:      <http://semweb.mmlab.be/ns/ql#> .\n"
        "@prefix dcterms: <http://purl.org/dc/terms/> .\n"
        "@prefix xsd:     <http://www.w3.org/2001/XMLSchema#> .\n"
        "@prefix cito:    <http://purl.org/spar/cito/> .\n"
        "@prefix c4o:     <http://purl.org/spar/c4o/> .\n"
        "@prefix rdfs:    <http://www.w3.org/2000/01/rdf-schema#> .\n"
        "@prefix foaf:    <http://xmlns.com/foaf/0.1/> .\n"
        "@prefix bibo:    <http://purl.org/ontology/bibo/> .\n"
        "@prefix fabio:   <http://purl.org/spar/fabio/> .\n"
        "@prefix pro:     <http://purl.org/spar/pro/> .\n"
        "@prefix citekg:  <https://citekg.org/ontology/> .\n"
        "@prefix :        <https://citekg.org/resource/> .\n\n"
    )

    output_path = os.path.join(EXTRACTED_DIR, "graph.ttl")
    written = 0

    with open(output_path, "w", encoding="utf-8") as out:
        out.write(prefix_block)
        for ttl_path in sorted(ttl_files):
            with open(ttl_path, "r", encoding="utf-8", errors="ignore") as src:
                for line in src:
                    if not line.startswith("@prefix"):
                        out.write(line)
            written += 1

    return jsonify({"success": True, "merged_files": written, "output_path": output_path}), 200