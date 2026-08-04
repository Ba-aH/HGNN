import os
import json
import logging
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path

from flask import Blueprint, jsonify

from config import EXTRACTED_DIR
from services.context import extract_context_for_folder

logger = logging.getLogger(__name__)

bp = Blueprint("context", __name__)

CONTEXT_FILENAME  = "context.json"
_FAILURES_PATH    = Path(EXTRACTED_DIR) / "context_failures.json"
_failures_lock    = threading.Lock()


def _log_context_failure(folder: str, error: str, tb: str) -> None:
    """Append a failure record to EXTRACTED_DIR/context_failures.json atomically."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "folder":    folder,
        "error":     error,
        "traceback": tb,
    }
    with _failures_lock:
        if _FAILURES_PATH.exists():
            try:
                with _FAILURES_PATH.open(encoding="utf-8") as f:
                    failures = json.load(f)
            except (json.JSONDecodeError, OSError):
                failures = []
        else:
            failures = []

        failures.append(record)

        tmp = str(_FAILURES_PATH) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(failures, f, indent=2, ensure_ascii=False)
        Path(tmp).replace(_FAILURES_PATH)

# Guards concurrent extractions for the same folder
_folder_locks: dict[str, threading.Lock] = {}
_locks_mutex = threading.Lock()

def _get_folder_lock(folder: str) -> threading.Lock:
    with _locks_mutex:
        if folder not in _folder_locks:
            _folder_locks[folder] = threading.Lock()
        return _folder_locks[folder]


@bp.route("/extract_context/<path:folder>", methods=["GET"])
def check_context(folder):
    context_path = os.path.join(EXTRACTED_DIR, folder, CONTEXT_FILENAME)
    return jsonify({
        "folder": folder,
        "exists": os.path.isfile(context_path),
        "path":   context_path,
    }), 200


@bp.route("/extract_context/<path:folder>", methods=["POST"])
def extract_context(folder):
    lock = _get_folder_lock(folder)

    if not lock.acquire(blocking=False):
        return jsonify({
            "folder":  folder,
            "status":  "already_processing",
            "message": "Extraction already in progress for this folder",
        }), 409

    def run():
        try:
            result = extract_context_for_folder(folder)

            context_path = os.path.join(EXTRACTED_DIR, folder, CONTEXT_FILENAME)
            tmp_path     = context_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            os.replace(tmp_path, context_path)
            logger.info(f"[{folder}] context.json written ({result.get('total_citations', 0)} citations)")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{folder}] extraction failed: {e}\n{tb}")
            _log_context_failure(folder, str(e), tb)
        finally:
            lock.release()

    threading.Thread(target=run, daemon=True).start()

    return jsonify({
        "folder":  folder,
        "status":  "processing",
        "message": "Extraction started in background",
    }), 202