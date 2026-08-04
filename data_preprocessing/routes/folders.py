import os

from flask import Blueprint, jsonify

from config import EXTRACTED_DIR

bp = Blueprint("folders", __name__)


@bp.route("/folders", methods=["GET"])
def list_folders():
    try:
        folders = sorted(
            entry.name
            for entry in os.scandir(EXTRACTED_DIR)
            if entry.is_dir()
        )
        return jsonify({"folders": folders, "count": len(folders)}), 200
    except FileNotFoundError:
        return jsonify({"error": f"Directory not found: {EXTRACTED_DIR}"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":        "ok",
        "extracted_dir": EXTRACTED_DIR,
        "dir_exists":    os.path.isdir(EXTRACTED_DIR),
    }), 200
