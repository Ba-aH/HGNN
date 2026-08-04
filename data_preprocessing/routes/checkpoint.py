import json
import os

from flask import Blueprint, jsonify, request

from config import CHECKPOINT_FILE

bp = Blueprint("checkpoint", __name__)


@bp.route("/checkpoint", methods=["GET"])
def get_checkpoint():
    if not os.path.isfile(CHECKPOINT_FILE):
        return jsonify({"completed": []}), 200
    with open(CHECKPOINT_FILE, "r") as f:
        return jsonify(json.load(f)), 200


@bp.route("/checkpoint", methods=["POST"])
def save_checkpoint():
    data   = request.get_json(silent=True)
    folder = data.get("folder") if data else None
    if not folder:
        return jsonify({"error": "Missing folder"}), 400

    completed = []
    if os.path.isfile(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            completed = json.load(f).get("completed", [])

    if folder not in completed:
        completed.append(folder)

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed": completed}, f)

    return jsonify({"completed": completed}), 200


@bp.route("/checkpoint", methods=["DELETE"])
def reset_checkpoint():
    """Wipe progress so the full pipeline can be re-run from scratch."""
    if os.path.isfile(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
    return jsonify({"status": "checkpoint cleared"}), 200
