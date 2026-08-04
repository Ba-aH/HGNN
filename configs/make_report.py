#!/usr/bin/env python3
"""
make_report.py
----------------
Builds ONE self-contained HTML report covering ALL training runs found
under a root folder (e.g. configs/P+PP), by recursively scanning for
directories that contain the triple:

    config.json
    history.json
    test_curve.json

Typical layout this is designed for:

    P+PP/
      MLP_500/
        exp_PP_MLP_hidden500(run1)/  <- a "run" dir (has the 3 json files)
        exp_PP_MLP_hidden500(run2)/
      MLP_768/
        exp_PP_MLP_hidden768(run1)/
        exp_PP_MLP_hidden768(run2)/
      MLP_900/
        (no runs yet -> skipped)
      no_MLP/
        exp_PP_noMLP(run1)/
        exp_PP_noMLP(run2)/

Each run gets its own plot grid (loss + MRR/Recall@k/nDCG@10, val vs test,
best-epoch marker) and a redesigned, readable config panel. A summary table
at the top lists every run with its best val/test MRR so you can compare at
a glance, and clicking a row jumps to that run's full section.

Usage:
    python make_report.py --root . --out report.html  

    # to also skip folders (e.g. archived experiments):
    python make_report.py --root "configs/P+PP" --out report.html --skip archive,old
"""

import argparse
import base64
import io
import json
import os
import re
from html import escape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Recall@20", "nDCG@10"]
REQUIRED_FILES = ("config.json", "history.json", "test_curve.json")


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------

def find_run_dirs(root, skip_names=None):
    """Recursively find every directory under root containing all of
    REQUIRED_FILES. Returns sorted list of absolute paths."""
    skip_names = skip_names or set()
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip_names]
        if all(f in filenames for f in REQUIRED_FILES):
            runs.append(dirpath)
    return sorted(runs)


def load_json(path):
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# plotting (unchanged logic from the single-run version)
# --------------------------------------------------------------------------

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def series(log, key):
    pts = [(e["epoch"], e[key]) for e in log if key in e]
    return zip(*pts) if pts else ([], [])


def plot_single_run(exp_name, config, train_hist, test_hist, best_epoch):
    n = len(METRICS) + 1
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows))
    axes = axes.flatten()

    ax = axes[0]
    epochs = [e["epoch"] for e in train_hist]
    losses = [e.get("train_loss") for e in train_hist]
    ax.plot(epochs, losses, color="tab:red", label="train_loss")
    ax.set_title("Train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend()

    for i, key in enumerate(METRICS, start=1):
        ax = axes[i]
        vx, vy = series(train_hist, key)
        tx, ty = series(test_hist, key)
        if vx:
            ax.plot(vx, vy, color="tab:blue", linewidth=1.6, label="val")
        if tx:
            ax.plot(tx, ty, color="tab:orange", marker="o", linewidth=1.6, label="test")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=1,
                       label=f"best@{best_epoch}")
        ax.set_title(key)
        ax.set_xlabel("epoch"); ax.set_ylabel(key)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    cfg = config or {}
    subtitle = (f"feat_keys={cfg.get('feat_keys')} | use_mlp={cfg.get('use_mlp')} | "
                f"embed_dim={cfg.get('embed_dim')} | "
                f"batch_size={cfg.get('batch_size')} | act={cfg.get('act')} | "
                f"dropout={cfg.get('dropout')} | input_drop={cfg.get('input_drop')}")
    fig.suptitle(f"{exp_name}\n{subtitle}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig_to_base64(fig)


# --------------------------------------------------------------------------
# config table (redesigned)
# --------------------------------------------------------------------------

# Loose grouping so related params sit together instead of dumping json
# order. Anything not matched falls into "Other".
GROUPS = [
    ("Architecture", ["embed_dim", "hidden", "use_mlp", "act", "dropout",
                       "input_drop", "num_layers", "feat_keys", "metapaths",
                       "no_MLP"]),
    ("Training", ["batch_size", "lr", "lr_scibert", "epochs", "temperature",
                   "weight_decay", "optimizer", "scheduler", "patience",
                   "grad_clip", "warmup"]),
    ("Data / Split", ["split_path", "split_uris", "train_path", "val_path",
                       "test_path", "corpus_path", "candidate_pool"]),
]


def humanize_key(k):
    return k.replace("_", " ")


def format_value(v):
    if isinstance(v, bool):
        return ("true" if v else "false"), "bool"
    if isinstance(v, (int, float)):
        return json.dumps(v), "num"
    if v is None:
        return "null", "null"
    if isinstance(v, (list, dict)):
        return json.dumps(v), "code"
    return str(v), "str"


def build_config_table(config):
    if not config:
        return "<p class='empty'>No config.json found.</p>"

    notes = {k: v for k, v in config.items() if k.startswith("note")}
    params = {k: v for k, v in config.items() if not k.startswith("note")}

    grouped = {name: {} for name, _ in GROUPS}
    grouped["Other"] = {}
    matched_keys = set()
    for name, keys in GROUPS:
        for k in keys:
            if k in params:
                grouped[name][k] = params[k]
                matched_keys.add(k)
    for k, v in params.items():
        if k not in matched_keys:
            grouped["Other"][k] = v

    sections = []
    for name, kv in list(grouped.items()):
        if not kv:
            continue
        rows = []
        for k, v in kv.items():
            text, kind = format_value(v)
            rows.append(
                f"<div class='cfg-row'>"
                f"<div class='cfg-key'>{escape(humanize_key(k))}</div>"
                f"<div class='cfg-val cfg-val--{kind}'>{escape(text)}</div>"
                f"</div>"
            )
        sections.append(
            f"<div class='cfg-group'>"
            f"<div class='cfg-group-title'>{escape(name)}</div>"
            f"<div class='cfg-rows'>{''.join(rows)}</div>"
            f"</div>"
        )

    notes_html = ""
    if notes:
        note_items = "".join(f"<li>{escape(str(v))}</li>" for v in notes.values())
        notes_html = (f"<details class='cfg-notes'><summary>Notes / comments "
                       f"({len(notes)})</summary><ul>{note_items}</ul></details>")

    return f"<div class='cfg-grid'>{''.join(sections)}</div>{notes_html}"


# --------------------------------------------------------------------------
# per-run data assembly
# --------------------------------------------------------------------------

def slugify(text):
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "run"


def build_run(run_dir, root):
    config = load_json(os.path.join(run_dir, "config.json"))
    history = load_json(os.path.join(run_dir, "history.json"))
    test_curve = load_json(os.path.join(run_dir, "test_curve.json"))

    rel = os.path.relpath(run_dir, root).replace("\\", "/")
    exp_name = (
        (history and history.get("experiment_name"))
        or (config and config.get("experiment_name"))
        or (test_curve and test_curve.get("experiment_name"))
        or rel
    )

    best_mrr = history.get("best_mrr") if history else None
    best_epoch = history.get("best_epoch") if history else None
    total_time_h = history.get("total_training_time_h") if history else None
    epochs_run = history.get("epochs_run") if history else None

    train_hist = history.get("history", []) if history else []
    test_hist = (test_curve.get("history", []) if test_curve
                 else (history.get("test_history", []) if history else []))

    best_test = None
    if test_hist:
        best_test = max(test_hist, key=lambda r: r.get("MRR", 0))

    test_every = test_curve.get("test_eval_every", 5) if test_curve else 5

    best_val_recall10 = None
    if best_epoch is not None:
        for e in train_hist:
            if e.get("epoch") == best_epoch and "Recall@10" in e:
                best_val_recall10 = e["Recall@10"]
                break

    best_test_recall10 = best_test.get("Recall@10") if best_test else None

    print(f"Rendering plot for {rel} ...")
    plot_b64 = plot_single_run(exp_name, config, train_hist, test_hist, best_epoch)

    return {
        "dir": run_dir,
        "rel": rel,
        "slug": slugify(rel),
        "exp_name": exp_name,
        "group": rel.split("/")[0] if "/" in rel else rel,
        "config": config,
        "best_mrr": best_mrr,
        "best_epoch": best_epoch,
        "total_time_h": total_time_h,
        "epochs_run": epochs_run,
        "best_val_recall10": best_val_recall10,
        "best_test_mrr": best_test["MRR"] if best_test else None,
        "best_test_epoch": best_test["epoch"] if best_test else None,
        "best_test_recall10": best_test_recall10,
        "test_every": test_every,
        "plot_b64": plot_b64,
        "config_html": build_config_table(config),
    }


# --------------------------------------------------------------------------
# HTML assembly
# --------------------------------------------------------------------------

def fmt(v, digits=4):
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "—"


def build_summary_table(runs):
    rows = []
    for r in runs:
        rows.append(f"""
        <tr onclick="location.hash='#{r['slug']}'">
          <td class="s-group">{escape(r['group'])}</td>
          <td class="s-name">{escape(r['exp_name'])}</td>
          <td class="s-num">{fmt(r['best_mrr'])}</td>
          <td class="s-num s-muted">ep {r['best_epoch'] if r['best_epoch'] is not None else '—'}</td>
          <td class="s-num">{fmt(r['best_val_recall10'])}</td>
          <td class="s-num">{fmt(r['best_test_mrr'])}</td>
          <td class="s-num s-muted">ep {r['best_test_epoch'] if r['best_test_epoch'] is not None else '—'}</td>
          <td class="s-num">{fmt(r['best_test_recall10'])}</td>
          <td class="s-num">{fmt(r['total_time_h'], 2)}h</td>
        </tr>""")
    return f"""
    <table class="summary-table">
      <thead>
        <tr>
          <th>Group</th><th>Run</th>
          <th>Best Val MRR</th><th></th><th>Val Recall@10</th>
          <th>Best Test MRR</th><th></th><th>Test Recall@10</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


def build_run_section(r):
    cards = ""
    if r["best_mrr"] is not None:
        cards += f"""
        <div class="card">
          <div class="card-label">Best Val MRR</div>
          <div class="card-value">{fmt(r['best_mrr'])}</div>
          <div class="card-sub">epoch {r['best_epoch']}</div>
        </div>"""
    if r["best_val_recall10"] is not None:
        cards += f"""
        <div class="card">
          <div class="card-label">Val Recall@10</div>
          <div class="card-value">{fmt(r['best_val_recall10'])}</div>
          <div class="card-sub">at best-MRR epoch {r['best_epoch']}</div>
        </div>"""
    if r["total_time_h"] is not None:
        cards += f"""
        <div class="card">
          <div class="card-label">Training Time</div>
          <div class="card-value">{fmt(r['total_time_h'], 2)}h</div>
          <div class="card-sub">{r['epochs_run']} epochs run</div>
        </div>"""
    if r["best_test_mrr"] is not None:
        cards += f"""
        <div class="card">
          <div class="card-label">Best Test MRR</div>
          <div class="card-value">{fmt(r['best_test_mrr'])}</div>
          <div class="card-sub">epoch {r['best_test_epoch']}</div>
        </div>"""
    if r["best_test_recall10"] is not None:
        cards += f"""
        <div class="card">
          <div class="card-label">Test Recall@10</div>
          <div class="card-value">{fmt(r['best_test_recall10'])}</div>
          <div class="card-sub">at best-test-MRR epoch {r['best_test_epoch']}</div>
        </div>"""

    return f"""
    <section class="run" id="{r['slug']}">
      <div class="run-header">
        <h2>{escape(r['exp_name'])}</h2>
        <div class="run-path">{escape(r['rel'])}</div>
      </div>

      <div class="cards">{cards}</div>

      <div class="grid">
        <div class="panel full">
          <img src="data:image/png;base64,{r['plot_b64']}">
        </div>
        <div class="panel full">
          <h3>Configuration</h3>
          {r['config_html']}
        </div>
      </div>
    </section>
    """


def build_nav(runs):
    by_group = {}
    for r in runs:
        by_group.setdefault(r["group"], []).append(r)
    items = []
    for group, group_runs in by_group.items():
        links = "".join(
            f"<li><a href='#{r['slug']}'>{escape(r['exp_name'])}</a></li>"
            for r in group_runs
        )
        items.append(f"<div class='nav-group'><div class='nav-group-title'>{escape(group)}</div><ul>{links}</ul></div>")
    return "".join(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Root folder to scan recursively (e.g. configs/P+PP)")
    ap.add_argument("--out", default="report.html")
    ap.add_argument("--skip", default="", help="Comma-separated dir names to skip (e.g. archive,old)")
    args = ap.parse_args()

    skip_names = {s.strip() for s in args.skip.split(",") if s.strip()}
    root = os.path.abspath(args.root)

    run_dirs = find_run_dirs(root, skip_names)
    if not run_dirs:
        print(f"No runs found under {root} (looking for {REQUIRED_FILES})")
        return

    print(f"Found {len(run_dirs)} run(s) under {root}:")
    for d in run_dirs:
        print(f"  - {os.path.relpath(d, root)}")

    runs = [build_run(d, root) for d in run_dirs]
    # sort by best val MRR desc (None last)
    runs.sort(key=lambda r: (r["best_mrr"] is None, -(r["best_mrr"] or 0)))

    summary_html = build_summary_table(runs)
    nav_html = build_nav(runs)
    sections_html = "".join(build_run_section(r) for r in runs)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Training Report — {escape(os.path.basename(root))}</title>
<style>
  :root {{
    --bg: #0f1115;
    --panel: #171a21;
    --panel-alt: #1c2029;
    --panel-border: #2a2e38;
    --text: #e6e8ee;
    --muted: #9aa1ad;
    --accent: #6ea8fe;
    --accent-soft: rgba(110,168,254,0.12);
    --good: #7ee787;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
  }}

  /* ---------- sidebar nav ---------- */
  nav.sidebar {{
    position: sticky;
    top: 0;
    height: 100vh;
    width: 240px;
    flex: 0 0 240px;
    overflow-y: auto;
    padding: 24px 16px;
    border-right: 1px solid var(--panel-border);
    background: var(--panel);
  }}
  nav.sidebar h3 {{
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--muted);
    margin: 0 0 14px 4px;
  }}
  .nav-group {{ margin-bottom: 16px; }}
  .nav-group-title {{
    font-size: 12px;
    font-weight: 600;
    color: var(--accent);
    margin: 0 0 4px 4px;
  }}
  .nav-group ul {{ list-style: none; margin: 0; padding: 0; }}
  .nav-group li a {{
    display: block;
    padding: 5px 8px;
    border-radius: 6px;
    color: var(--text);
    text-decoration: none;
    font-size: 13px;
    opacity: 0.85;
  }}
  .nav-group li a:hover {{ background: var(--accent-soft); opacity: 1; }}

  /* ---------- main ---------- */
  main {{ flex: 1; padding: 32px 40px; max-width: 1200px; }}
  h1 {{ font-size: 24px; margin: 0 0 4px 0; }}
  .subtitle {{ color: var(--muted); margin-bottom: 24px; font-size: 14px; }}

  /* ---------- summary table ---------- */
  .summary-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13.5px;
    margin-bottom: 40px;
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .summary-table th {{
    text-align: left;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .04em;
    color: var(--muted);
    padding: 10px 14px;
    border-bottom: 1px solid var(--panel-border);
  }}
  .summary-table td {{
    padding: 9px 14px;
    border-bottom: 1px solid var(--panel-border);
  }}
  .summary-table tbody tr {{ cursor: pointer; }}
  .summary-table tbody tr:hover {{ background: var(--accent-soft); }}
  .summary-table tbody tr:last-child td {{ border-bottom: none; }}
  .s-group {{ color: var(--muted); font-size: 12px; }}
  .s-name {{ font-weight: 600; }}
  .s-num {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--good); }}
  .s-muted {{ color: var(--muted); font-size: 12px; }}

  /* ---------- run sections ---------- */
  .run {{ margin-bottom: 56px; padding-top: 12px; scroll-margin-top: 16px; }}
  .run-header {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    border-bottom: 1px solid var(--panel-border);
    padding-bottom: 10px;
    margin-bottom: 18px;
  }}
  .run-header h2 {{ font-size: 19px; margin: 0; }}
  .run-path {{ color: var(--muted); font-size: 12.5px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}

  .cards {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 22px; }}
  .card {{
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 16px 20px;
    min-width: 160px;
  }}
  .card-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
  .card-value {{ font-size: 26px; font-weight: 600; margin-top: 4px; }}
  .card-sub {{ color: var(--muted); font-size: 12px; margin-top: 2px; }}

  .grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; }}
  .panel {{
    background: var(--panel);
    border: 1px solid var(--panel-border);
    border-radius: 10px;
    padding: 20px;
  }}
  .panel h3 {{ font-size: 14px; margin: 0 0 14px 0; color: var(--text); }}
  .full {{ grid-column: 1 / -1; }}
  .panel img {{ width: 100%; border-radius: 6px; background: #fff; }}

  /* ---------- redesigned config table ---------- */
  .cfg-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 18px;
  }}
  .cfg-group {{
    background: var(--panel-alt);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 12px 14px;
  }}
  .cfg-group-title {{
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 8px;
  }}
  .cfg-rows {{ display: flex; flex-direction: column; }}
  .cfg-row {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 10px;
    padding: 5px 0;
    border-bottom: 1px solid var(--panel-border);
    font-size: 13px;
  }}
  .cfg-row:last-child {{ border-bottom: none; }}
  .cfg-key {{ color: var(--muted); text-transform: capitalize; }}
  .cfg-val {{
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    text-align: right;
    word-break: break-word;
  }}
  .cfg-val--bool {{ color: var(--accent); }}
  .cfg-val--num {{ color: var(--good); }}
  .cfg-val--null {{ color: var(--muted); font-style: italic; }}
  .cfg-val--code {{ color: #f0b355; font-size: 12px; }}

  .cfg-notes {{ margin-top: 14px; }}
  .cfg-notes summary {{ cursor: pointer; color: var(--accent); font-size: 13px; }}
  .cfg-notes ul {{ color: var(--muted); font-size: 12.5px; padding-left: 18px; }}
  .empty {{ color: var(--muted); font-style: italic; }}
</style>
</head>
<body>

<nav class="sidebar">
  <h3>Runs</h3>
  {nav_html}
</nav>

<main>
  <h1>Training Report</h1>
  <div class="subtitle">{escape(os.path.basename(root))} &middot; {len(runs)} run(s) &middot; sorted by best val MRR</div>

  {summary_html}

  {sections_html}
</main>

</body>
</html>
"""

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nWrote {args.out} ({len(runs)} runs)")


if __name__ == "__main__":
    main()
