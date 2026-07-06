#!/usr/bin/env python3
"""
generate_dashboard.py
----------------------
Walks a root folder (e.g. your `configs/` tree), auto-discovers every run
directory that has a config.json + history.json (optionally test_curve.json),
and builds ONE self-contained HTML file: a sidebar to browse each run's
matplotlib dashboard individually, plus an "All runs" comparison page.

No server needed — the output is a single static .html file with all plots
embedded as base64 PNGs. Just double-click it to open in a browser.

Usage:
    python generate_dashboard.py --root "C:\\...\\configs" --out dashboard.html

A run directory is recognized as any folder containing both config.json and
history.json — so it auto-finds exp_PP_MLP_hidden500(run1),
exp_PP_MLP_hidden500(run2), exp_PP_noMLP(run1), etc. regardless of depth or
naming, and skips incomplete folders like .ipynb_checkpoints.
"""

import argparse
import base64
import io
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRICS = ["MRR", "Recall@1", "Recall@5", "Recall@10", "Recall@20", "nDCG@10"]


# ---------------------------------------------------------------------------
# Discovery + loading
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.config = json.loads((run_dir / "config.json").read_text())
        hist = json.loads((run_dir / "history.json").read_text())
        self.epoch_log = hist.get("history", [])
        self.best_mrr = hist.get("best_mrr")
        self.best_epoch = hist.get("best_epoch")

        tc_path = run_dir / "test_curve.json"
        if tc_path.exists():
            self.test_log = json.loads(tc_path.read_text()).get("history", [])
        else:
            self.test_log = hist.get("test_history", [])

        self.folder_name = run_dir.name
        self.group = run_dir.parent.name
        self.label = self.config.get("experiment_name") or self.folder_name

    def epochs(self):
        return [e["epoch"] for e in self.epoch_log]

    def train_losses(self):
        return [e.get("train_loss") for e in self.epoch_log]

    def val_series(self, key):
        pts = [(e["epoch"], e[key]) for e in self.epoch_log if key in e]
        return zip(*pts) if pts else ([], [])

    def test_series(self, key):
        pts = [(e["epoch"], e[key]) for e in self.test_log if key in e]
        return zip(*pts) if pts else ([], [])


def discover_runs(root: Path):
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Don't descend into obvious junk dirs
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        fset = set(filenames)
        if "config.json" in fset and "history.json" in fset:
            try:
                runs.append(Run(Path(dirpath)))
            except Exception as e:
                print(f"[skip] {dirpath}: {e}")
    runs.sort(key=lambda r: (r.group, r.folder_name))
    return runs


# ---------------------------------------------------------------------------
# Plotting (same layout as plot_lcr_runs.py's dashboard, rendered to base64)
# ---------------------------------------------------------------------------

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def plot_single_run(run: Run):
    n = len(METRICS) + 1
    ncols = 3
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows))
    axes = axes.flatten()

    ax = axes[0]
    ax.plot(run.epochs(), run.train_losses(), color="tab:red", label="train_loss")
    ax.set_title("Train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss")
    ax.grid(alpha=0.3); ax.legend()

    for i, key in enumerate(METRICS, start=1):
        ax = axes[i]
        vx, vy = run.val_series(key)
        tx, ty = run.test_series(key)
        if vx:
            ax.plot(vx, vy, color="tab:blue", linewidth=1.6, label="val")
        if tx:
            ax.plot(tx, ty, color="tab:orange", marker="o", linewidth=1.6, label="test")
        if run.best_epoch is not None:
            ax.axvline(run.best_epoch, color="gray", linestyle="--", linewidth=1,
                       label=f"best@{run.best_epoch}")
        ax.set_title(key)
        ax.set_xlabel("epoch"); ax.set_ylabel(key)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    cfg = run.config
    subtitle = (f"feat_keys={cfg.get('feat_keys')} | use_mlp={cfg.get('use_mlp')} | "
                f"hidden={cfg.get('hidden')} | embed_dim={cfg.get('embed_dim')} | "
                f"batch_size={cfg.get('batch_size')} | act={cfg.get('act')} | "
                f"dropout={cfg.get('dropout')} | input_drop={cfg.get('input_drop')}")
    fig.suptitle(f"{run.label}\n{subtitle}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig_to_base64(fig)


def plot_compare_all(runs):
    ncols = 3
    nrows = -(-len(METRICS) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows))
    axes = axes.flatten()
    colors = plt.cm.tab10.colors

    for i, key in enumerate(METRICS):
        ax = axes[i]
        for c, run in zip(colors * 3, runs):
            vx, vy = run.val_series(key)
            if vx:
                ax.plot(vx, vy, color=c, linestyle="-", linewidth=1.6, label=f"{run.label} (val)")
            tx, ty = run.test_series(key)
            if tx:
                ax.plot(tx, ty, color=c, linestyle="--", marker="o", linewidth=1.6,
                       markersize=4, label=f"{run.label} (test)")
        ax.set_title(key)
        ax.set_xlabel("epoch"); ax.set_ylabel(key)
        ax.grid(alpha=0.3)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    fig.legend(handles, labels, loc="lower center",
              ncol=min(len(labels), 4) or 1, fontsize=8, bbox_to_anchor=(0.5, -0.03))

    for j in range(len(METRICS), len(axes)):
        axes[j].axis("off")

    fig.suptitle("All runs — val (solid) vs test (dashed)", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return fig_to_base64(fig)


def plot_loss_compare_all(runs):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = plt.cm.tab10.colors
    for c, run in zip(colors * 3, runs):
        ax.plot(run.epochs(), run.train_losses(), color=c, label=run.label)
    ax.set_title("Train loss — all runs")
    ax.set_xlabel("epoch"); ax.set_ylabel("train_loss")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

PAGE_CSS = """
* { box-sizing: border-box; }
body { margin:0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
       background:#f5f5f6; color:#1f1f22; }
#layout { display:flex; min-height:100vh; }
#sidebar { width:280px; background:#fff; border-right:1px solid #e6e6e8; padding:18px;
           overflow-y:auto; position:sticky; top:0; height:100vh; }
#sidebar h2 { font-size:15px; margin:0 0 14px; }
.group-title { font-size:11px; text-transform:uppercase; letter-spacing:.04em;
               color:#8a8a90; margin:16px 0 6px; }
.nav-link { display:block; padding:7px 10px; border-radius:8px; font-size:13px;
            color:#1f1f22; text-decoration:none; margin-bottom:2px; }
.nav-link:hover { background:#f0eefb; }
.nav-link.compare { font-weight:700; color:#6d5bd0; }
.nav-link.hidden-by-filter { display:none; }
#main { flex:1; padding:30px 36px; max-width:1400px; }
.section { margin-bottom:56px; scroll-margin-top:20px; }
.section.hidden-by-filter { display:none; }
.section h1 { font-size:22px; margin:0 0 6px; }
.section .meta { color:#8a8a90; font-size:12.5px; margin-bottom:14px; }
.section img { width:100%; border:1px solid #e6e6e8; border-radius:12px; background:#fff; }
hr { border:none; border-top:1px solid #e6e6e8; margin:40px 0; }
hr.hidden-by-filter { display:none; }

#filter-box { margin-bottom:10px; padding-bottom:14px; border-bottom:1px solid #e6e6e8; }
#filter-box .filter-title { font-size:11px; text-transform:uppercase; letter-spacing:.04em;
                            color:#8a8a90; margin-bottom:8px; }
#filter-box label { display:block; font-size:12px; color:#4a4a50; margin:8px 0 3px; }
#filter-box select { width:100%; padding:6px; border-radius:6px; border:1px solid #d6d6da;
                     font-size:12.5px; background:#fafafa; }
#filter-clear { margin-top:10px; width:100%; padding:6px; border-radius:6px; border:1px solid #d6d6da;
               background:#fff; font-size:12px; cursor:pointer; color:#6d5bd0; }
#filter-clear:hover { background:#f0eefb; }
#filter-status { font-size:11.5px; color:#8a8a90; margin-top:8px; }
"""

# Config keys exposed as dropdown filters in the sidebar. Add more here if
# you want to filter on other hyperparameters (e.g. "hidden", "use_mlp",
# "batch_size", "act", ...) — no other code needs to change.
FILTER_KEYS = ["dropout", "input_drop"]


def build_filter_box(runs):
    options_html = []
    for key in FILTER_KEYS:
        values = sorted({r.config.get(key) for r in runs if r.config.get(key) is not None},
                        key=lambda v: (isinstance(v, str), v))
        opts = "".join(f'<option value="{v}">{v}</option>' for v in values)
        options_html.append(f"""
        <label for="filter-{key}">{key}</label>
        <select id="filter-{key}" data-key="{key}" onchange="applyFilters()">
          <option value="">All</option>
          {opts}
        </select>
        """)
    return f"""
    <div id="filter-box">
      <div class="filter-title">Filters</div>
      {''.join(options_html)}
      <button id="filter-clear" onclick="clearFilters()">Clear filters</button>
      <div id="filter-status"></div>
    </div>
    """


FILTER_JS = """
function applyFilters() {
  const active = {};
  document.querySelectorAll('#filter-box select').forEach(sel => {
    if (sel.value !== "") active[sel.dataset.key] = sel.value;
  });

  let shown = 0, total = 0;
  document.querySelectorAll('.section[data-run="1"]').forEach(sec => {
    total++;
    const match = Object.entries(active).every(([k, v]) => sec.dataset[k] === v);
    sec.classList.toggle('hidden-by-filter', !match);
    const link = document.querySelector(`.nav-link[href="#${sec.id}"]`);
    if (link) link.classList.toggle('hidden-by-filter', !match);
    const rule = document.getElementById('hr-' + sec.id);
    if (rule) rule.classList.toggle('hidden-by-filter', !match);
    if (match) shown++;
  });

  document.getElementById('filter-status').textContent =
    Object.keys(active).length ? `Showing ${shown} of ${total} runs` : "";
}

function clearFilters() {
  document.querySelectorAll('#filter-box select').forEach(sel => sel.value = "");
  applyFilters();
}
"""


def build_html(runs, out_path: Path):
    groups = {}
    for r in runs:
        groups.setdefault(r.group, []).append(r)

    nav_html = []
    nav_html.append('<a class="nav-link compare" href="#compare-all">All runs comparison</a>')
    for group, grs in groups.items():
        nav_html.append(f'<div class="group-title">{group}</div>')
        for r in grs:
            anchor = anchor_id(r)
            nav_html.append(f'<a class="nav-link" href="#{anchor}">{r.folder_name}</a>')

    sections = []

    print("Rendering all-runs comparison ...")
    cmp_metrics_b64 = plot_compare_all(runs)
    cmp_loss_b64 = plot_loss_compare_all(runs)
    sections.append(f"""
    <div class="section" id="compare-all">
      <h1>All runs — comparison</h1>
      <div class="meta">{len(runs)} runs · val (solid) vs test (dashed)</div>
      <img src="data:image/png;base64,{cmp_metrics_b64}">
      <br><br>
      <img src="data:image/png;base64,{cmp_loss_b64}">
    </div>
    <hr>
    """)

    for r in runs:
        print(f"Rendering {r.label} ...")
        img_b64 = plot_single_run(r)
        cfg = r.config
        anchor = anchor_id(r)
        data_attrs = " ".join(f'data-{k}="{cfg.get(k)}"' for k in FILTER_KEYS)
        sections.append(f"""
        <div class="section" id="{anchor}" data-run="1" {data_attrs}>
          <h1>{r.label}</h1>
          <div class="meta">
            {r.group} / {r.folder_name} &nbsp;|&nbsp;
            best epoch #{r.best_epoch} &nbsp;|&nbsp; best MRR {fmt(r.best_mrr)} &nbsp;|&nbsp;
            dropout {cfg.get('dropout')} &nbsp;|&nbsp; input_drop {cfg.get('input_drop')} &nbsp;|&nbsp;
            {len(r.epoch_log)} epochs &nbsp;|&nbsp; {len(r.test_log)} test checkpoints
          </div>
          <img src="data:image/png;base64,{img_b64}">
        </div>
        <hr id="hr-{anchor}">
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LCR Training Dashboard</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div id="layout">
  <div id="sidebar">
    <h2>LCR Runs ({len(runs)})</h2>
    {build_filter_box(runs)}
    {''.join(nav_html)}
  </div>
  <div id="main">
    {''.join(sections)}
  </div>
</div>
<script>{FILTER_JS}</script>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"\nSaved: {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")


def anchor_id(run: Run):
    return f"{run.group}-{run.folder_name}".replace(" ", "_").replace("(", "").replace(")", "")


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="Root folder to search recursively for run directories "
                             "(e.g. your configs/ folder).")
    parser.add_argument("--out", default="dashboard.html",
                        help="Output HTML file path.")
    args = parser.parse_args()

    root = Path(os.path.expanduser(args.root))
    runs = discover_runs(root)
    if not runs:
        raise SystemExit(f"No run directories found under {root} "
                         f"(looked for folders containing config.json + history.json).")

    print(f"Found {len(runs)} run(s):")
    for r in runs:
        print(f"  - {r.group}/{r.folder_name}")

    build_html(runs, Path(args.out))


if __name__ == "__main__":
    main()