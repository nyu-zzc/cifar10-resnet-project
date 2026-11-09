"""Reproduce report figures from the included aggregated, SHA-verified CSV tables.

Requirements: Python, numpy, matplotlib. No training or model inference.
All metrics are observed single-run descriptive results; no statistical tests.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

mpl.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 8.5, "axes.labelsize": 8.5,
    "axes.titlesize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "svg.fonttype": "none", "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .7,
    "legend.frameon": False, "savefig.facecolor": "white", "figure.facecolor": "white",
    "axes.axisbelow": True, "grid.alpha": .22, "grid.linewidth": .5,
})
BLUE, ORANGE, TEAL, PURPLE = "#416B91", "#C17D42", "#4E877D", "#806C98"
GREY = "#707579"
WIDTH, HEIGHT = 6.9, 3.0
QA = {"backend": "Python/matplotlib", "width_inches": WIDTH, "height_inches": HEIGHT,
      "png_dpi": 300, "svg_text": "editable", "pdf_fonttype": 42, "figures": {}}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(name):
    manifest = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    assert sha(HERE / name) == manifest["aggregated_tables"][name]["sha256"]
    with (HERE / name).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == manifest["aggregated_tables"][name]["rows"]
    return rows


def panel(ax, letter, title):
    ax.text(-.12, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.set_title(title, loc="left", pad=9)


def save(fig, name):
    fig.canvas.draw()
    texts = [t for t in fig.findobj(mpl.text.Text) if t.get_visible() and t.get_text()]
    assert min(t.get_fontsize() for t in texts) >= 8
    fig.savefig(HERE / f"{name}.svg")
    fig.savefig(HERE / f"{name}.pdf")
    fig.savefig(HERE / f"{name}.png", dpi=300)
    svg = (HERE / f"{name}.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    image = plt.imread(HERE / f"{name}.png")
    assert image.shape[:2] == (900, 2070)
    QA["figures"][name] = {"minimum_font_pt": min(t.get_fontsize() for t in texts),
        "png_pixels": [2070, 900], "svg_text_elements": svg.count("<text"),
        "files_sha256": {suffix: sha(HERE / f"{name}.{suffix}") for suffix in ("svg", "pdf", "png")}}
    plt.close(fig)


def training():
    rows = table("figure_1_training_curves.csv")
    fig = plt.figure(figsize=(6.9, 3.0))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=.085, right=.985, bottom=.28, top=.82, wspace=.34)
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    for ax, names, colors, shorthand in [(left, ["Three-stage,400", "Four-stage,400"], [BLUE, ORANGE], ["3-stage", "4-stage"]),
            (right, ["Control,60", "Mild appearance,60"], [GREY, TEAL], ["Control", "Mild"] )]:
        all_values = []
        for name, color, short in zip(names, colors, shorthand):
            for weights, style in (("Raw", "-"), ("EMA", "--")):
                values = [r for r in rows if r["run"] == name and r["weights"] == weights]
                x = np.array([int(r["epoch"]) for r in values])
                y = np.array([float(r["accuracy_percent"]) for r in values])
                all_values.extend(y)
                ax.plot(x, y, color=color, linestyle=style, linewidth=.9, label=f"{short} {weights}")
        ax.set_ylim(min(all_values) - .04 * (max(all_values)-min(all_values)), max(all_values) + .06 * (max(all_values)-min(all_values)))
        ax.set_xlabel("Epoch within run")
        ax.grid(axis="y")
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23), ncol=2, columnspacing=.9, handlelength=1.6, borderaxespad=0)
    left.set_xlim(1, 400)
    right.set_xlim(1, 60)
    left.set_ylabel("Validation accuracy (%)")
    panel(left, "a", "Full training runs")
    panel(right, "b", "Matched refinement runs")
    zoom = left.inset_axes([.38, .09, .57, .47])
    late_values = []
    for name, color in [("Three-stage,400", BLUE), ("Four-stage,400", ORANGE)]:
        for weights, style in (("Raw", "-"), ("EMA", "--")):
            values = [r for r in rows if r["run"] == name and r["weights"] == weights and int(r["epoch"]) >= 301]
            xx, yy = [int(r["epoch"]) for r in values], [float(r["accuracy_percent"]) for r in values]
            late_values.extend(yy)
            zoom.plot(xx, yy, color=color, linestyle=style, linewidth=.75)
    zoom.set_xlim(301, 400)
    zoom.set_ylim(min(late_values) - .08, max(late_values) + .08)
    zoom.set_xticks([320, 360, 400])
    zoom.set_yticks([96, 97])
    zoom.tick_params(labelsize=8, length=2, pad=1)
    zoom.set_title("Last 100 epochs", fontsize=8, pad=2)
    fig.text(.5, .012, "Trainer validation: BF16, no TTA; all epochs shown, no smoothing.", ha="center", fontsize=8)
    save(fig, "fig1_training_curves")


def official():
    rows = table("figure_2_official_scores.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=.22, right=.975, bottom=.20, top=.78)
    for i, row in enumerate(rows):
        selected = row["selected_for_final"] == "True"
        if selected:
            ax.axhspan(i-.43, i+.43, color="#F6F6F6", ec="#777777", linewidth=.7, zorder=0)
        for field, offset, color, marker in [("local_percent", -.19, BLUE, "o"), ("public_percent", 0, ORANGE, "s"), ("private_percent", .19, PURPLE, "D")]:
            value = float(row[field])
            ax.scatter(value, i+offset, s=27, marker=marker, color=color,
                       edgecolors="#222222" if selected else "white", linewidths=.9 if selected else .4, zorder=3)
            ax.text(value+.16, i+offset, f"{value:.2f}" if field == "local_percent" else f"{value:.1f}", fontsize=8, va="center")
    ax.set_yticks(range(len(rows)), [r["model"] for r in rows])
    ax.set_ylim(4.55, -.55)
    ax.set_xlim(83.7, 99.4)
    ax.set_xticks([84, 88, 92, 96])
    ax.set_xlabel("Accuracy (%)")
    ax.grid(axis="x")
    handles = [Line2D([], [], marker=marker, color=color, linestyle="", label=name, markersize=5)
               for name, color, marker in [("Local", BLUE, "o"), ("Public", ORANGE, "s"), ("Private", PURPLE, "D")]]
    handles.append(Patch(facecolor="#F6F6F6", edgecolor="#777777", label="Selected"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(.47, 1.06), ncol=4, columnspacing=1.1, handlelength=1.2)
    fig.text(.5, .025, "Selected final: 86.3%; highest observed private: 87.3% (unselected).", ha="center", fontsize=8)
    save(fig, "fig2_official_scores")


def architecture():
    rows = table("figure_3_architecture_scores.csv")
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    fig.subplots_adjust(left=.35, right=.96, bottom=.19, top=.90)
    colors = {"Original": BLUE, "Kernel3": PURPLE, "Pool4": TEAL, "Four-stage": ORANGE}
    labels = []
    for i, row in enumerate(rows):
        acc = float(row["accuracy_percent"])
        ax.scatter(acc, i, color=colors[row["family"]], marker="D" if row["artifact"] == "SWA" else "o", s=32, zorder=3)
        ax.text(acc+.025, i, f"{acc:.2f}%", va="center", fontsize=8.5)
        labels.append(f"{row['model']} ({row['artifact']}; {row['branch_compute_epochs']} ep)")
    ax.axvline(97.52, color=BLUE, linestyle=":", linewidth=.9, zorder=0)
    ax.set_xlim(96.65, 97.75)
    ax.set_ylim(6.6, -.6)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_xticks([96.8, 97.0, 97.2, 97.4, 97.6])
    ax.set_xlabel("Fixed-validation accuracy (%)")
    ax.grid(axis="x")
    fig.text(.5, .95, "CPU FP32 + flip; training budgets shown explicitly", ha="center", fontsize=9)
    fig.text(.5, .025, "Different budgets and selected artifacts; descriptive comparison, not an isolated causal test.", ha="center", fontsize=8)
    save(fig, "fig3_architecture_comparison")


def errors():
    class_rows = table("figure_4_class_errors.csv")
    proxy_rows = table("figure_4_proxy_scores.csv")
    names = ["Local champion", "Mild appearance", "Robust control", "Robust refinement"]
    class_names = [r["class_name"] for r in class_rows if r["model"] == names[0]]
    matrix = np.array([[int(r["errors"]) for r in class_rows if r["model"] == name] for name in names])
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    grid = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], left=.16, right=.97, bottom=.28, top=.80, wspace=.31)
    left, right = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
    left.imshow(matrix, cmap="Blues", vmin=0, vmax=45, aspect="auto")
    for i in range(4):
        for j in range(10):
            left.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8, color="white" if matrix[i, j] >= 29 else "#1D2935")
    left.set_xticks(range(10), class_names, rotation=50, ha="right", rotation_mode="anchor")
    left.set_yticks(range(4), names)
    left.tick_params(length=0)
    panel(left, "a", "Errors per class / 500 (flip)")
    for i, name in enumerate(names):
        values = {r["condition"]: float(r["accuracy_percent"]) for r in proxy_rows if r["model"] == name}
        right.plot([values["Mild blur"], values["Clean"]], [i+.13, i-.13], color="#A6A6A6", linewidth=.8, zorder=1)
        for key, offset, color, marker in [("Clean", -.13, BLUE, "o"), ("Mild blur", .13, ORANGE, "s")]:
            x = values[key]
            right.scatter(x, i+offset, color=color, marker=marker, s=23, zorder=2)
            right.text(x+.13, i+offset, f"{x:.1f}", va="center", fontsize=8)
    right.set_ylim(3.5, -.5)
    right.set_xlim(92.9, 99.0)
    right.set_xticks([94, 96, 98])
    right.set_yticks([])
    right.set_xlabel("Accuracy (%)")
    right.grid(axis="x")
    panel(right, "b", "Clean and mild blur")
    handles = [Line2D([], [], color=c, marker=m, linestyle="", markersize=5, label=n)
               for n, c, m in [("Clean", BLUE, "o"), ("Mild blur", ORANGE, "s")]]
    right.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.29), ncol=2, handlelength=1, columnspacing=.6)
    fig.text(.5, .025, "Left: 5,000 images. Right: same 1,000, no TTA; Gaussian blur 3×3, σ=0.6.", ha="center", fontsize=8)
    save(fig, "fig4_errors_and_robustness")


if __name__ == "__main__":
    training()
    official()
    architecture()
    errors()
    QA["plotting_script_sha256"] = sha(__file__)
    QA["sources_sha256"] = sha(HERE / "sources.json")
    QA["visual_inspection"] = "Required after render; recorded separately in visual_qa.json"
    (HERE / "export_qa.json").write_text(json.dumps(QA, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "rendered", "figures": list(QA["figures"]), "dimensions_inches": [WIDTH, HEIGHT], "minimum_font_pt": 8}))
