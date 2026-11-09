"""Summarize measured training logs without inventing missing experiment results."""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path


def read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_run(directory: Path):
    rows = []
    path = directory / "metrics.jsonl"
    if path.exists():
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    best = read_json(directory / "best.json", {})
    config = read_json(directory / "config.json", {})
    synthetic = bool(best.get("synthetic", config.get("dataset", {}).get("synthetic", False)))
    synthetic = synthetic or any(bool(row.get("synthetic", False)) for row in rows)
    # Only explicit best.json constitutes checkpoint-selection evidence.
    summary = {
        "run": directory.name,
        "path": str(directory.resolve()),
        "synthetic": synthetic,
        "seed": config.get("seed", ""),
        "epochs_logged": len(rows),
        "best_epoch": best.get("epoch", ""),
        "best_validation_accuracy": best.get("accuracy", ""),
        "selection_source": best.get("source", ""),
        "parameters": best.get("parameters", ""),
        "seconds_logged": round(sum(r["seconds"] for r in rows if finite_number(r.get("seconds"))), 3),
        "selection_metric": best.get("selection_metric", ""),
        "status": "synthetic_smoke_only" if synthetic else ("measured_validation" if best else "no_selected_checkpoint"),
    }
    return {"summary": summary, "rows": rows, "best": best, "config": config}


def curve_svg(rows, keys, title, probability=False):
    width, height, left, top, bottom, right = 820, 280, 55, 36, 38, 20
    points = {}
    for key, label, color in keys:
        points[key] = [(r.get("epoch", i + 1), r[key]) for i, r in enumerate(rows)
                       if finite_number(r.get("epoch", i + 1)) and finite_number(r.get(key))]
    all_points = [p for line in points.values() for p in line]
    if not all_points:
        return '<p class="muted">No measured values for this chart.</p>'
    xmin, xmax = min(p[0] for p in all_points), max(p[0] for p in all_points)
    if xmax == xmin:
        xmax += 1
    ymin, ymax = (0.0, 1.0) if probability else (0.0, max(p[1] for p in all_points) * 1.06)
    ymax = max(ymax, ymin + 1e-6)
    plot_width, plot_height = width - left - right, height - top - bottom
    def x(v):
        return left + (v - xmin) / (xmax - xmin) * plot_width

    def y(v):
        return top + (ymax - v) / (ymax - ymin) * plot_height
    elements = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title, quote=True)}">',
                '<rect width="100%" height="100%" fill="white"/>',
                f'<text x="{left}" y="20" class="chart-title">{html.escape(title)}</text>']
    for tick in range(5):
        v = ymin + (ymax - ymin) * tick / 4
        label = f"{v:.0%}" if probability else f"{v:.3g}"
        elements += [f'<line x1="{left}" y1="{y(v):.2f}" x2="{width-right}" y2="{y(v):.2f}" stroke="#e5e7eb"/>',
                     f'<text x="{left-8}" y="{y(v)+4:.2f}" text-anchor="end">{label}</text>']
    for key, label, color in keys:
        line = points[key]
        if not line:
            continue
        coords = " ".join(f"{x(a):.2f},{y(b):.2f}" for a, b in line)
        elements.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
        if len(line) == 1:
            elements.append(f'<circle cx="{x(line[0][0]):.2f}" cy="{y(line[0][1]):.2f}" r="3" fill="{color}"/>')
    for tick in range(5):
        value = xmin + (xmax - xmin) * tick / 4
        elements.append(f'<text x="{x(value):.2f}" y="{height-20}" text-anchor="middle">{value:g}</text>')
    elements.append(f'<text x="{width-right}" y="{height-3}" text-anchor="end">Epoch</text></svg>')
    legend = " ".join(f'<span style="color:{color}">{html.escape(label)}</span>' for key, label, color in keys if points[key])
    return "".join(elements) + f'<p class="legend">{legend}</p>'


def render_html(runs):
    fields = ["run", "status", "best_epoch", "best_validation_accuracy", "selection_source", "parameters", "seconds_logged"]
    table = '<table><thead><tr>' + ''.join(f'<th>{html.escape(f)}</th>' for f in fields) + '</tr></thead><tbody>'
    for run in runs:
        record = run["summary"]
        cells = []
        for field in fields:
            value = record[field]
            if field == "best_validation_accuracy" and finite_number(value):
                value = f"{value:.2%}"
            cells.append(f'<td>{html.escape(str(value))}</td>')
        table += '<tr>' + ''.join(cells) + '</tr>'
    table += '</tbody></table>'
    sections = []
    for run in runs:
        record, rows = run["summary"], run["rows"]
        warning = '<p class="warning">SYNTHETIC SMOKE TEST — NOT CIFAR-10 OR COMPETITION ACCURACY</p>' if record["synthetic"] else ''
        accuracy = curve_svg(rows, [("val_acc", "Validation", "#0f766e"), ("ema_val_acc", "EMA validation", "#7c3aed")], "Validation accuracy", True)
        loss = curve_svg(rows, [("train_loss", "Train (possibly mixed targets)", "#b45309"), ("val_loss", "Validation CE", "#0f766e")], "Loss (train and validation objectives may differ)")
        sections.append(f'<section><h2>{html.escape(record["run"])}</h2>{warning}<p class="muted">{html.escape(record["path"])}</p>{accuracy}{loss}</section>')
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DLEND measured run summary</title><style>
body{font:15px system-ui,sans-serif;background:#f4f6f8;color:#172033;margin:0;padding:28px}main{max-width:1100px;margin:auto}h1{font-size:28px}h2{font-size:21px}section,.table-wrap{background:white;border:1px solid #dbe2ea;border-radius:10px;margin:20px 0;padding:20px;overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}td,th{padding:10px;text-align:left;border-bottom:1px solid #e5e7eb}svg{display:block;width:100%;max-width:900px}svg text{font:12px system-ui,sans-serif;fill:#475569}.chart-title{font-weight:600!important}.legend{font-size:13px;word-spacing:12px}.muted{color:#64748b;overflow-wrap:anywhere}.warning{background:#fff3cd;padding:12px;font-weight:650}
</style><main><h1>Measured experiment summary</h1><p>Only recorded artifacts are shown. Validation selects models; it is not official test accuracy. Synthetic runs are implementation checks. Missing values remain blank. Comparing different epochs, splits or budgets is not a controlled ablation.</p>''' + '<div class="table-wrap">' + table + '</div>' + ''.join(sections) + '</main></html>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or parent directories to scan recursively")
    parser.add_argument("--output", type=Path, default=Path("analysis"))
    args = parser.parse_args()
    directories = set()
    for root in args.runs:
        if not root.exists():
            parser.error(f"Path does not exist: {root}")
        for filename in ("metrics.jsonl", "best.json"):
            if (root / filename).exists():
                directories.add(root.resolve())
            directories.update(p.parent.resolve() for p in root.rglob(filename))
    if not directories:
        parser.error("No metrics.jsonl or best.json found; run training first.")
    runs = [load_run(directory) for directory in sorted(directories)]
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(runs[0]["summary"]))
        writer.writeheader()
        writer.writerows(run["summary"] for run in runs)
    html_path = args.output / "report.html"
    html_path.write_text(render_html(runs), encoding="utf-8")
    print(csv_path.resolve())
    print(html_path.resolve())


if __name__ == "__main__":
    main()
