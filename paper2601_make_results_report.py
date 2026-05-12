from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


DATASET_LABEL = {"chinese": "EENT", "german": "SVD"}


@dataclass
class MetricRow:
    result_id: str
    group: str
    protocol: str
    dataset: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    specificity: Optional[float]
    f1: Optional[float]
    auc: Optional[float]
    cm: Optional[list[list[int]]]
    source_file: str
    notes: str = ""


def _safe_load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[skip] cannot parse JSON {path}: {exc}")
        return None


def _fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return ""
    return f"{float(v):.{digits}f}"


def _short_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _metric(combined: dict[str, Any], key: str) -> Optional[float]:
    metrics = combined.get("metrics") or {}
    value = metrics.get(key)
    if value is None and key == "f1_score":
        value = metrics.get("f1")
    if value is None:
        value = combined.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _auc(combined: dict[str, Any]) -> Optional[float]:
    for key in ("auc", "roc_auc"):
        if combined.get(key) is not None:
            try:
                return float(combined[key])
            except Exception:
                return None
    return None


def _cm(combined: dict[str, Any]) -> Optional[list[list[int]]]:
    cm = combined.get("confusion_matrix") or (combined.get("metrics") or {}).get("confusion_matrix")
    if not cm:
        return None
    try:
        return [[int(cm[0][0]), int(cm[0][1])], [int(cm[1][0]), int(cm[1][1])]]
    except Exception:
        return None


def _cm_text(cm: Optional[list[list[int]]]) -> str:
    if not cm:
        return ""
    return f"[{cm[0][0]} {cm[0][1]}; {cm[1][0]} {cm[1][1]}]"


def _dataset_blocks(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("evaluation"), dict):
        return payload["evaluation"]
    if isinstance(payload.get("datasets"), dict):
        return payload["datasets"]
    return payload


def _baseline_name(path: Path, payload: dict[str, Any]) -> Optional[str]:
    name = path.name
    if name == "eval_results_fixed.json":
        return "centralized CNN"
    if name == "eval_results_split_fixed.json":
        return "split CNN"
    if name == "eval_results_split_ssast_fixed.json":
        return "split SSAST"
    meta = payload.get("meta") or {}
    model_type = meta.get("model_type")
    if model_type and "saved_models" in path.parts:
        return str(model_type)
    return None


def _group_for(path: Path) -> str:
    text = str(path).lower()
    if "saved_models" in path.parts:
        return "baseline"
    if "control" in text:
        return "controlled"
    if "ablation" in text:
        return "static ablation"
    if "youden" in text:
        return "youden diagnostic"
    if "svd_independent" in text:
        return "scaling diagnostic"
    return "paper2601"


def _metadata(payload: dict[str, Any], vowel: str = "a") -> dict[str, Any]:
    key = f"loaded_metadata_{vowel}"
    if isinstance(payload.get(key), dict):
        return payload[key]
    meta = payload.get("meta")
    return meta if isinstance(meta, dict) else {}


def _result_id(path: Path, payload: dict[str, Any], root: Path) -> str:
    baseline = _baseline_name(path, payload)
    if baseline:
        return baseline

    parent = path.parent.name
    name = parent
    if path.parent == root:
        name = path.stem

    meta = _metadata(payload, "a")
    preset = meta.get("static_feature_preset")
    local = meta.get("n_local_epochs")
    controlled = meta.get("controlled_fusion") or {}
    if controlled:
        mode = controlled.get("fusion_mode")
        clip = controlled.get("static_z_clip")
        bits = [name]
        if mode:
            bits.append(str(mode))
        if preset:
            bits.append(str(preset))
        if local:
            bits.append(f"L{local}")
        if clip:
            bits.append(f"clip{clip}")
        return " | ".join(bits)

    if preset or local:
        bits = [name]
        if preset:
            bits.append(str(preset))
        if local:
            bits.append(f"L{local}")
        return " | ".join(bits)
    return name


def _protocol(path: Path, payload: dict[str, Any]) -> str:
    blocks = _dataset_blocks(payload)
    strategy = payload.get("patient_eval_strategy") or blocks.get("patient_eval_strategy")
    threshold = payload.get("patient_prob_threshold") or blocks.get("patient_prob_threshold")
    text = []
    fname = path.name.lower()
    if "youden" in fname or strategy == "best_threshold":
        text.append("Youden/best_threshold")
    elif strategy:
        if strategy in {"fixed", "guding"} and threshold is not None:
            text.append(f"fixed {float(threshold):g}")
        else:
            text.append(str(strategy))
    else:
        text.append("fixed/unknown")
    if "svd_independent" in fname:
        text.append("SVD-independent static scaling")
    elif "eent_train_scaling" in fname:
        text.append("EENT-train static scaling")
    return " + ".join(text)


def _notes(row: MetricRow) -> str:
    notes = []
    if row.dataset == "SVD":
        if row.specificity is not None and row.specificity <= 0.05:
            notes.append("all-positive or near all-positive")
        if row.auc is not None and row.auc < 0.65:
            notes.append("weak ranking")
        if row.accuracy is not None and row.accuracy >= 0.80:
            notes.append("strong external accuracy")
    if row.dataset == "EENT":
        if row.accuracy is not None and row.accuracy >= 0.94:
            notes.append("strong EENT")
    return "; ".join(notes)


def discover_result_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in (
        "saved_models/*eval*.json",
        "paper2601_splitmae_runs*/**/*eval*.json",
        "paper2601_splitmae_runs*/**/*summary*.json",
        "paper2601_splitmae_runs_youden_summary/*.json",
    ):
        files.update(p for p in root.glob(pattern) if p.is_file())
    return sorted(files)


def extract_rows(path: Path, root: Path) -> list[MetricRow]:
    payload = _safe_load_json(path)
    if payload is None:
        return []
    blocks = _dataset_blocks(payload)
    result_id = _result_id(path, payload, root)
    group = _group_for(path)
    protocol = _protocol(path, payload)
    rows: list[MetricRow] = []
    for dataset_key, dataset_label in DATASET_LABEL.items():
        block = blocks.get(dataset_key)
        if not isinstance(block, dict):
            continue
        combined = block.get("combined")
        if not isinstance(combined, dict):
            continue
        row = MetricRow(
            result_id=result_id,
            group=group,
            protocol=protocol,
            dataset=dataset_label,
            accuracy=_metric(combined, "accuracy"),
            precision=_metric(combined, "precision"),
            recall=_metric(combined, "recall"),
            specificity=_metric(combined, "specificity"),
            f1=_metric(combined, "f1_score"),
            auc=_auc(combined),
            cm=_cm(combined),
            source_file=_short_path(path, root),
        )
        row.notes = _notes(row)
        rows.append(row)
    return rows


def _wide_rows(rows: Iterable[MetricRow]) -> list[dict[str, Any]]:
    by_id: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row.result_id, row.group, row.protocol)
        item = by_id.setdefault(
            key,
            {
                "result_id": row.result_id,
                "group": row.group,
                "protocol": row.protocol,
                "source_files": set(),
            },
        )
        item[row.dataset] = row
        item["source_files"].add(row.source_file)
    out = []
    for item in by_id.values():
        item["source_files"] = sorted(item["source_files"])
        out.append(item)
    return sorted(
        out,
        key=lambda x: (
            x.get("group", ""),
            -(x.get("EENT").accuracy or -1) if x.get("EENT") else 1,
            x.get("result_id", ""),
        ),
    )


def _cell(row: Optional[MetricRow], field: str) -> str:
    if row is None:
        return ""
    return _fmt(getattr(row, field))


def _cm_cell(row: Optional[MetricRow]) -> str:
    if row is None:
        return ""
    return _cm_text(row.cm)


def write_tsv(path: Path, wide: list[dict[str, Any]]) -> None:
    headers = [
        "result",
        "group",
        "protocol",
        "eent_acc",
        "eent_f1",
        "eent_auc",
        "eent_cm",
        "svd_acc",
        "svd_f1",
        "svd_auc",
        "svd_specificity",
        "svd_recall",
        "svd_cm",
        "notes",
        "source_files",
    ]
    lines = ["\t".join(headers)]
    for item in wide:
        e = item.get("EENT")
        s = item.get("SVD")
        notes = "; ".join(n for n in [getattr(e, "notes", ""), getattr(s, "notes", "")] if n)
        values = [
            item["result_id"],
            item["group"],
            item["protocol"],
            _cell(e, "accuracy"),
            _cell(e, "f1"),
            _cell(e, "auc"),
            _cm_cell(e),
            _cell(s, "accuracy"),
            _cell(s, "f1"),
            _cell(s, "auc"),
            _cell(s, "specificity"),
            _cell(s, "recall"),
            _cm_cell(s),
            notes,
            " | ".join(item["source_files"]),
        ]
        lines.append("\t".join(str(v).replace("\t", " ") for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape(text: Any) -> str:
    return html.escape(str(text), quote=True)


def _metric_badge(value: Optional[float], high: float, low: float) -> str:
    if value is None:
        return ""
    cls = "good" if value >= high else "bad" if value < low else "mid"
    return f'<span class="{cls}">{_fmt(value)}</span>'


def _scatter_svg(wide: list[dict[str, Any]]) -> str:
    points = []
    for item in wide:
        e = item.get("EENT")
        s = item.get("SVD")
        if not e or not s or e.accuracy is None or s.accuracy is None:
            continue
        points.append((item, e.accuracy, s.accuracy))
    if not points:
        return '<div class="empty">No paired EENT/SVD rows available.</div>'
    x_min, x_max = 0.50, 1.00
    y_min, y_max = 0.45, 0.90
    W, H = 820, 430
    left, right, top, bottom = 72, 28, 28, 58
    plot_w, plot_h = W - left - right, H - top - bottom

    def sx(v: float) -> float:
        return left + (v - x_min) / (x_max - x_min) * plot_w

    def sy(v: float) -> float:
        return top + (y_max - v) / (y_max - y_min) * plot_h

    palette = {
        "baseline": "#3867B7",
        "paper2601": "#277566",
        "static ablation": "#C9862A",
        "controlled": "#6D5AA7",
        "youden diagnostic": "#777777",
        "scaling diagnostic": "#B94A48",
    }
    elems = [
        f'<svg viewBox="0 0 {W} {H}" class="chart">',
        f'<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="#fff"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#17202A" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#17202A" stroke-width="1.5"/>',
    ]
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        x = sx(tick)
        elems.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#ECE7DD"/>')
        elems.append(f'<text x="{x:.1f}" y="{top + plot_h + 24}" text-anchor="middle" class="axis">{tick:.1f}</text>')
    for tick in [0.5, 0.6, 0.7, 0.8, 0.9]:
        y = sy(tick)
        elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ECE7DD"/>')
        elems.append(f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="axis">{tick:.1f}</text>')
    elems.append(f'<text x="{left + plot_w / 2:.1f}" y="{H - 16}" text-anchor="middle" class="axis-title">EENT accuracy</text>')
    elems.append(f'<text x="18" y="{top + plot_h / 2:.1f}" text-anchor="middle" class="axis-title" transform="rotate(-90 18 {top + plot_h / 2:.1f})">SVD accuracy</text>')
    for item, xv, yv in points:
        color = palette.get(item["group"], "#17202A")
        x, y = sx(xv), sy(yv)
        label = item["result_id"]
        if len(label) > 34:
            label = label[:31] + "..."
        elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"><title>{_escape(item["result_id"])}: EENT {_fmt(xv)}, SVD {_fmt(yv)}</title></circle>')
        elems.append(f'<text x="{x + 9:.1f}" y="{y - 8:.1f}" class="point-label" fill="{color}">{_escape(label)}</text>')
    elems.append("</svg>")
    return "\n".join(elems)


def _bar_svg(wide: list[dict[str, Any]], metric: str, dataset: str, title: str, limit: int = 18) -> str:
    items = []
    for item in wide:
        row = item.get(dataset)
        value = getattr(row, metric) if row else None
        if value is not None:
            items.append((item["result_id"], item["group"], value))
    items.sort(key=lambda x: x[2], reverse=True)
    items = items[:limit]
    if not items:
        return '<div class="empty">No metric rows available.</div>'
    W = 820
    row_h = 28
    H = 58 + row_h * len(items)
    left = 260
    right = 34
    bar_w = W - left - right
    elems = [f'<svg viewBox="0 0 {W} {H}" class="chart small">', f'<rect width="{W}" height="{H}" rx="12" fill="#fff"/>']
    elems.append(f'<text x="22" y="30" class="chart-title">{_escape(title)}</text>')
    for idx, (name, group, value) in enumerate(items):
        y = 52 + idx * row_h
        label = name if len(name) <= 36 else name[:33] + "..."
        color = "#3867B7" if group == "baseline" else "#6D5AA7" if group == "controlled" else "#277566" if group == "paper2601" else "#C9862A"
        elems.append(f'<text x="22" y="{y + 15}" class="bar-label">{_escape(label)}</text>')
        elems.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="16" fill="#ECE7DD"/>')
        elems.append(f'<rect x="{left}" y="{y}" width="{bar_w * max(0.0, min(value, 1.0)):.1f}" height="16" fill="{color}"/>')
        elems.append(f'<text x="{left + bar_w + 10}" y="{y + 13}" class="bar-value">{_fmt(value)}</text>')
    elems.append("</svg>")
    return "\n".join(elems)


def _summary_cards(wide: list[dict[str, Any]]) -> str:
    def best(dataset: str, metric: str):
        candidates = []
        for item in wide:
            row = item.get(dataset)
            if row is not None and getattr(row, metric) is not None:
                candidates.append((getattr(row, metric), item["result_id"], item["group"]))
        return max(candidates, default=None)

    cards = [
        ("Best EENT Acc", best("EENT", "accuracy")),
        ("Best EENT F1", best("EENT", "f1")),
        ("Best SVD Acc", best("SVD", "accuracy")),
        ("Best SVD AUC", best("SVD", "auc")),
    ]
    html_cards = []
    for title, item in cards:
        if item is None:
            value, name, group = "", "missing", ""
        else:
            value, name, group = _fmt(item[0]), item[1], item[2]
        html_cards.append(
            f'<div class="card"><div class="card-title">{_escape(title)}</div>'
            f'<div class="card-value">{_escape(value)}</div>'
            f'<div class="card-note">{_escape(name)}<br>{_escape(group)}</div></div>'
        )
    return "\n".join(html_cards)


def _wide_table(wide: list[dict[str, Any]]) -> str:
    rows = []
    for item in wide:
        e = item.get("EENT")
        s = item.get("SVD")
        notes = "; ".join(n for n in [getattr(e, "notes", ""), getattr(s, "notes", "")] if n)
        rows.append(
            "<tr>"
            f"<td>{_escape(item['result_id'])}</td>"
            f"<td>{_escape(item['group'])}</td>"
            f"<td>{_escape(item['protocol'])}</td>"
            f"<td>{_metric_badge(e.accuracy if e else None, 0.94, 0.88)}</td>"
            f"<td>{_metric_badge(e.f1 if e else None, 0.94, 0.88)}</td>"
            f"<td>{_metric_badge(e.auc if e else None, 0.96, 0.85)}</td>"
            f"<td>{_escape(_cm_cell(e))}</td>"
            f"<td>{_metric_badge(s.accuracy if s else None, 0.80, 0.70)}</td>"
            f"<td>{_metric_badge(s.f1 if s else None, 0.80, 0.70)}</td>"
            f"<td>{_metric_badge(s.auc if s else None, 0.85, 0.70)}</td>"
            f"<td>{_metric_badge(s.specificity if s else None, 0.75, 0.40)}</td>"
            f"<td>{_escape(_cm_cell(s))}</td>"
            f"<td>{_escape(notes)}</td>"
            f"<td class='source'>{_escape(' | '.join(item['source_files']))}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_html(path: Path, wide: list[dict[str, Any]], scanned: list[Path], root: Path, tsv_name: str) -> None:
    missing_control = not any(item.get("group") == "controlled" for item in wide)
    warning = ""
    if missing_control:
        warning = (
            '<div class="warn"><b>No controlled-run JSON found in this workspace.</b> '
            'Run this report on the Linux host after the training directories exist, or copy '
            '<code>paper2601_splitmae_runs_control_*</code> back here.</div>'
        )
    body = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Paper2601 Result Report</title>
<style>
:root {{
  --bg:#f7f4ee; --ink:#17202a; --muted:#69727f; --line:#d6d0c7;
  --green:#277566; --blue:#3867b7; --amber:#c9862a; --red:#b94a48; --purple:#6d5aa7;
}}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:Arial, "Microsoft YaHei UI", sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:36px 28px 60px; }}
h1 {{ margin:0 0 8px; font-size:34px; line-height:1.15; }}
h2 {{ margin:34px 0 14px; font-size:22px; }}
p {{ color:var(--muted); line-height:1.55; }}
code {{ background:#ece7dd; padding:2px 5px; border-radius:4px; }}
.cards {{ display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin:26px 0; }}
.card {{ background:white; border:1px solid var(--line); padding:18px; min-height:118px; }}
.card-title {{ color:var(--green); font-size:12px; font-weight:700; text-transform:uppercase; }}
.card-value {{ font-size:34px; font-weight:800; margin:10px 0 8px; }}
.card-note {{ color:var(--muted); font-size:12px; line-height:1.35; }}
.grid {{ display:grid; grid-template-columns:1fr; gap:18px; }}
.chart {{ width:100%; max-width:900px; border:1px solid var(--line); box-shadow:0 1px 0 rgba(0,0,0,.03); }}
.chart.small {{ max-width:900px; }}
.axis,.bar-label,.bar-value,.point-label {{ font-family:Arial, sans-serif; font-size:12px; fill:#69727f; }}
.axis-title,.chart-title {{ font-family:Arial, sans-serif; font-size:14px; font-weight:700; fill:#17202a; }}
.point-label {{ font-weight:700; font-size:11px; }}
.bar-label {{ font-weight:700; fill:#17202a; }}
.bar-value {{ fill:#17202a; }}
.warn {{ background:#fff5f5; border-left:5px solid var(--red); padding:14px 16px; margin:20px 0; color:#5b2625; }}
.note {{ background:#fff; border-left:5px solid var(--green); padding:14px 16px; margin:20px 0; color:var(--muted); }}
table {{ width:100%; border-collapse:collapse; background:#fff; border:1px solid var(--line); font-size:13px; }}
th,td {{ border-bottom:1px solid #ebe6dd; padding:9px 10px; text-align:right; vertical-align:top; }}
th {{ background:#ece7dd; font-size:12px; text-transform:uppercase; color:#333; position:sticky; top:0; }}
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2), td:nth-child(3), th:nth-child(3), td:nth-child(13), th:nth-child(13), td:nth-child(14), th:nth-child(14) {{ text-align:left; }}
.source {{ color:var(--muted); font-size:11px; max-width:280px; word-break:break-word; }}
.good {{ color:var(--green); font-weight:800; }}
.mid {{ color:var(--amber); font-weight:800; }}
.bad {{ color:var(--red); font-weight:800; }}
.table-wrap {{ overflow:auto; border:1px solid var(--line); }}
.empty {{ padding:20px; background:#fff; border:1px solid var(--line); color:var(--muted); }}
details {{ margin-top:18px; }}
summary {{ cursor:pointer; font-weight:700; }}
@media (max-width:900px) {{ .cards {{ grid-template-columns:1fr 1fr; }} }}
</style>
</head>
<body>
<main>
<h1>Paper2601 / SplitFed Results Report</h1>
<p>Standalone report rebuilt from evaluation JSON files. This does not depend on terminal scrollback. TSV export: <code>{_escape(tsv_name)}</code>.</p>
{warning}
<div class="cards">
{_summary_cards(wide)}
</div>
<div class="note">Read fixed-threshold rows as formal evaluation. Youden and SVD-independent scaling rows are diagnostics unless explicitly selected as the protocol.</div>

<h2>EENT vs SVD fixed/diagnostic landscape</h2>
{_scatter_svg(wide)}

<h2>SVD ranking / accuracy views</h2>
<div class="grid">
{_bar_svg(wide, "auc", "SVD", "Top SVD AUC rows")}
{_bar_svg(wide, "accuracy", "SVD", "Top SVD accuracy rows")}
</div>

<h2>Combined /a/+/i/ patient-level table</h2>
<div class="table-wrap">
<table>
<thead>
<tr>
<th>Result</th><th>Group</th><th>Protocol</th>
<th>EENT Acc</th><th>EENT F1</th><th>EENT AUC</th><th>EENT CM</th>
<th>SVD Acc</th><th>SVD F1</th><th>SVD AUC</th><th>SVD Spec</th><th>SVD CM</th>
<th>Notes</th><th>Source</th>
</tr>
</thead>
<tbody>
{_wide_table(wide)}
</tbody>
</table>
</div>

<details>
<summary>Scanned JSON files ({len(scanned)})</summary>
<ul>
{''.join(f'<li><code>{_escape(_short_path(p, root))}</code></li>' for p in scanned)}
</ul>
</details>
</main>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")


def build_report(root: Path, out_html: Path, out_tsv: Path) -> None:
    files = discover_result_files(root)
    rows: list[MetricRow] = []
    for file in files:
        rows.extend(extract_rows(file, root))
    wide = _wide_rows(rows)
    write_tsv(out_tsv, wide)
    write_html(out_html, wide, files, root, out_tsv.name)
    print(f"Wrote HTML report: {out_html.resolve()}")
    print(f"Wrote TSV summary: {out_tsv.resolve()}")
    print(f"Rows: {len(wide)} result rows from {len(files)} JSON files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a visual HTML report from Paper2601/SplitFed eval JSON files.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-html", type=Path, default=Path("paper2601_results_report.html"))
    parser.add_argument("--out-tsv", type=Path, default=Path("paper2601_results_report.tsv"))
    args = parser.parse_args()
    root = args.root.resolve()
    build_report(root, args.out_html, args.out_tsv)


if __name__ == "__main__":
    main()
