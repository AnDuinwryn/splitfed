from __future__ import annotations

import os
import sys


def _ansi() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") not in {"", "dumb"}


def _panel(title: str, lines: list[str]) -> str:
    # subtle title, no heavy separators
    if _ansi():
        # Claude Code-ish muted orange for dataset titles
        head = f"\x1b[38;2;226;156;83m{title}\x1b[0m"
    else:
        head = title
    return "\n".join([head, *lines])


def _fmt(x, *, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


def _metrics_line(block: dict) -> str:
    auc = block.get("auc", block.get("roc_auc"))
    m = block.get("metrics") or {}
    return (
        f"Accuracy {_fmt(m.get('accuracy'))}  "
        f"Precision {_fmt(m.get('precision'))}  "
        f"Recall {_fmt(m.get('recall'))}  "
        f"F1 {_fmt(m.get('f1_score'))}  "
        f"AUC {_fmt(auc)}"
    )

def _cr_table(block: dict) -> list[str]:
    """Compact classification_report table (sklearn output_dict)."""
    cr = block.get("classification_report")
    if not isinstance(cr, dict):
        return []

    def row(lbl: str, d: dict) -> str:
        p = _fmt(d.get("precision"))
        r = _fmt(d.get("recall"))
        f1 = _fmt(d.get("f1-score"))
        sup = d.get("support")
        sup_s = str(int(sup)) if isinstance(sup, (int, float)) else "—"
        return f"{lbl:<12}{p:>10}{r:>10}{f1:>10}{sup_s:>10}"

    header = f"{'label':<12}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}"
    if _ansi():
        header = f"\x1b[7m{header}\x1b[0m"
    lines = [header]
    for k in ("0", "1"):
        v = cr.get(k)
        if isinstance(v, dict):
            lines.append(row(k, v))
    for k in ("accuracy", "macro avg", "weighted avg"):
        v = cr.get(k)
        if isinstance(v, dict):
            lines.append(row(k, v))
    return lines


def _cm_line(block: dict) -> str:
    m = block.get("metrics") or {}
    cm = m.get("confusion_matrix")
    if cm is None:
        cm = block.get("confusion_matrix")
    if cm is None:
        return "cm _"
    # numpy arrays can't be used in truthy checks
    try:
        import numpy as np

        if isinstance(cm, np.ndarray):
            cm = cm.tolist()
    except Exception:
        pass
    if not isinstance(cm, (list, tuple)) or len(cm) != 2:
        return "cm _"
    try:
        tn, fp = cm[0]
        fn, tp = cm[1]
        tn_s = f"{int(tn):>3}"
        tp_s = f"{int(tp):>3}"
        if _ansi():
            tn_s = f"\x1b[7m{tn_s}\x1b[0m"
            tp_s = f"\x1b[7m{tp_s}\x1b[0m"
        return f"CM [[{tn_s} {int(fp):>3}] [{int(fn):>3} {tp_s}]]"
    except Exception:
        return "cm _"


def _cm_lines(block: dict) -> tuple[str, str]:
    """Return two CM lines for right-side table placement."""
    m = block.get("metrics") or {}
    cm = m.get("confusion_matrix")
    if cm is None:
        cm = block.get("confusion_matrix")
    if cm is None:
        return ("", "")
    try:
        import numpy as np

        if isinstance(cm, np.ndarray):
            cm = cm.tolist()
    except Exception:
        pass
    try:
        tn, fp = cm[0]
        fn, tp = cm[1]
        tn_s = f"{int(tn):>3}"
        fp_s = f"{int(fp):>3}"
        fn_s = f"{int(fn):>3}"
        tp_s = f"{int(tp):>3}"
        # requested format (no reverse video): two aligned bracket lines
        return (f"[{tn_s} {fp_s}]", f"[{fn_s} {tp_s}]")
    except Exception:
        return ("", "")


def _attach_cm_right(table_lines: list[str], block: dict) -> list[str]:
    """Attach 2x2 CM to the right of classification table rows 0/1."""
    if not table_lines:
        return table_lines
    left_w = max(len(x) for x in table_lines)
    cm0, cm1 = _cm_lines(block)
    out: list[str] = []
    for idx, ln in enumerate(table_lines):
        pad = " " * (left_w - len(ln))
        extra = ""
        if idx == 1 and cm0:
            extra = "   CM " + cm0
        elif idx == 2 and cm1:
            extra = "      " + cm1
        out.append(ln + pad + extra)
    return out


def print_eval(evaluation: dict, *, verbose: bool = False) -> None:
    """Pretty CLI print for evaluation dict returned by eval_pair/evaluation.py."""
    for ds_key, ds_title in (("chinese", "EENT"), ("german", "SVD")):
        if ds_key not in evaluation:
            continue
        sec = evaluation[ds_key]
        if not isinstance(sec, dict):
            continue

        blocks: list[str] = []
        if verbose:
            a = sec.get("single_a") or {}
            i = sec.get("single_i") or {}
            left = f"/a/:  {_metrics_line(a)}"
            right = f"/i/:  {_metrics_line(i)}"
            blocks.append(f"{left}    |    {right}")
        c = sec.get("combined") or {}
        blocks.append(f"/a/+/i/:  {_metrics_line(c)}")
        blocks.extend(_attach_cm_right(_cr_table(c), c))

        print(_panel(ds_title, blocks))
        print()

