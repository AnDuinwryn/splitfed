from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from sklearn.metrics import auc, classification_report, roc_auc_score, roc_curve

from .metrics_basic import calculate_performance_metrics, print_performance_metrics


def _normalize_patient_id(pid) -> str:
    """Hashable key for Chinese string IDs (e.g. EENT) or numeric German IDs."""
    if hasattr(pid, "item"):
        pid = pid.item()
    return str(pid).strip()


def model_eval_by_id(
    x_test: np.ndarray,
    y_test: np.ndarray,
    id_test: list,
    segment_positive_probs: np.ndarray,
    *,
    strategy: Literal["best_threshold", "fixed", "relative", "percentage", "max recall", "guding"] = "fixed",
    segment_threshold: float = 0.5,
    patient_prob_threshold: float = 0.5,
    percentage: float = 0.2,
    vowel_type: Optional[str] = None,
    roc_label: int = 1,
    dataset_type: str = "Test",
    show_plots: bool = False,
    sensitive_attrs: Optional[dict] = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Patient-level evaluation: average segment positive probabilities per patient, then threshold.
    """
    _ = sensitive_attrs
    _ = x_test
    y_test = np.asarray(y_test)
    if y_test.ndim == 2 and y_test.shape[1] == 1:
        y_test_lbl = y_test.reshape(-1).astype(int)
    elif y_test.ndim == 1:
        y_test_lbl = y_test.astype(int)
    else:
        y_test_lbl = np.argmax(y_test, axis=1)

    segment_positive_probs = np.asarray(segment_positive_probs).reshape(-1)
    y_pred = np.stack([1.0 - segment_positive_probs, segment_positive_probs], axis=1)

    patient_data: dict = {}
    for i, patient_id in enumerate(id_test):
        if patient_id not in patient_data:
            patient_data[patient_id] = {
                "segments": [],
                "true_label": int(y_test_lbl[i]),
                "segment_probs": [],
                "segment_preds": [],
                "pos_segments": 0,
                "neg_segments": 0,
            }
        segment_prob = float(y_pred[i, 1])
        segment_pred = 1 if segment_prob >= segment_threshold else 0
        d = patient_data[patient_id]
        d["segments"].append(i)
        d["segment_probs"].append(segment_prob)
        d["segment_preds"].append(segment_pred)
        if segment_pred == 1:
            d["pos_segments"] += 1
        else:
            d["neg_segments"] += 1

    patient_ids = list(patient_data.keys())
    for pid in patient_ids:
        data = patient_data[pid]
        data["avg_prob_0"] = float(np.mean([y_pred[j, 0] for j in data["segments"]]))
        data["avg_prob_1"] = float(np.mean([y_pred[j, 1] for j in data["segments"]]))
        data["avg_prob"] = data["avg_prob_1"]

    pt_test_lbl = np.array([patient_data[pid]["true_label"] for pid in patient_ids], dtype=int)
    pt_pred = np.array([[patient_data[pid]["avg_prob_0"], patient_data[pid]["avg_prob_1"]] for pid in patient_ids])

    fpr, tpr, thresholds = roc_curve(pt_test_lbl, pt_pred[:, roc_label], pos_label=roc_label)
    roc_auc = auc(fpr, tpr)
    optimal_idx = int(np.argmax(tpr - fpr))
    optimal_threshold = float(thresholds[optimal_idx])

    pt_pred_lbl: list[int] = []
    for pid in patient_ids:
        data = patient_data[pid]
        prob = float(data["avg_prob"])
        pos_segments = int(data["pos_segments"])
        neg_segments = int(data["neg_segments"])
        total_segments = pos_segments + neg_segments

        if strategy == "best_threshold":
            pred = 1 if prob >= optimal_threshold else 0
        elif strategy in ("fixed", "guding"):
            pred = 1 if prob >= patient_prob_threshold else 0
        elif strategy == "relative":
            pred = 1 if pos_segments >= neg_segments else 0
        elif strategy == "percentage":
            segment_pct = pos_segments / total_segments if total_segments > 0 else 0.0
            pred = 1 if segment_pct >= percentage else 0
        elif strategy == "max recall":
            pred = 1 if pos_segments > 0 else 0
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        pt_pred_lbl.append(pred)

    pt_pred_lbl_arr = np.array(pt_pred_lbl, dtype=int)
    metrics = calculate_performance_metrics(pt_test_lbl, pt_pred_lbl_arr)
    vowel_name = vowel_type or "unknown"
    if strategy in ("fixed", "guding"):
        title = f"{dataset_type} Single Vowel /{vowel_name}/ (patient prob>={patient_prob_threshold})"
    else:
        title = f"{dataset_type} Single Vowel /{vowel_name}/ ({strategy})"
    if verbose:
        print(f"\nClassification report (single vowel): {strategy}")
        print(classification_report(pt_test_lbl, pt_pred_lbl_arr))
        print_performance_metrics(metrics, roc_auc, title)

    if show_plots:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues")
        plt.title(f"CM /{vowel_name}/ {strategy}")
        plt.close()

    return {
        "roc_auc": float(roc_auc),
        "roc_fpr": np.asarray(fpr, dtype=np.float64).tolist(),
        "roc_tpr": np.asarray(tpr, dtype=np.float64).tolist(),
        "roc_thresholds": np.asarray(thresholds, dtype=np.float64).tolist(),
        "classification_report": classification_report(pt_test_lbl, pt_pred_lbl_arr, output_dict=True),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "optimal_threshold": float(optimal_threshold) if strategy == "best_threshold" else None,
        "patient_prob_threshold": float(patient_prob_threshold) if strategy in ("fixed", "guding") else None,
        "metrics": metrics,
    }


def combined_vowel_ai_eval(
    segment_probs_a: np.ndarray,
    segment_probs_i: np.ndarray,
    y_a: np.ndarray,
    y_i: np.ndarray,
    ids_a,
    ids_i,
    *,
    strategy: Literal["best_threshold", "fixed", "relative", "percentage", "max recall", "guding"] = "fixed",
    segment_threshold: float = 0.5,
    patient_prob_threshold: float = 0.5,
    percentage: float = 0.2,
    dataset_type: str = "Test",
    show_plots: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Combine /a/ and /i/ segment scores: concatenate segments, group by patient, mean prob, threshold.
    """
    segment_probs_a = np.asarray(segment_probs_a).reshape(-1)
    segment_probs_i = np.asarray(segment_probs_i).reshape(-1)
    y_a = np.asarray(y_a).reshape(-1)
    y_i = np.asarray(y_i).reshape(-1)
    ids_a = np.asarray(ids_a).reshape(-1)
    ids_i = np.asarray(ids_i).reshape(-1)

    y_prob = np.concatenate([segment_probs_a, segment_probs_i])
    ids = np.concatenate([ids_a, ids_i])
    y = np.concatenate([y_a, y_i])

    patient_data: dict = {}
    for i, pid in enumerate(ids):
        segment_prob = float(y_prob[i])
        segment_pred = 1 if segment_prob >= segment_threshold else 0
        true_label = int(y[i])
        pid = _normalize_patient_id(pid)
        if pid not in patient_data:
            patient_data[pid] = {"probs": [], "true_label": true_label, "pos_segments": 0, "neg_segments": 0}
        d = patient_data[pid]
        d["probs"].append(segment_prob)
        if segment_pred == 1:
            d["pos_segments"] += 1
        else:
            d["neg_segments"] += 1

    for pid in patient_data:
        patient_data[pid]["avg_prob"] = float(np.mean(patient_data[pid]["probs"]))

    patient_ids = list(patient_data.keys())
    patient_prob = [patient_data[pid]["avg_prob"] for pid in patient_ids]
    patient_true = [int(patient_data[pid]["true_label"]) for pid in patient_ids]

    fpr, tpr, thresholds = roc_curve(patient_true, patient_prob, drop_intermediate=True)
    optimal_idx = int(np.argmax(tpr - fpr))
    optimal_threshold = float(thresholds[optimal_idx])

    patient_pred: list[int] = []
    for pid in patient_ids:
        data = patient_data[pid]
        prob = float(data["avg_prob"])
        pos_segments = int(data["pos_segments"])
        neg_segments = int(data["neg_segments"])
        total = pos_segments + neg_segments

        if strategy == "best_threshold":
            pred = 1 if prob >= optimal_threshold else 0
        elif strategy in ("fixed", "guding"):
            pred = 1 if prob >= patient_prob_threshold else 0
        elif strategy == "relative":
            pred = 1 if pos_segments >= neg_segments else 0
        elif strategy == "percentage":
            segment_pct = pos_segments / total if total > 0 else 0.0
            pred = 1 if segment_pct >= percentage else 0
        elif strategy == "max recall":
            pred = 1 if pos_segments > 0 else 0
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        patient_pred.append(pred)

    auc_score = float(roc_auc_score(patient_true, patient_prob))
    metrics = calculate_performance_metrics(np.asarray(patient_true), np.asarray(patient_pred))
    if strategy in ("fixed", "guding"):
        ctitle = f"{dataset_type} Combined /a+i/ (patient prob>={patient_prob_threshold})"
    else:
        ctitle = f"{dataset_type} Combined /a+i/ ({strategy})"
    if verbose:
        print(f"\nClassification report (combined): {strategy}")
        print(classification_report(patient_true, patient_pred))
        print_performance_metrics(metrics, auc_score, ctitle)

    if show_plots:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC AUC={auc_score:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()
        plt.close()

    return {
        "auc": auc_score,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": np.asarray(thresholds, dtype=np.float64).tolist(),
        "optimal_threshold": float(optimal_threshold) if strategy == "best_threshold" else None,
        "patient_prob_threshold": float(patient_prob_threshold) if strategy in ("fixed", "guding") else None,
        "classification_report": classification_report(patient_true, patient_pred, output_dict=True),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "metrics": metrics,
    }
