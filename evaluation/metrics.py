"""Automated metrics calculation for Intent, Retrieval, and Escalation."""

from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.config import INTENT_LABELS


def compute_intent_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Computes Macro F1, Weighted F1, Accuracy, and per-class metrics."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, labels=INTENT_LABELS, zero_division=0)
    per_class = {intent: round(float(score), 4) for intent, score in zip(INTENT_LABELS, per_class_f1)}

    cm = confusion_matrix(y_true, y_pred, labels=INTENT_LABELS).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "macro_precision": round(float(precision_macro), 4),
        "macro_recall": round(float(recall_macro), 4),
        "per_class_f1": per_class,
        "confusion_matrix": cm,
        "labels": INTENT_LABELS
    }


def compute_retrieval_metrics(
    query_intents: List[str],
    retrieved_cases_list: List[List[Any]]
) -> Dict[str, float]:
    """Computes Recall@3, Recall@5, Recall@10, and MRR based on intent relevance of retrieved historical cases."""
    recalls_at_3 = []
    recalls_at_5 = []
    recalls_at_10 = []
    reciprocal_ranks = []

    for target_intent, retrieved in zip(query_intents, retrieved_cases_list):
        if not retrieved:
            recalls_at_3.append(0.0)
            recalls_at_5.append(0.0)
            recalls_at_10.append(0.0)
            reciprocal_ranks.append(0.0)
            continue

        retrieved_intents = [case.intent for case in retrieved]
        
        r3 = 1.0 if target_intent in retrieved_intents[:3] else 0.0
        r5 = 1.0 if target_intent in retrieved_intents[:5] else 0.0
        r10 = 1.0 if target_intent in retrieved_intents[:10] else 0.0

        rr = 0.0
        for rank, item_intent in enumerate(retrieved_intents, start=1):
            if item_intent == target_intent:
                rr = 1.0 / rank
                break

        recalls_at_3.append(r3)
        recalls_at_5.append(r5)
        recalls_at_10.append(r10)
        reciprocal_ranks.append(rr)

    return {
        "recall_at_3": round(float(np.mean(recalls_at_3)), 4),
        "recall_at_5": round(float(np.mean(recalls_at_5)), 4),
        "recall_at_10": round(float(np.mean(recalls_at_10)), 4),
        "mrr": round(float(np.mean(reciprocal_ranks)), 4)
    }


def compute_escalation_metrics(
    y_true_decision: List[str],
    y_pred_decision: List[str]
) -> Dict[str, float]:
    """Computes Precision, Recall, F1 for ESCALATE, and critical False Auto-Handle Rate.

    False Auto-Handle Rate = (Mistakenly AUTO_HANDLE when True was ESCALATE) / Total True ESCALATE.
    This is the most critical safety metric.
    """
    # Binary mapping: ESCALATE = 1, AUTO_HANDLE = 0
    true_bin = [1 if d == "ESCALATE" else 0 for d in y_true_decision]
    pred_bin = [1 if d == "ESCALATE" else 0 for d in y_pred_decision]

    prec = precision_score(true_bin, pred_bin, zero_division=0)
    rec = recall_score(true_bin, pred_bin, zero_division=0)
    f1 = f1_score(true_bin, pred_bin, zero_division=0)

    # Confusion matrix elements
    # TN: True Auto, Pred Auto
    # FP: True Auto, Pred Escalate (Unnecessary human review)
    # FN: True Escalate, Pred Auto (UNSAFE AUTO-HANDLE)
    # TP: True Escalate, Pred Escalate (Safe human review)
    tn, fp, fn, tp = confusion_matrix(true_bin, pred_bin, labels=[0, 1]).ravel()

    total_true_escalate = fn + tp
    false_auto_handle_rate = (fn / total_true_escalate) if total_true_escalate > 0 else 0.0
    auto_handle_rate = (tn + fn) / len(true_bin) if true_bin else 0.0

    return {
        "escalation_precision": round(float(prec), 4),
        "escalation_recall": round(float(rec), 4),
        "escalation_f1": round(float(f1), 4),
        "false_auto_handle_rate": round(float(false_auto_handle_rate), 4),
        "automation_rate": round(float(auto_handle_rate), 4),
        "unsafe_auto_handled_count": int(fn),
        "safe_auto_handled_count": int(tn),
        "correct_escalated_count": int(tp),
        "unnecessary_escalated_count": int(fp)
    }
