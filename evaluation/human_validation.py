"""Human vs LLM-as-Judge validation and agreement statistics (Spearman, Cohen's Kappa)."""

from typing import List, Dict, Any, Tuple
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def evaluate_judge_agreement(
    human_scores: List[float],
    judge_scores: List[float]
) -> Dict[str, float]:
    """Calculates Spearman rank correlation, Cohen's Kappa, and absolute agreement between Human and Judge."""
    if len(human_scores) != len(judge_scores) or len(human_scores) == 0:
        raise ValueError("Human and judge score lists must be non-empty and equal length.")

    # 1. Spearman Rank Correlation
    spearman_corr, p_value = spearmanr(human_scores, judge_scores)
    spearman_corr = float(spearman_corr) if not np.isnan(spearman_corr) else 0.0

    # 2. Score Difference Agreement (within 1 point on 12-point scale)
    diffs = np.abs(np.array(human_scores) - np.array(judge_scores))
    close_agreement = float(np.mean(diffs <= 1.0))
    exact_agreement = float(np.mean(diffs == 0.0))

    # 3. Cohen's Kappa on Binned Scores (e.g. Low [0-7], Medium [8-10], High [11-12])
    def bin_score(s: float) -> int:
        if s <= 7.0:
            return 0
        elif s <= 10.0:
            return 1
        return 2

    human_binned = [bin_score(s) for s in human_scores]
    judge_binned = [bin_score(s) for s in judge_scores]
    kappa = float(cohen_kappa_score(human_binned, judge_binned, weights="linear"))

    return {
        "spearman_correlation": round(spearman_corr, 4),
        "spearman_p_value": round(float(p_value), 6),
        "close_agreement_rate": round(close_agreement, 4),
        "exact_agreement_rate": round(exact_agreement, 4),
        "weighted_cohens_kappa": round(kappa, 4),
        "sample_size": len(human_scores)
    }
