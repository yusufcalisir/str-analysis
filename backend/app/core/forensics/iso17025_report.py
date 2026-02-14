"""
ISO 17025 Verbal Scale Report Generator — VANTAGE-STR Phase 3.7.

Generates standardized forensic verbal equivalence reports following
international guidelines (ISFG 2021, ENFSI 2015) for court-admissible
DNA evidence evaluation.

The verbal scale maps Bayesian posterior probabilities to human-readable
strength-of-evidence categories, with automatic downgrading for degraded
or error-prone profiles.

Reference: ISFG DNA Commission (2021), ENFSI Guideline (2015), NRC II (1996).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VERBAL EQUIVALENCE SCALE — BAYESIAN POSTERIOR
# ═══════════════════════════════════════════════════════════════════════════════

# Maps posterior P(Hp|E) thresholds to verbal descriptions.
# Ordered from strongest to weakest.
BAYESIAN_VERBAL_SCALE = [
    (0.99999, "IDENTIFICATION_PRACTICALLY_PROVEN",
     "The forensic evidence practically proves that the evidential and reference "
     "profiles originate from the same individual."),
    (0.9999,  "EXTREMELY_STRONG_SUPPORT",
     "The evidence provides extremely strong support for the proposition that "
     "the profiles originate from the same source."),
    (0.999,   "VERY_STRONG_SUPPORT",
     "The evidence provides very strong support for the prosecution hypothesis."),
    (0.99,    "STRONG_SUPPORT",
     "The evidence provides strong support for the prosecution hypothesis. "
     "Secondary verification with additional markers is recommended."),
    (0.9,     "MODERATELY_STRONG_SUPPORT",
     "The evidence provides moderately strong support. Additional testing "
     "or expanded loci panels should be considered."),
    (0.5,     "LIMITED_SUPPORT",
     "The evidence provides limited support for the prosecution hypothesis. "
     "The result is not sufficient for identification on its own."),
    (0.0,     "INCONCLUSIVE",
     "The evidence is inconclusive and does not discriminate between the "
     "prosecution and defense hypotheses."),
]

# LR-based scale (fallback when posterior is not available)
LR_VERBAL_SCALE = [
    (1e15, "IDENTIFICATION_PRACTICALLY_PROVEN"),
    (1e12, "EXTREMELY_STRONG_SUPPORT"),
    (1e9,  "VERY_STRONG_SUPPORT"),
    (1e6,  "STRONG_SUPPORT"),
    (1e4,  "MODERATELY_STRONG_SUPPORT"),
    (1e2,  "MODERATE_SUPPORT"),
    (1e1,  "LIMITED_SUPPORT"),
    (1e0,  "INCONCLUSIVE"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ISO17025Verdict:
    """Complete ISO 17025-compliant verbal scale verdict."""
    verbal_scale: str
    verbal_description: str
    numerical_basis: str
    posterior_hp: float
    prior_used: float
    combined_lr: float
    degradation_index: float
    was_downgraded: bool = False
    downgrade_reason: str = ""
    limitations: List[str] = field(default_factory=list)
    methodology: str = (
        "Likelihood Ratio framework per ISFG 2021 guidelines. "
        "Per-locus LR computed using the Balding-Nichols NRC II Recommendation 4.4 "
        "with population substructure correction (θ = 0.01). "
        "Combined via product rule assuming locus independence. "
        "Bayesian posterior computed using dynamic prior based on suspect pool size. "
        "Error-adjusted for allele dropout and stutter artifacts."
    )

    def to_dict(self) -> Dict:
        return {
            "verbal_scale": self.verbal_scale,
            "verbal_description": self.verbal_description,
            "numerical_basis": self.numerical_basis,
            "posterior_hp": round(self.posterior_hp, 8),
            "prior_used": self.prior_used,
            "combined_lr": self.combined_lr,
            "degradation_index": round(self.degradation_index, 4),
            "was_downgraded": self.was_downgraded,
            "downgrade_reason": self.downgrade_reason,
            "limitations": self.limitations,
            "methodology": self.methodology,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VERBAL SCALE COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def classify_verbal_from_posterior(posterior_hp: float) -> tuple:
    """
    Classify the verbal equivalence from the Bayesian posterior.

    Returns:
        (verbal_label, verbal_description)
    """
    for threshold, label, description in BAYESIAN_VERBAL_SCALE:
        if posterior_hp >= threshold:
            return label, description

    return "INCONCLUSIVE", BAYESIAN_VERBAL_SCALE[-1][2]


def classify_verbal_from_lr(combined_lr: float) -> str:
    """Classify verbal equivalence from the Combined LR (fallback)."""
    for threshold, label in LR_VERBAL_SCALE:
        if combined_lr >= threshold:
            return label
    return "INCONCLUSIVE"


def should_downgrade(
    verbal: str,
    degradation_index: float,
    dropout_count: int = 0,
    stutter_count: int = 0,
    total_loci: int = 20,
) -> tuple:
    """
    Determine if the verbal scale should be downgraded due to quality issues.

    Rules:
        1. degradation_index > 0.5 → downgrade by 2 levels
        2. degradation_index > 0.3 → downgrade by 1 level
        3. dropout_count > total_loci * 0.3 → downgrade by 1 level
        4. stutter_count > total_loci * 0.2 → downgrade by 1 level

    Returns:
        (downgraded_verbal, was_downgraded, reason)
    """
    labels = [item[0] for item in BAYESIAN_VERBAL_SCALE]

    if verbal not in labels:
        # Try matching without the full scale — use position 0
        current_idx = 0
    else:
        current_idx = labels.index(verbal)

    steps_down = 0
    reasons = []

    if degradation_index > 0.5:
        steps_down += 2
        reasons.append(f"High degradation index ({degradation_index:.3f} > 0.5)")
    elif degradation_index > 0.3:
        steps_down += 1
        reasons.append(f"Elevated degradation index ({degradation_index:.3f} > 0.3)")

    if total_loci > 0:
        if dropout_count > total_loci * 0.3:
            steps_down += 1
            reasons.append(f"Excessive dropouts ({dropout_count}/{total_loci})")
        if stutter_count > total_loci * 0.2:
            steps_down += 1
            reasons.append(f"Multiple stutter artifacts ({stutter_count}/{total_loci})")

    if steps_down == 0:
        return verbal, False, ""

    new_idx = min(current_idx + steps_down, len(labels) - 1)
    return labels[new_idx], True, "; ".join(reasons)


# ═══════════════════════════════════════════════════════════════════════════════
# VERDICT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_iso17025_verdict(
    posterior_hp: float,
    prior_used: float,
    combined_lr: float,
    degradation_index: float,
    dropout_count: int = 0,
    stutter_count: int = 0,
    total_loci: int = 20,
    population: str = "European",
) -> ISO17025Verdict:
    """
    Generate a complete ISO 17025-compliant verbal scale verdict.

    Args:
        posterior_hp: Bayesian posterior P(Hp|E).
        prior_used: Prior probability P(Hp) used.
        combined_lr: Combined Likelihood Ratio.
        degradation_index: Profile degradation index (0–1).
        dropout_count: Number of loci with dropout warnings.
        stutter_count: Number of loci with stutter warnings.
        total_loci: Total loci analyzed.
        population: Population used for frequency lookup.

    Returns:
        ISO17025Verdict with complete verbal report.
    """
    # Classify from posterior
    verbal, description = classify_verbal_from_posterior(posterior_hp)

    # Check for downgrade
    downgraded_verbal, was_downgraded, reason = should_downgrade(
        verbal, degradation_index, dropout_count, stutter_count, total_loci
    )

    if was_downgraded:
        verbal = downgraded_verbal
        _, description = classify_verbal_from_posterior(
            # Find the threshold for the downgraded level
            next(
                (t for t, l, _ in BAYESIAN_VERBAL_SCALE if l == downgraded_verbal),
                0.0
            )
        )

    # Build limitations
    limitations = [
        f"Population frequency data: {population} reference panel (NIST STRBase)",
        "Results depend on the completeness of the STR panel used",
        "This analysis does not constitute a legal identification",
    ]

    if degradation_index > 0.3:
        limitations.insert(0,
            f"Profile degradation detected (index: {degradation_index:.3f}). "
            f"Results should be interpreted with caution."
        )

    if dropout_count > 0:
        limitations.append(
            f"Allele dropout suspected at {dropout_count} locus/loci. "
            f"LR values have been adjusted to account for missing alleles."
        )

    if stutter_count > 0:
        limitations.append(
            f"Stutter artifacts detected at {stutter_count} locus/loci. "
            f"Affected loci have reduced evidential weight."
        )

    # Numerical basis summary
    numerical_basis = (
        f"P(Hp|E) = {posterior_hp:.6f}, "
        f"Prior P(Hp) = {prior_used:.2e}, "
        f"Combined LR = {combined_lr:.2e}, "
        f"Degradation = {degradation_index:.3f}"
    )

    verdict = ISO17025Verdict(
        verbal_scale=verbal,
        verbal_description=description,
        numerical_basis=numerical_basis,
        posterior_hp=posterior_hp,
        prior_used=prior_used,
        combined_lr=combined_lr,
        degradation_index=degradation_index,
        was_downgraded=was_downgraded,
        downgrade_reason=reason,
        limitations=limitations,
    )

    logger.info(
        f"[ISO-17025] Verdict: {verbal} "
        f"(posterior={posterior_hp:.6f}, downgraded={was_downgraded})"
    )

    return verdict
