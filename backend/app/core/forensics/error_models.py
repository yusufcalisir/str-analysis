"""
Stochastic Error Models — VANTAGE-STR Phase 3.7.

Models the three primary sources of forensic STR uncertainty:
    1. Allele Dropout (Pr(D)) — Probability that a true allele was not detected.
    2. Stutter Artifacts — n±1 repeat products of PCR amplification.
    3. Profile Degradation — Overall quality assessment driving prior attenuation.

These models adjust per-locus Likelihood Ratios to prevent false certainty
on low-quality, partial, or contaminated profiles.

Reference: Gill et al. (2012), ISFG (2020), SWGDAM Interpretation Guidelines.
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Logistic curve parameters for dropout probability
_DROPOUT_K: float = 8.0          # Steepness of the logistic curve
_DROPOUT_MIDPOINT: float = 0.5   # Midpoint (quality = 0.5 → Pr(D) ≈ 0.5)

# Stutter ratio threshold (typical STR stutter is 5–15% of parent peak)
STUTTER_RATIO_THRESHOLD: float = 0.15
# Maximum repeat unit difference to flag as potential stutter
STUTTER_REPEAT_DELTA: int = 1

# Quality tier thresholds
QUALITY_MAP: Dict[str, float] = {
    "pristine": 0.02,    # Buccal swab, fresh blood
    "moderate": 0.15,    # Aged but intact
    "severe":   0.45,    # Decomposed, burned, old
    "trace":    0.70,    # Touch DNA, skin cells
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DropoutAssessment:
    """Dropout probability assessment for a single locus."""
    marker: str
    dropout_probability: float       # Pr(D) — 0.0 to 1.0
    is_flagged: bool                 # True if Pr(D) > 0.05
    adjustment_factor: float         # Multiplier for the LR at this locus
    message: str = ""


@dataclass
class StutterAssessment:
    """Stutter artifact assessment for a single locus."""
    marker: str
    stutter_probability: float       # 0.0 to 1.0
    is_flagged: bool
    suspect_allele: Optional[float] = None   # The allele suspected as stutter
    parent_allele: Optional[float] = None    # The allele suspected as parent
    adjustment_factor: float = 1.0
    message: str = ""


@dataclass
class DegradationReport:
    """Complete degradation assessment for a profile."""
    degradation_index: float         # 0.0 (pristine) to 1.0 (fully degraded)
    quality_tier: str                # pristine / moderate / severe / trace
    total_loci: int = 0
    missing_loci: int = 0
    dropout_count: int = 0
    stutter_count: int = 0
    dropout_assessments: List[DropoutAssessment] = field(default_factory=list)
    stutter_assessments: List[StutterAssessment] = field(default_factory=list)
    dropout_warnings: List[str] = field(default_factory=list)
    stutter_warnings: List[str] = field(default_factory=list)
    lr_cap: Optional[float] = None   # Maximum LR allowed given degradation


# ═══════════════════════════════════════════════════════════════════════════════
# DROPOUT PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_dropout_probability(
    quality_tier: str,
    is_homozygous: bool,
    allele_freq: float = 0.1,
) -> float:
    """
    Compute allele dropout probability using a logistic model.

    Pr(D) = 1 / (1 + e^(-k * (degradation - midpoint)))

    Heterozygotes have higher dropout risk than homozygotes since
    losing one of two different alleles is more impactful.

    Args:
        quality_tier: One of 'pristine', 'moderate', 'severe', 'trace'.
        is_homozygous: Whether the locus is homozygous.
        allele_freq: Frequency of the allele (rarer alleles have slightly
                     higher dropout risk in degraded samples).

    Returns:
        Dropout probability Pr(D) in [0, 1].
    """
    base_degradation = QUALITY_MAP.get(quality_tier, 0.15)

    # Rarer alleles have marginally higher dropout risk
    rarity_bonus = max(0.0, (0.05 - allele_freq) * 0.5)
    effective_degradation = min(1.0, base_degradation + rarity_bonus)

    # Logistic function
    exponent = -_DROPOUT_K * (effective_degradation - _DROPOUT_MIDPOINT)
    pr_d = 1.0 / (1.0 + math.exp(exponent))

    # Homozygotes are less affected (both copies of same allele)
    if is_homozygous:
        pr_d *= 0.3

    return round(min(pr_d, 0.95), 6)


def compute_dropout_lr_adjustment(pr_d: float) -> float:
    """
    Compute the LR adjustment factor given dropout probability.

    When dropout is possible, we must account for the probability
    that the observed genotype is not the true genotype:

        LR_adjusted = LR * (1 - Pr(D)) + Pr(D) * LR_dropout_scenario

    Simplified: we attenuate the LR by (1 - Pr(D)) as the dropout
    scenario contributes ~1 to the LR (uninformative).

    Returns:
        Adjustment factor in (0, 1].
    """
    return max(0.05, 1.0 - pr_d)


def assess_dropout(
    marker: str,
    is_homozygous: bool,
    allele_freq: float,
    quality_tier: str,
) -> DropoutAssessment:
    """Compute full dropout assessment for a single locus."""
    pr_d = compute_dropout_probability(quality_tier, is_homozygous, allele_freq)
    adjustment = compute_dropout_lr_adjustment(pr_d)
    flagged = pr_d > 0.05

    message = ""
    if flagged:
        message = (
            f"Dropout risk at {marker}: Pr(D) = {pr_d:.4f}. "
            f"{'Homozygous locus — reduced risk.' if is_homozygous else 'Heterozygous locus — elevated risk.'} "
            f"LR adjusted by factor {adjustment:.4f}."
        )

    return DropoutAssessment(
        marker=marker,
        dropout_probability=pr_d,
        is_flagged=flagged,
        adjustment_factor=adjustment,
        message=message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STUTTER SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_stutter(
    marker: str,
    allele_1: float,
    allele_2: float,
    freq_1: float,
    freq_2: float,
) -> StutterAssessment:
    """
    Detect if an allele pair contains a potential stutter artifact.

    Stutter products are n-1 (back-stutter) or n+1 (forward-stutter) repeat
    units from the true allele. A stutter is suspected when:
        1. |allele_1 - allele_2| == 1 (single repeat unit apart).
        2. One allele is significantly rarer (consistent with artifact).

    The stutter probability is derived from the frequency ratio:
        Pr(stutter) = min(freq_minor / freq_major, STUTTER_RATIO_THRESHOLD)
                      / STUTTER_RATIO_THRESHOLD

    Args:
        marker: Locus name.
        allele_1, allele_2: Observed alleles.
        freq_1, freq_2: Population frequencies of the alleles.

    Returns:
        StutterAssessment with probability and flagging.
    """
    # Only integer-repeat markers can produce classic stutter
    delta = abs(allele_1 - allele_2)

    # Check for n±1 pattern (integer repeat difference of 1)
    if delta < 0.5 or delta > 1.5:
        return StutterAssessment(marker=marker, stutter_probability=0.0, is_flagged=False)

    # Identify suspect (rarer) and parent (more common) alleles
    if freq_1 <= freq_2:
        suspect, parent = allele_1, allele_2
        freq_suspect, freq_parent = freq_1, freq_2
    else:
        suspect, parent = allele_2, allele_1
        freq_suspect, freq_parent = freq_2, freq_1

    # Stutter ratio: how much rarer is the suspect vs the parent
    if freq_parent <= 0:
        return StutterAssessment(marker=marker, stutter_probability=0.0, is_flagged=False)

    ratio = freq_suspect / freq_parent

    if ratio >= STUTTER_RATIO_THRESHOLD * 3:
        # Both alleles are similarly common — unlikely stutter
        return StutterAssessment(marker=marker, stutter_probability=0.0, is_flagged=False)

    # Normalize ratio to a probability
    stutter_prob = min(1.0, max(0.0, 1.0 - (ratio / (STUTTER_RATIO_THRESHOLD * 3))))

    # Apply a base penalty — even clearly separated alleles at n±1 get light scrutiny
    stutter_prob = stutter_prob * 0.7  # Scale down — stutter is relatively uncommon

    flagged = stutter_prob > 0.15
    adjustment = max(0.1, 1.0 - stutter_prob * 0.8) if flagged else 1.0

    message = ""
    if flagged:
        message = (
            f"Potential stutter at {marker}: allele {suspect} may be artifact of {parent}. "
            f"Pr(stutter) = {stutter_prob:.4f}. "
            f"LR penalty factor: {adjustment:.4f}."
        )

    return StutterAssessment(
        marker=marker,
        stutter_probability=round(stutter_prob, 6),
        is_flagged=flagged,
        suspect_allele=suspect if flagged else None,
        parent_allele=parent if flagged else None,
        adjustment_factor=adjustment,
        message=message,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DEGRADATION INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_degradation_index(
    quality_tier: str,
    total_loci: int,
    expected_loci: int = 20,
    dropout_count: int = 0,
    stutter_count: int = 0,
) -> float:
    """
    Compute an overall degradation index for the profile.

    Combines:
        1. Base quality tier degradation (40% weight)
        2. Missing loci ratio (30% weight)
        3. Dropout and stutter counts (30% weight)

    Returns:
        Degradation index in [0.0, 1.0].
    """
    base = QUALITY_MAP.get(quality_tier, 0.15)

    # Missing loci component
    missing_ratio = max(0.0, 1.0 - total_loci / max(expected_loci, 1))

    # Error component
    error_ratio = min(1.0, (dropout_count + stutter_count) / max(total_loci, 1))

    index = (base * 0.4) + (missing_ratio * 0.3) + (error_ratio * 0.3)
    return round(min(1.0, max(0.0, index)), 4)


def compute_lr_cap(degradation_index: float) -> Optional[float]:
    """
    Compute an LR cap for heavily degraded profiles.

    Prevents false 100% certainty by capping the maximum Combined LR
    when degradation is significant:
        - index < 0.3: No cap (pristine profiles)
        - index 0.3–0.5: Cap at 10^15
        - index 0.5–0.7: Cap at 10^9
        - index > 0.7: Cap at 10^6

    Returns:
        Maximum LR allowed, or None if no cap.
    """
    if degradation_index < 0.3:
        return None
    elif degradation_index < 0.5:
        return 1e15
    elif degradation_index < 0.7:
        return 1e9
    else:
        return 1e6


# ═══════════════════════════════════════════════════════════════════════════════
# FULL PROFILE ERROR ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def assess_profile_errors(
    markers: Dict[str, Tuple[float, float]],
    frequencies: Dict[str, Tuple[float, float]],
    homozygosity: Dict[str, bool],
    quality_tier: str = "moderate",
    expected_loci: int = 20,
) -> DegradationReport:
    """
    Run full stochastic error assessment on a profile.

    Args:
        markers: Dict of marker → (allele_1, allele_2).
        frequencies: Dict of marker → (freq_1, freq_2).
        homozygosity: Dict of marker → is_homozygous.
        quality_tier: Quality assessment string.
        expected_loci: Expected number of loci for a complete profile.

    Returns:
        DegradationReport with all assessments and warnings.
    """
    dropout_assessments: List[DropoutAssessment] = []
    stutter_assessments: List[StutterAssessment] = []
    dropout_warnings: List[str] = []
    stutter_warnings: List[str] = []

    for marker in markers:
        a1, a2 = markers[marker]
        f1, f2 = frequencies.get(marker, (0.08, 0.08))
        is_homo = homozygosity.get(marker, a1 == a2)

        # Dropout assessment
        avg_freq = (f1 + f2) / 2
        dropout = assess_dropout(marker, is_homo, avg_freq, quality_tier)
        dropout_assessments.append(dropout)
        if dropout.is_flagged:
            dropout_warnings.append(dropout.message)

        # Stutter assessment
        stutter = detect_stutter(marker, a1, a2, f1, f2)
        stutter_assessments.append(stutter)
        if stutter.is_flagged:
            stutter_warnings.append(stutter.message)

    dropout_count = sum(1 for d in dropout_assessments if d.is_flagged)
    stutter_count = sum(1 for s in stutter_assessments if s.is_flagged)
    total_loci = len(markers)

    deg_index = compute_degradation_index(
        quality_tier, total_loci, expected_loci, dropout_count, stutter_count
    )
    lr_cap = compute_lr_cap(deg_index)

    logger.info(
        f"[ERROR-MODEL] Profile assessed: quality={quality_tier}, "
        f"degradation={deg_index:.4f}, "
        f"dropouts={dropout_count}/{total_loci}, "
        f"stutters={stutter_count}/{total_loci}, "
        f"lr_cap={lr_cap}"
    )

    return DegradationReport(
        degradation_index=deg_index,
        quality_tier=quality_tier,
        total_loci=total_loci,
        missing_loci=max(0, expected_loci - total_loci),
        dropout_count=dropout_count,
        stutter_count=stutter_count,
        dropout_assessments=dropout_assessments,
        stutter_assessments=stutter_assessments,
        dropout_warnings=dropout_warnings,
        stutter_warnings=stutter_warnings,
        lr_cap=lr_cap,
    )
