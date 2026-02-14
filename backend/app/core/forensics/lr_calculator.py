"""
Likelihood Ratio Calculator — VANTAGE-STR Phase 3.7.

Computes forensic Likelihood Ratios using real population allele frequencies
from the knowledge base. Implements the Balding-Nichols model as specified
in NRC II Recommendation 4.4 for population substructure correction.

Phase 3.7 additions:
    - Bayesian posterior P(Hp|E) with dynamic priors based on suspect pool.
    - Error-adjusted per-locus LR (dropout + stutter models).
    - 95% HPD confidence intervals via log-normal approximation.
    - ISO 17025 verbal scale with automatic degradation downgrading.
    - LR capping for degraded/partial profiles.

Formulas (NRC II 4.4 — Balding-Nichols):
    Homozygous locus (allele pᵢ):
        P(AᵢAᵢ) = (2θ + (1-θ)pᵢ)(3θ + (1-θ)pᵢ) / ((1+θ)(1+2θ))

    Heterozygous locus (alleles pᵢ, pⱼ):
        P(AᵢAⱼ) = 2(θ + (1-θ)pᵢ)(θ + (1-θ)pⱼ) / ((1+θ)(1+2θ))

    Per-locus LR = 1 / P(genotype)
    Combined LR (CLR) = Π(per-locus LR)
    Random Match Probability (RMP) = 1 / CLR

    Bayesian Posterior:
        P(Hp|E) = LR × P(Hp) / (LR × P(Hp) + P(Hd))

Reference: NRC II (1996), ISFG (2021), Balding & Nichols (1994),
           Gill et al. (2012), ENFSI (2015).
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.core.forensics.population_data import (
    get_frequency,
    get_average_frequency,
    CODIS_LOCI,
    MIN_FREQUENCY,
)
from app.core.forensics.error_models import (
    assess_profile_errors,
    DegradationReport,
)
from app.core.forensics.iso17025_report import (
    generate_iso17025_verdict,
    classify_verbal_from_lr,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LocusLRDetail:
    """LR computation result for a single locus."""
    marker: str
    allele_1: float
    allele_2: float
    is_homozygous: bool
    freq_1: float
    freq_2: float
    genotype_probability: float
    individual_lr: float
    log10_lr: float
    rarity_score: float  # 0.0 (common) to 1.0 (extremely rare)
    # Phase 3.7 — Error-adjusted fields
    dropout_probability: float = 0.0
    stutter_probability: float = 0.0
    adjusted_lr: float = 0.0  # LR after error adjustment
    sensitivity_contribution: float = 0.0  # % of total log₁₀(CLR) from this locus

    def to_dict(self) -> Dict:
        return {
            "marker": self.marker,
            "alleles": [self.allele_1, self.allele_2],
            "is_homozygous": self.is_homozygous,
            "frequencies": [round(self.freq_1, 6), round(self.freq_2, 6)],
            "genotype_probability": self.genotype_probability,
            "individual_lr": self.individual_lr,
            "log10_lr": round(self.log10_lr, 2),
            "rarity_score": round(self.rarity_score, 4),
            "dropout_probability": round(self.dropout_probability, 6),
            "stutter_probability": round(self.stutter_probability, 6),
            "adjusted_lr": self.adjusted_lr,
            "sensitivity_contribution": round(self.sensitivity_contribution, 2),
        }


@dataclass
class LRResult:
    """Complete LR computation result."""
    combined_lr: float = 0.0
    log10_lr: float = 0.0
    random_match_probability: float = 1.0
    random_match_probability_str: str = "1 in 1"
    verbal_equivalence: str = "INCONCLUSIVE"
    prosecution_probability: float = 0.0
    defense_probability: float = 1.0
    population_used: str = "European"
    loci_analyzed: int = 0
    per_locus_details: List[LocusLRDetail] = field(default_factory=list)
    high_frequency_warning: bool = False
    warning_message: str = ""
    # Phase 3.7 — Bayesian inference fields
    posterior_hp: float = 0.0           # Bayesian posterior P(Hp|E)
    posterior_hd: float = 1.0           # P(Hd|E) = 1 - P(Hp|E)
    prior_hp: float = 0.5              # Prior P(Hp) used
    bayesian_ci_lower: float = 0.0     # 95% HPD lower bound
    bayesian_ci_upper: float = 0.0     # 95% HPD upper bound
    degradation_index: float = 0.0     # Profile quality metric
    dropout_warnings: List[str] = field(default_factory=list)
    stutter_warnings: List[str] = field(default_factory=list)
    iso17025_verbal: str = "INCONCLUSIVE"
    sensitivity_map: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "combined_lr": self.combined_lr,
            "log10_lr": round(self.log10_lr, 2),
            "random_match_probability": self.random_match_probability,
            "random_match_probability_str": self.random_match_probability_str,
            "verbal_equivalence": self.verbal_equivalence,
            "prosecution_probability": round(self.prosecution_probability, 8),
            "defense_probability": round(self.defense_probability, 8),
            "population_used": self.population_used,
            "loci_analyzed": self.loci_analyzed,
            "per_locus_details": [d.to_dict() for d in self.per_locus_details],
            "high_frequency_warning": self.high_frequency_warning,
            "warning_message": self.warning_message,
            "posterior_hp": round(self.posterior_hp, 8),
            "posterior_hd": round(self.posterior_hd, 8),
            "prior_hp": self.prior_hp,
            "bayesian_ci_lower": round(self.bayesian_ci_lower, 8),
            "bayesian_ci_upper": round(self.bayesian_ci_upper, 8),
            "degradation_index": round(self.degradation_index, 4),
            "dropout_warnings": self.dropout_warnings,
            "stutter_warnings": self.stutter_warnings,
            "iso17025_verbal": self.iso17025_verbal,
            "sensitivity_map": self.sensitivity_map,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VERBAL EQUIVALENCE SCALE (ISFG 2021) — Legacy, now in iso17025_report.py
# ═══════════════════════════════════════════════════════════════════════════════

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
# LR CALCULATOR — Balding-Nichols NRC II 4.4
# ═══════════════════════════════════════════════════════════════════════════════

# Population substructure correction factor (Balding-Nichols θ)
# θ = 0.01 for well-characterized populations (SWGDAM default)
# θ = 0.03 for isolated/endogamous populations
THETA: float = 0.01

# Rarity threshold
HIGH_FREQ_THRESHOLD: float = 0.15


def compute_genotype_probability(
    allele_1: float,
    allele_2: float,
    marker: str,
    population: str = "European",
    theta: float = THETA,
) -> Tuple[float, float, float, bool]:
    """
    Compute genotype probability using NRC II Recommendation 4.4
    (Balding-Nichols model for population substructure).

    Homozygous AᵢAᵢ:
        P = (2θ + (1-θ)pᵢ)(3θ + (1-θ)pᵢ) / ((1+θ)(1+2θ))

    Heterozygous AᵢAⱼ:
        P = 2(θ + (1-θ)pᵢ)(θ + (1-θ)pⱼ) / ((1+θ)(1+2θ))

    Returns:
        (genotype_probability, freq_1, freq_2, is_homozygous)
    """
    freq_1 = get_frequency(marker, allele_1, population)
    freq_2 = get_frequency(marker, allele_2, population)

    is_homozygous = allele_1 == allele_2

    # NRC II 4.4 denominator: (1+θ)(1+2θ)
    denominator = (1 + theta) * (1 + 2 * theta)

    if is_homozygous:
        # P(AᵢAᵢ) = (2θ + (1-θ)pᵢ)(3θ + (1-θ)pᵢ) / ((1+θ)(1+2θ))
        term_a = 2 * theta + (1 - theta) * freq_1
        term_b = 3 * theta + (1 - theta) * freq_1
        prob = (term_a * term_b) / denominator
    else:
        # P(AᵢAⱼ) = 2(θ + (1-θ)pᵢ)(θ + (1-θ)pⱼ) / ((1+θ)(1+2θ))
        term_a = theta + (1 - theta) * freq_1
        term_b = theta + (1 - theta) * freq_2
        prob = (2 * term_a * term_b) / denominator

    return prob, freq_1, freq_2, is_homozygous


def compute_rarity_score(freq_1: float, freq_2: float, marker: str, population: str) -> float:
    """
    Compute a rarity score for the allele combination at this locus.

    0.0 = completely common (both alleles are the population mode)
    1.0 = extremely rare (both alleles are unobserved/near minimum)

    Uses log-scale comparison against the population's average frequency.
    """
    avg_freq = get_average_frequency(marker, population)
    if avg_freq <= 0:
        return 1.0

    # Geometric mean of allele frequencies, compared to average
    combined = (freq_1 * freq_2) ** 0.5
    ratio = combined / avg_freq

    # Invert and clamp: ratio < 1 means rarer than average
    if ratio >= 1.0:
        return 0.0
    elif ratio <= 0.01:
        return 1.0
    else:
        # Log-scale mapping for intuitive gradient
        rarity = -math.log10(ratio) / 2.0
        return min(1.0, max(0.0, rarity))


def compute_per_locus_lr(
    allele_1: float,
    allele_2: float,
    marker: str,
    population: str = "European",
) -> LocusLRDetail:
    """
    Compute the Likelihood Ratio for a single STR locus.

    LR = 1 / P(genotype | population)
    """
    prob, freq_1, freq_2, is_homozygous = compute_genotype_probability(
        allele_1, allele_2, marker, population
    )

    lr = 1.0 / max(prob, 1e-30)
    log10_lr = math.log10(max(lr, 1e-300))
    rarity = compute_rarity_score(freq_1, freq_2, marker, population)

    return LocusLRDetail(
        marker=marker,
        allele_1=allele_1,
        allele_2=allele_2,
        is_homozygous=is_homozygous,
        freq_1=freq_1,
        freq_2=freq_2,
        genotype_probability=prob,
        individual_lr=lr,
        log10_lr=log10_lr,
        rarity_score=rarity,
    )


def _compute_hpd_interval(
    posterior: float,
    log10_lr: float,
    n_loci: int,
) -> Tuple[float, float]:
    """
    Compute 95% Highest Posterior Density interval via log-normal approximation.

    Uses the variance of log₁₀(per-locus LR) to estimate spread.
    The posterior uncertainty shrinks with more loci (√n effect).

    Returns:
        (lower_bound, upper_bound) of the 95% HPD on the posterior scale.
    """
    if posterior <= 0 or posterior >= 1 or n_loci <= 0:
        return (0.0, min(1.0, posterior + 0.01))

    # Standard error on the log-odds scale
    # Approximate: SE(log₁₀ LR) ≈ |log₁₀ LR| / (√n_loci × 2)
    se_log_lr = max(0.5, abs(log10_lr)) / (math.sqrt(n_loci) * 2)

    # Convert to posterior scale using logistic transformation
    log_odds = math.log(max(posterior, 1e-15) / max(1 - posterior, 1e-15))
    se_posterior = se_log_lr * math.log(10)  # Convert log₁₀ SE to natural log SE

    # 95% HPD: ±1.96 SE on log-odds, then transform back
    lower_log_odds = log_odds - 1.96 * se_posterior
    upper_log_odds = log_odds + 1.96 * se_posterior

    lower = 1.0 / (1.0 + math.exp(-lower_log_odds))
    upper = 1.0 / (1.0 + math.exp(-upper_log_odds))

    return (round(max(0.0, lower), 8), round(min(1.0, upper), 8))


def compute_combined_lr(
    str_markers: Dict[str, Tuple[float, float]],
    population: str = "European",
    sample_quality: str = "pristine",
    suspect_pool_size: int = 1_000_000,
) -> LRResult:
    """
    Compute the Combined Likelihood Ratio (CLR) across all provided loci
    with Bayesian posterior inference and stochastic error adjustment.

    Phase 3.7 Pipeline:
        1. Compute raw per-locus LR (Balding-Nichols).
        2. Run stochastic error assessment (dropout + stutter).
        3. Adjust per-locus LR for detected errors.
        4. Combine via product rule → CLR.
        5. Apply degradation LR cap if needed.
        6. Compute Bayesian posterior: P(Hp|E) = LR·P(Hp) / (LR·P(Hp) + P(Hd))
        7. Compute 95% HPD confidence interval.
        8. Generate ISO 17025 verbal verdict.

    Args:
        str_markers: Dict mapping marker name to (allele_1, allele_2) tuple.
        population: Population group for frequency lookup.
        sample_quality: One of 'pristine', 'moderate', 'severe', 'trace'.
        suspect_pool_size: Size of the relevant suspect population.

    Returns:
        LRResult with complete Bayesian statistical breakdown.
    """
    per_locus_details: List[LocusLRDetail] = []

    for marker, (a1, a2) in str_markers.items():
        if marker == "AMEL":
            continue  # Skip sex marker for LR computation

        detail = compute_per_locus_lr(a1, a2, marker, population)
        per_locus_details.append(detail)

    if not per_locus_details:
        return LRResult(population_used=population)

    # ── Phase 3.7: Stochastic Error Assessment ─────────────────────────
    markers_dict = {d.marker: (d.allele_1, d.allele_2) for d in per_locus_details}
    freqs_dict = {d.marker: (d.freq_1, d.freq_2) for d in per_locus_details}
    homo_dict = {d.marker: d.is_homozygous for d in per_locus_details}

    error_report: DegradationReport = assess_profile_errors(
        markers=markers_dict,
        frequencies=freqs_dict,
        homozygosity=homo_dict,
        quality_tier=sample_quality,
        expected_loci=20,
    )

    # Build lookup maps for per-locus error data
    dropout_map = {a.marker: a for a in error_report.dropout_assessments}
    stutter_map = {a.marker: a for a in error_report.stutter_assessments}

    # ── Apply error adjustments to per-locus LRs ──────────────────────
    for detail in per_locus_details:
        dropout = dropout_map.get(detail.marker)
        stutter = stutter_map.get(detail.marker)

        adj_factor = 1.0
        if dropout:
            detail.dropout_probability = dropout.dropout_probability
            adj_factor *= dropout.adjustment_factor
        if stutter:
            detail.stutter_probability = stutter.stutter_probability
            adj_factor *= stutter.adjustment_factor

        detail.adjusted_lr = detail.individual_lr * adj_factor

    # ── Combined LR using error-adjusted values ───────────────────────
    combined_lr = 1.0
    for detail in per_locus_details:
        combined_lr *= detail.adjusted_lr

    # Apply degradation cap
    if error_report.lr_cap is not None and combined_lr > error_report.lr_cap:
        logger.warning(
            f"[LR-CALC] CLR capped from {combined_lr:.2e} to {error_report.lr_cap:.2e} "
            f"due to degradation (index={error_report.degradation_index:.3f})"
        )
        combined_lr = error_report.lr_cap

    log10_lr = math.log10(max(combined_lr, 1e-300))

    # ── Sensitivity contributions ─────────────────────────────────────
    total_log = sum(
        math.log10(max(d.adjusted_lr, 1e-300)) for d in per_locus_details
    )
    sensitivity_map = []
    for detail in per_locus_details:
        locus_log = math.log10(max(detail.adjusted_lr, 1e-300))
        pct = (locus_log / total_log * 100) if total_log != 0 else 0.0
        detail.sensitivity_contribution = round(pct, 2)
        sensitivity_map.append({
            "marker": detail.marker,
            "log10_lr": round(locus_log, 2),
            "contribution_pct": round(pct, 2),
            "has_dropout": detail.dropout_probability > 0.05,
            "has_stutter": detail.stutter_probability > 0.15,
        })

    # ── Random Match Probability ──────────────────────────────────────
    rmp = 1.0 / max(combined_lr, 1e-300)
    if combined_lr > 1:
        rmp_str = f"1 in {combined_lr:.2e}"
    else:
        rmp_str = "1 in 1 (inconclusive)"

    # ── Bayesian Posterior with Dynamic Prior ──────────────────────────
    prior_hp = 1.0 / max(suspect_pool_size, 1)
    prior_hd = 1.0 - prior_hp

    numerator = combined_lr * prior_hp
    denominator = numerator + prior_hd
    posterior_hp = numerator / denominator if denominator > 0 else 0.0
    posterior_hd = 1.0 - posterior_hp

    # Cap posterior for heavily degraded profiles
    if error_report.degradation_index > 0.5 and posterior_hp > 0.999:
        posterior_hp = min(posterior_hp, 0.999)
        posterior_hd = 1.0 - posterior_hp

    # Legacy prosecution/defense (50/50 prior — kept for backward compat)
    p_hp_legacy = combined_lr / (combined_lr + 1) if combined_lr > 0 else 0.0
    p_hd_legacy = 1.0 / (combined_lr + 1) if combined_lr > 0 else 1.0

    # ── 95% HPD Confidence Interval ───────────────────────────────────
    ci_lower, ci_upper = _compute_hpd_interval(
        posterior_hp, log10_lr, len(per_locus_details)
    )

    # ── Verbal Equivalence (legacy LR-based) ──────────────────────────
    verbal = "INCONCLUSIVE"
    for threshold, label in LR_VERBAL_SCALE:
        if combined_lr >= threshold:
            verbal = label
            break

    # ── ISO 17025 Verdict ─────────────────────────────────────────────
    iso_verdict = generate_iso17025_verdict(
        posterior_hp=posterior_hp,
        prior_used=prior_hp,
        combined_lr=combined_lr,
        degradation_index=error_report.degradation_index,
        dropout_count=error_report.dropout_count,
        stutter_count=error_report.stutter_count,
        total_loci=len(per_locus_details),
        population=population,
    )

    # ── High-frequency allele warning ─────────────────────────────────
    common_count = sum(1 for d in per_locus_details if d.rarity_score < 0.1)
    high_freq_warning = common_count > len(per_locus_details) * 0.6

    warning_msg = ""
    if high_freq_warning:
        warning_msg = (
            f"WARNING: {common_count}/{len(per_locus_details)} loci contain "
            f"high-frequency alleles (rarity < 0.1). "
            f"Secondary verification with additional markers is recommended."
        )

    result = LRResult(
        combined_lr=combined_lr,
        log10_lr=round(log10_lr, 2),
        random_match_probability=rmp,
        random_match_probability_str=rmp_str,
        verbal_equivalence=verbal,
        prosecution_probability=p_hp_legacy,
        defense_probability=p_hd_legacy,
        population_used=population,
        loci_analyzed=len(per_locus_details),
        per_locus_details=per_locus_details,
        high_frequency_warning=high_freq_warning,
        warning_message=warning_msg,
        # Phase 3.7 — Bayesian fields
        posterior_hp=posterior_hp,
        posterior_hd=posterior_hd,
        prior_hp=prior_hp,
        bayesian_ci_lower=ci_lower,
        bayesian_ci_upper=ci_upper,
        degradation_index=error_report.degradation_index,
        dropout_warnings=error_report.dropout_warnings,
        stutter_warnings=error_report.stutter_warnings,
        iso17025_verbal=iso_verdict.verbal_scale,
        sensitivity_map=sensitivity_map,
    )

    logger.info(
        f"[LR-CALC] ═══ Bayesian CLR Computed ═══\n"
        f"  Population: {population}\n"
        f"  Quality: {sample_quality} (degradation: {error_report.degradation_index:.3f})\n"
        f"  Loci: {len(per_locus_details)}\n"
        f"  CLR: {combined_lr:.2e} (log10 = {log10_lr:.2f})\n"
        f"  RMP: {rmp_str}\n"
        f"  Prior P(Hp): {prior_hp:.2e} (pool: {suspect_pool_size:,})\n"
        f"  Posterior P(Hp|E): {posterior_hp:.6f}\n"
        f"  95% HPD: [{ci_lower:.6f}, {ci_upper:.6f}]\n"
        f"  ISO 17025: {iso_verdict.verbal_scale}\n"
        f"  Verbal: {verbal}\n"
        f"  Dropouts: {error_report.dropout_count}, Stutters: {error_report.stutter_count}"
    )

    return result
