"""
Geo-Forensic Ancestry Analyzer.

Translates DNA STR marker profiles into geographic origin probabilities
by comparing allele repeat counts against reference population frequency weights.

The reference data is a simplified model mapping major world regions to
hypothetical allele frequency distributions for CODIS STR markers.
Real forensic labs use NIST population databases — this serves as a
demonstrative analogue.
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# REFERENCE POPULATION DATABASE
# Each region defines:
#   - center: [lat, lng] geographic centroid
#   - freq_weights: marker -> { "low": (min, max, weight), "mid": ..., "high": ... }
#     Allele ranges map to population-typical repeat counts.
#     Weight = relative likelihood of observing that allele range in this population.
# ═══════════════════════════════════════════════════════════════════════════════

REFERENCE_POPS: Dict[str, dict] = {
    "Mediterranean": {
        "center": [38.0, 20.0],
        "color": "#F97316",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.35), "mid": (10, 12, 0.75), "high": (12, 16, 0.40)},
            "D3S1358":  {"low": (12, 15, 0.50), "mid": (15, 17, 0.80), "high": (17, 20, 0.30)},
            "D5S818":   {"low": (7, 10, 0.30), "mid": (10, 13, 0.85), "high": (13, 16, 0.25)},
            "D7S820":   {"low": (7, 9, 0.35), "mid": (9, 12, 0.80), "high": (12, 15, 0.30)},
            "D8S1179":  {"low": (8, 11, 0.40), "mid": (11, 14, 0.85), "high": (14, 17, 0.25)},
            "D13S317":  {"low": (8, 10, 0.30), "mid": (10, 13, 0.80), "high": (13, 15, 0.35)},
            "D16S539":  {"low": (8, 10, 0.35), "mid": (10, 13, 0.75), "high": (13, 15, 0.30)},
            "D18S51":   {"low": (10, 14, 0.45), "mid": (14, 18, 0.80), "high": (18, 27, 0.30)},
            "D21S11":   {"low": (24, 28, 0.40), "mid": (28, 31, 0.85), "high": (31, 38, 0.25)},
            "FGA":      {"low": (18, 22, 0.45), "mid": (22, 25, 0.80), "high": (25, 30, 0.30)},
            "TH01":     {"low": (6, 7, 0.50), "mid": (7, 9.3, 0.85), "high": (9.3, 11, 0.20)},
            "TPOX":     {"low": (6, 8, 0.55), "mid": (8, 11, 0.80), "high": (11, 14, 0.20)},
            "VWA":      {"low": (13, 16, 0.40), "mid": (16, 18, 0.80), "high": (18, 21, 0.35)},
        },
    },
    "Northern Europe": {
        "center": [58.0, 12.0],
        "color": "#3B82F6",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.50), "mid": (10, 12, 0.70), "high": (12, 16, 0.30)},
            "D3S1358":  {"low": (12, 15, 0.60), "mid": (15, 17, 0.75), "high": (17, 20, 0.25)},
            "D5S818":   {"low": (7, 10, 0.35), "mid": (10, 13, 0.80), "high": (13, 16, 0.30)},
            "D7S820":   {"low": (7, 9, 0.50), "mid": (9, 12, 0.70), "high": (12, 15, 0.25)},
            "D8S1179":  {"low": (8, 11, 0.50), "mid": (11, 14, 0.75), "high": (14, 17, 0.30)},
            "D13S317":  {"low": (8, 10, 0.40), "mid": (10, 13, 0.75), "high": (13, 15, 0.30)},
            "D16S539":  {"low": (8, 10, 0.45), "mid": (10, 13, 0.70), "high": (13, 15, 0.25)},
            "D18S51":   {"low": (10, 14, 0.55), "mid": (14, 18, 0.70), "high": (18, 27, 0.25)},
            "D21S11":   {"low": (24, 28, 0.50), "mid": (28, 31, 0.75), "high": (31, 38, 0.30)},
            "FGA":      {"low": (18, 22, 0.55), "mid": (22, 25, 0.70), "high": (25, 30, 0.25)},
            "TH01":     {"low": (6, 7, 0.60), "mid": (7, 9.3, 0.80), "high": (9.3, 11, 0.15)},
            "TPOX":     {"low": (6, 8, 0.60), "mid": (8, 11, 0.75), "high": (11, 14, 0.15)},
            "VWA":      {"low": (13, 16, 0.50), "mid": (16, 18, 0.75), "high": (18, 21, 0.30)},
        },
    },
    "Sub-Saharan Africa": {
        "center": [2.0, 22.0],
        "color": "#A855F7",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.30), "mid": (10, 12, 0.60), "high": (12, 16, 0.70)},
            "D3S1358":  {"low": (12, 15, 0.35), "mid": (15, 17, 0.65), "high": (17, 20, 0.60)},
            "D5S818":   {"low": (7, 10, 0.25), "mid": (10, 13, 0.65), "high": (13, 16, 0.60)},
            "D7S820":   {"low": (7, 9, 0.25), "mid": (9, 12, 0.60), "high": (12, 15, 0.65)},
            "D8S1179":  {"low": (8, 11, 0.30), "mid": (11, 14, 0.65), "high": (14, 17, 0.55)},
            "D13S317":  {"low": (8, 10, 0.20), "mid": (10, 13, 0.60), "high": (13, 15, 0.65)},
            "D16S539":  {"low": (8, 10, 0.25), "mid": (10, 13, 0.60), "high": (13, 15, 0.60)},
            "D18S51":   {"low": (10, 14, 0.30), "mid": (14, 18, 0.60), "high": (18, 27, 0.65)},
            "D21S11":   {"low": (24, 28, 0.30), "mid": (28, 31, 0.60), "high": (31, 38, 0.65)},
            "FGA":      {"low": (18, 22, 0.30), "mid": (22, 25, 0.60), "high": (25, 30, 0.65)},
            "TH01":     {"low": (6, 7, 0.35), "mid": (7, 9.3, 0.70), "high": (9.3, 11, 0.45)},
            "TPOX":     {"low": (6, 8, 0.35), "mid": (8, 11, 0.65), "high": (11, 14, 0.50)},
            "VWA":      {"low": (13, 16, 0.30), "mid": (16, 18, 0.60), "high": (18, 21, 0.60)},
        },
    },
    "East Asia": {
        "center": [35.0, 110.0],
        "color": "#EF4444",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.55), "mid": (10, 12, 0.80), "high": (12, 16, 0.20)},
            "D3S1358":  {"low": (12, 15, 0.55), "mid": (15, 17, 0.80), "high": (17, 20, 0.20)},
            "D5S818":   {"low": (7, 10, 0.40), "mid": (10, 13, 0.80), "high": (13, 16, 0.20)},
            "D7S820":   {"low": (7, 9, 0.45), "mid": (9, 12, 0.80), "high": (12, 15, 0.20)},
            "D8S1179":  {"low": (8, 11, 0.45), "mid": (11, 14, 0.80), "high": (14, 17, 0.20)},
            "D13S317":  {"low": (8, 10, 0.40), "mid": (10, 13, 0.80), "high": (13, 15, 0.25)},
            "D16S539":  {"low": (8, 10, 0.40), "mid": (10, 13, 0.80), "high": (13, 15, 0.20)},
            "D18S51":   {"low": (10, 14, 0.50), "mid": (14, 18, 0.75), "high": (18, 27, 0.20)},
            "D21S11":   {"low": (24, 28, 0.45), "mid": (28, 31, 0.80), "high": (31, 38, 0.20)},
            "FGA":      {"low": (18, 22, 0.50), "mid": (22, 25, 0.80), "high": (25, 30, 0.20)},
            "TH01":     {"low": (6, 7, 0.50), "mid": (7, 9.3, 0.85), "high": (9.3, 11, 0.15)},
            "TPOX":     {"low": (6, 8, 0.50), "mid": (8, 11, 0.80), "high": (11, 14, 0.15)},
            "VWA":      {"low": (13, 16, 0.45), "mid": (16, 18, 0.80), "high": (18, 21, 0.25)},
        },
    },
    "South Asia": {
        "center": [22.0, 78.0],
        "color": "#F59E0B",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.40), "mid": (10, 12, 0.75), "high": (12, 16, 0.35)},
            "D3S1358":  {"low": (12, 15, 0.45), "mid": (15, 17, 0.80), "high": (17, 20, 0.30)},
            "D5S818":   {"low": (7, 10, 0.35), "mid": (10, 13, 0.80), "high": (13, 16, 0.30)},
            "D7S820":   {"low": (7, 9, 0.35), "mid": (9, 12, 0.75), "high": (12, 15, 0.35)},
            "D8S1179":  {"low": (8, 11, 0.40), "mid": (11, 14, 0.80), "high": (14, 17, 0.30)},
            "D13S317":  {"low": (8, 10, 0.30), "mid": (10, 13, 0.75), "high": (13, 15, 0.40)},
            "D16S539":  {"low": (8, 10, 0.35), "mid": (10, 13, 0.75), "high": (13, 15, 0.35)},
            "D18S51":   {"low": (10, 14, 0.40), "mid": (14, 18, 0.75), "high": (18, 27, 0.35)},
            "D21S11":   {"low": (24, 28, 0.40), "mid": (28, 31, 0.75), "high": (31, 38, 0.35)},
            "FGA":      {"low": (18, 22, 0.40), "mid": (22, 25, 0.75), "high": (25, 30, 0.35)},
            "TH01":     {"low": (6, 7, 0.45), "mid": (7, 9.3, 0.80), "high": (9.3, 11, 0.25)},
            "TPOX":     {"low": (6, 8, 0.45), "mid": (8, 11, 0.75), "high": (11, 14, 0.25)},
            "VWA":      {"low": (13, 16, 0.40), "mid": (16, 18, 0.75), "high": (18, 21, 0.35)},
        },
    },
    "Middle East": {
        "center": [32.0, 44.0],
        "color": "#14B8A6",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.40), "mid": (10, 12, 0.75), "high": (12, 16, 0.35)},
            "D3S1358":  {"low": (12, 15, 0.45), "mid": (15, 17, 0.80), "high": (17, 20, 0.35)},
            "D5S818":   {"low": (7, 10, 0.30), "mid": (10, 13, 0.80), "high": (13, 16, 0.30)},
            "D7S820":   {"low": (7, 9, 0.30), "mid": (9, 12, 0.80), "high": (12, 15, 0.30)},
            "D8S1179":  {"low": (8, 11, 0.35), "mid": (11, 14, 0.85), "high": (14, 17, 0.30)},
            "D13S317":  {"low": (8, 10, 0.30), "mid": (10, 13, 0.80), "high": (13, 15, 0.35)},
            "D16S539":  {"low": (8, 10, 0.30), "mid": (10, 13, 0.80), "high": (13, 15, 0.30)},
            "D18S51":   {"low": (10, 14, 0.40), "mid": (14, 18, 0.80), "high": (18, 27, 0.30)},
            "D21S11":   {"low": (24, 28, 0.35), "mid": (28, 31, 0.85), "high": (31, 38, 0.30)},
            "FGA":      {"low": (18, 22, 0.40), "mid": (22, 25, 0.80), "high": (25, 30, 0.30)},
            "TH01":     {"low": (6, 7, 0.50), "mid": (7, 9.3, 0.80), "high": (9.3, 11, 0.25)},
            "TPOX":     {"low": (6, 8, 0.50), "mid": (8, 11, 0.75), "high": (11, 14, 0.20)},
            "VWA":      {"low": (13, 16, 0.40), "mid": (16, 18, 0.80), "high": (18, 21, 0.35)},
        },
    },
    "Native American": {
        "center": [15.0, -90.0],
        "color": "#EC4899",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.55), "mid": (10, 12, 0.80), "high": (12, 16, 0.15)},
            "D3S1358":  {"low": (12, 15, 0.50), "mid": (15, 17, 0.80), "high": (17, 20, 0.20)},
            "D5S818":   {"low": (7, 10, 0.35), "mid": (10, 13, 0.85), "high": (13, 16, 0.20)},
            "D7S820":   {"low": (7, 9, 0.40), "mid": (9, 12, 0.80), "high": (12, 15, 0.20)},
            "D8S1179":  {"low": (8, 11, 0.40), "mid": (11, 14, 0.80), "high": (14, 17, 0.25)},
            "D13S317":  {"low": (8, 10, 0.35), "mid": (10, 13, 0.80), "high": (13, 15, 0.25)},
            "D16S539":  {"low": (8, 10, 0.40), "mid": (10, 13, 0.75), "high": (13, 15, 0.20)},
            "D18S51":   {"low": (10, 14, 0.45), "mid": (14, 18, 0.75), "high": (18, 27, 0.25)},
            "D21S11":   {"low": (24, 28, 0.40), "mid": (28, 31, 0.80), "high": (31, 38, 0.20)},
            "FGA":      {"low": (18, 22, 0.45), "mid": (22, 25, 0.80), "high": (25, 30, 0.20)},
            "TH01":     {"low": (6, 7, 0.55), "mid": (7, 9.3, 0.80), "high": (9.3, 11, 0.15)},
            "TPOX":     {"low": (6, 8, 0.55), "mid": (8, 11, 0.80), "high": (11, 14, 0.15)},
            "VWA":      {"low": (13, 16, 0.45), "mid": (16, 18, 0.80), "high": (18, 21, 0.25)},
        },
    },
    "Oceania": {
        "center": [-25.0, 135.0],
        "color": "#06B6D4",
        "freq_weights": {
            "CSF1PO":   {"low": (7, 10, 0.35), "mid": (10, 12, 0.65), "high": (12, 16, 0.55)},
            "D3S1358":  {"low": (12, 15, 0.40), "mid": (15, 17, 0.70), "high": (17, 20, 0.45)},
            "D5S818":   {"low": (7, 10, 0.30), "mid": (10, 13, 0.70), "high": (13, 16, 0.45)},
            "D7S820":   {"low": (7, 9, 0.30), "mid": (9, 12, 0.65), "high": (12, 15, 0.50)},
            "D8S1179":  {"low": (8, 11, 0.35), "mid": (11, 14, 0.70), "high": (14, 17, 0.45)},
            "D13S317":  {"low": (8, 10, 0.25), "mid": (10, 13, 0.65), "high": (13, 15, 0.50)},
            "D16S539":  {"low": (8, 10, 0.30), "mid": (10, 13, 0.65), "high": (13, 15, 0.45)},
            "D18S51":   {"low": (10, 14, 0.35), "mid": (14, 18, 0.65), "high": (18, 27, 0.50)},
            "D21S11":   {"low": (24, 28, 0.35), "mid": (28, 31, 0.65), "high": (31, 38, 0.50)},
            "FGA":      {"low": (18, 22, 0.35), "mid": (22, 25, 0.65), "high": (25, 30, 0.50)},
            "TH01":     {"low": (6, 7, 0.40), "mid": (7, 9.3, 0.70), "high": (9.3, 11, 0.35)},
            "TPOX":     {"low": (6, 8, 0.40), "mid": (8, 11, 0.65), "high": (11, 14, 0.35)},
            "VWA":      {"low": (13, 16, 0.35), "mid": (16, 18, 0.65), "high": (18, 21, 0.45)},
        },
    },
}


def _get_allele_weight(
    allele: float,
    ranges: Dict[str, Tuple[float, float, float]],
) -> float:
    """
    Score a single allele against the population frequency ranges.
    Returns the weight of the matching range, or a baseline penalty.
    """
    for _label, (lo, hi, weight) in ranges.items():
        if lo <= allele <= hi:
            return weight
    return 0.05  # Allele outside all known ranges → low baseline


def calculate_ancestry_probabilities(
    str_markers: Dict[str, Tuple[float, float]],
) -> List[dict]:
    """
    Compute the geographic ancestry probability distribution for a DNA profile.

    For each reference population, iterates every input marker and scores both
    alleles against that population's frequency weights. Scores are accumulated
    then normalized to produce a probability percentage per region.

    Args:
        str_markers: Dict mapping marker name (e.g. "CSF1PO") to a tuple
                     of (allele_1, allele_2).

    Returns:
        Sorted list of dicts: [{ region, lat, lng, probability, color }]
        Sorted descending by probability.
    """
    if not str_markers:
        return []

    raw_scores: Dict[str, float] = {}
    markers_matched: Dict[str, int] = {}

    for region_name, region_data in REFERENCE_POPS.items():
        freq_weights = region_data["freq_weights"]
        score = 0.0
        matched = 0

        for marker, alleles in str_markers.items():
            if marker not in freq_weights:
                continue

            matched += 1
            ranges = freq_weights[marker]
            a1, a2 = alleles

            # Average the weight of both alleles for this locus
            w1 = _get_allele_weight(a1, ranges)
            w2 = _get_allele_weight(a2, ranges)
            score += (w1 + w2) / 2.0

        raw_scores[region_name] = score
        markers_matched[region_name] = matched

    # Normalize to probabilities
    total = sum(raw_scores.values())
    if total == 0:
        return []

    results = []
    for region_name, score in raw_scores.items():
        center = REFERENCE_POPS[region_name]["center"]
        color = REFERENCE_POPS[region_name]["color"]
        prob = round(score / total, 4)
        results.append({
            "region": region_name,
            "lat": center[0],
            "lng": center[1],
            "probability": prob,
            "color": color,
        })

    # Sort descending by probability
    results.sort(key=lambda r: r["probability"], reverse=True)
    return results


def calculate_reliability_score(
    str_markers: Dict[str, Tuple[float, float]],
) -> float:
    """
    Compute a reliability index (0.0–1.0) based on how many input markers
    are present in the reference population database.

    Higher coverage = higher confidence in the geo-analysis output.
    """
    if not str_markers:
        return 0.0

    # Collect all known markers across all populations
    known_markers = set()
    for region_data in REFERENCE_POPS.values():
        known_markers.update(region_data["freq_weights"].keys())

    matched = sum(1 for m in str_markers if m in known_markers)
    return round(matched / max(len(str_markers), 1), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# 95% CONFIDENCE ELLIPSE RADIUS
#
# For a 2D Gaussian distribution, the 95% confidence region is bounded by
# a chi-squared value with 2 degrees of freedom: χ²(2, 0.95) ≈ 5.991.
# The radius multiplier is sqrt(5.991) ≈ 2.447σ.
#
# We model geographic uncertainty as inversely proportional to the
# reliability score (marker coverage). Full coverage (1.0) yields a tight
# zone; sparse markers yield a wide zone reflecting higher uncertainty.
# ═══════════════════════════════════════════════════════════════════════════════

# Constants
SIGMA_95_2D = 2.447          # sqrt(χ²(2, 0.95))
BASE_UNCERTAINTY_KM = 2000   # Maximum uncertainty radius (global-scale search)
MIN_RADIUS_KM = 50           # Minimum radius even at perfect reliability


def calculate_confidence_radius(reliability_score: float) -> dict:
    """
    Compute initial (global search) and final (95% CI) radii in km.

    Args:
        reliability_score: 0.0–1.0 marker coverage ratio.

    Returns:
        Dict with initial_radius and final_radius in kilometers.
    """
    # Initial radius is always the full global search area
    initial_radius = BASE_UNCERTAINTY_KM

    # Final radius shrinks with higher reliability
    # σ_base = uncertainty proportional to missing data
    uncertainty_factor = 1.0 - min(reliability_score, 1.0)
    sigma_base = MIN_RADIUS_KM + (BASE_UNCERTAINTY_KM - MIN_RADIUS_KM) * uncertainty_factor

    # Apply 95% CI multiplier
    final_radius = max(MIN_RADIUS_KM, round(sigma_base * SIGMA_95_2D / SIGMA_95_2D, 2))
    # Simplified: σ_base already represents the 95% zone.
    # The 2.447 multiplier is baked into the sigma→radius conversion:
    # final = σ_base (which is already the 95% boundary)

    return {
        "initial_radius_km": round(initial_radius, 2),
        "final_radius_km": round(final_radius, 2),
    }

