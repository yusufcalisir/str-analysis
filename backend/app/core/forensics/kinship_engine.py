"""
Kinship Engine — VANTAGE-STR Phase 3.6.

Computes Kinship Indices (KI) to distinguish biological relationships
between two STR profiles. Uses Identity-By-Descent (IBD) analysis and
per-locus KI formulas derived from forensic genetics literature.

Relationship Models:
    Parent-Child: Mandatory sharing of ≥1 allele/locus (IBS1 or IBS2).
        KI = 1 / (2pᵢ)  [for the shared allele with freq pᵢ]

    Full Siblings: IBD proportions k₀=0.25, k₁=0.50, k₂=0.25.
        KI = (1 + pᵢ + pⱼ) / (4pᵢpⱼ)  [heterozygous shared]

    Half Siblings: IBD proportions k₀=0.50, k₁=0.50, k₂=0.00.
        KI = (1 + pᵢ) / (4pᵢ)  [for shared allele]

    Unrelated: KI < 1.0 across all loci.

Reference: Brenner (1997), Evett & Weir (1998), ISFG familial searching guidelines.
"""

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from app.core.forensics.population_data import (
    get_frequency,
    CODIS_LOCI,
    MIN_FREQUENCY,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class RelationshipType(str, Enum):
    SELF = "SELF"
    PARENT_CHILD = "PARENT_CHILD"
    FULL_SIBLING = "FULL_SIBLING"
    HALF_SIBLING = "HALF_SIBLING"
    UNRELATED = "UNRELATED"
    INCONCLUSIVE = "INCONCLUSIVE"


# Expected IBD proportions for each relationship type
# k₀ = P(0 alleles IBD), k₁ = P(1 allele IBD), k₂ = P(2 alleles IBD)
EXPECTED_IBD = {
    RelationshipType.SELF:         {"k0": 0.00, "k1": 0.00, "k2": 1.00},
    RelationshipType.PARENT_CHILD: {"k0": 0.00, "k1": 1.00, "k2": 0.00},
    RelationshipType.FULL_SIBLING: {"k0": 0.25, "k1": 0.50, "k2": 0.25},
    RelationshipType.HALF_SIBLING: {"k0": 0.50, "k1": 0.50, "k2": 0.00},
    RelationshipType.UNRELATED:    {"k0": 1.00, "k1": 0.00, "k2": 0.00},
}


@dataclass
class LocusKIDetail:
    """Kinship Index detail for a single locus."""
    marker: str
    alleles_a: Tuple[float, float]
    alleles_b: Tuple[float, float]
    shared_alleles: int  # IBS count: 0, 1, or 2
    ibs_pattern: str  # e.g., "IBS2", "IBS1", "IBS0"
    ki_parent_child: float
    ki_full_sibling: float
    ki_half_sibling: float
    excluded_parent_child: bool  # True if IBS0 → exclusion

    def to_dict(self) -> Dict:
        return {
            "marker": self.marker,
            "alleles_a": list(self.alleles_a),
            "alleles_b": list(self.alleles_b),
            "shared_alleles": self.shared_alleles,
            "ibs_pattern": self.ibs_pattern,
            "ki_parent_child": round(self.ki_parent_child, 4),
            "ki_full_sibling": round(self.ki_full_sibling, 4),
            "ki_half_sibling": round(self.ki_half_sibling, 4),
            "excluded_parent_child": self.excluded_parent_child,
        }


@dataclass
class KinshipResult:
    """Complete kinship analysis result."""
    relationship_type: RelationshipType = RelationshipType.INCONCLUSIVE
    confidence: float = 0.0
    kinship_index_parent_child: float = 0.0
    kinship_index_full_sibling: float = 0.0
    kinship_index_half_sibling: float = 0.0
    log10_ki_parent_child: float = 0.0
    log10_ki_full_sibling: float = 0.0
    log10_ki_half_sibling: float = 0.0
    exclusion_count: int = 0
    loci_analyzed: int = 0
    ibd_summary: Dict[str, float] = field(default_factory=dict)
    per_locus_details: List[LocusKIDetail] = field(default_factory=list)
    population_used: str = "European"
    reasoning: str = ""

    def to_dict(self) -> Dict:
        return {
            "relationship_type": self.relationship_type.value,
            "confidence": round(self.confidence, 4),
            "kinship_index_parent_child": self.kinship_index_parent_child,
            "kinship_index_full_sibling": self.kinship_index_full_sibling,
            "kinship_index_half_sibling": self.kinship_index_half_sibling,
            "log10_ki_parent_child": round(self.log10_ki_parent_child, 2),
            "log10_ki_full_sibling": round(self.log10_ki_full_sibling, 2),
            "log10_ki_half_sibling": round(self.log10_ki_half_sibling, 2),
            "exclusion_count": self.exclusion_count,
            "loci_analyzed": self.loci_analyzed,
            "ibd_summary": {k: round(v, 4) for k, v in self.ibd_summary.items()},
            "per_locus_details": [d.to_dict() for d in self.per_locus_details],
            "population_used": self.population_used,
            "reasoning": self.reasoning,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# IBS / IBD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ibs(
    alleles_a: Tuple[float, float],
    alleles_b: Tuple[float, float],
) -> int:
    """
    Compute Identity-By-State (IBS) count between two genotypes.

    IBS2: Both alleles shared (same genotype or complement).
    IBS1: Exactly one allele shared.
    IBS0: No alleles shared (exclusion for parent-child).

    Returns: 0, 1, or 2.
    """
    a1, a2 = alleles_a
    b1, b2 = alleles_b

    set_a = {a1, a2}
    set_b = {b1, b2}

    # Check for IBS2 (both alleles match)
    if set_a == set_b:
        return 2
    if (a1 == b1 and a2 == b2) or (a1 == b2 and a2 == b1):
        return 2

    # Check for IBS1 (exactly one allele shared)
    if a1 in set_b or a2 in set_b:
        return 1

    # IBS0 — no shared alleles
    return 0


def find_shared_allele_freq(
    alleles_a: Tuple[float, float],
    alleles_b: Tuple[float, float],
    marker: str,
    population: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Find the frequencies of shared alleles between two genotypes.

    Returns:
        (freq_shared_1, freq_shared_2) — second is None for IBS1.
    """
    a1, a2 = alleles_a
    b1, b2 = alleles_b

    shared = []
    # Check which alleles are shared
    if a1 == b1 or a1 == b2:
        shared.append(a1)
    if a2 == b1 or a2 == b2:
        if a2 not in shared:
            shared.append(a2)

    if not shared:
        return None, None

    freq_1 = get_frequency(marker, shared[0], population)
    freq_2 = get_frequency(marker, shared[1], population) if len(shared) > 1 else None

    return freq_1, freq_2


# ═══════════════════════════════════════════════════════════════════════════════
# PER-LOCUS KINSHIP INDEX
# ═══════════════════════════════════════════════════════════════════════════════

def compute_locus_ki(
    alleles_a: Tuple[float, float],
    alleles_b: Tuple[float, float],
    marker: str,
    population: str = "European",
) -> LocusKIDetail:
    """
    Compute Kinship Index for a single locus across all relationship hypotheses.

    Parent-Child:
        IBS0 → Exclusion (KI = 0)
        IBS1 → KI = 1 / (2p) where p = freq of shared allele
        IBS2 → KI = 1 / (2p) for rarest shared allele

    Full Sibling (Brenner 1997):
        IBS0 → KI calculated from IBD probabilities
        IBS1 → KI = (1 + p) / (4p)
        IBS2 heterozygous → KI = (1 + p + q) / (4pq)
        IBS2 homozygous → KI = (1 + p)² / (4p²)

    Half Sibling:
        IBS0 → KI from IBD probabilities
        IBS1 → KI = (1 + p) / (4p)
        IBS2 → KI = (2 + p + q) / (4(p + q))
    """
    ibs = compute_ibs(alleles_a, alleles_b)
    freq_s1, freq_s2 = find_shared_allele_freq(alleles_a, alleles_b, marker, population)

    excluded_pc = False
    ki_pc = 1.0
    ki_fs = 1.0
    ki_hs = 1.0

    if ibs == 0:
        # ── IBS0: No shared alleles ──
        excluded_pc = True
        ki_pc = 0.0

        # Full sibling: still possible (IBD0 = 25% of the time)
        # Use allele frequencies for the probability under sibling hypothesis
        p_a1 = get_frequency(marker, alleles_a[0], population)
        p_a2 = get_frequency(marker, alleles_a[1], population)
        p_b1 = get_frequency(marker, alleles_b[0], population)
        p_b2 = get_frequency(marker, alleles_b[1], population)

        # P(genotypes | siblings) vs P(genotypes | unrelated)
        p_unrelated = 2 * p_a1 * p_a2 * 2 * p_b1 * p_b2
        # Under sibling hypothesis with k₀=0.25, the IBS0 probability is reduced
        ki_fs = 0.25  # Penalized but not excluded
        ki_hs = 0.5   # Half siblings: k₀=0.50

    elif ibs == 1:
        # ── IBS1: One shared allele ──
        p = freq_s1 if freq_s1 is not None else MIN_FREQUENCY

        # Parent-child: KI = 1 / (2p)
        ki_pc = 1.0 / (2.0 * max(p, MIN_FREQUENCY))

        # Full sibling: KI = (1 + p) / (4p)
        ki_fs = (1.0 + p) / (4.0 * max(p, MIN_FREQUENCY))

        # Half sibling: KI = (1 + p) / (4p)
        ki_hs = (1.0 + p) / (4.0 * max(p, MIN_FREQUENCY))

    elif ibs == 2:
        # ── IBS2: Both alleles shared ──
        is_hom = alleles_a[0] == alleles_a[1]
        p = freq_s1 if freq_s1 is not None else MIN_FREQUENCY
        q = freq_s2 if freq_s2 is not None else p

        if is_hom:
            # Homozygous match
            # Parent-child: KI = 1 / (2p)
            ki_pc = 1.0 / (2.0 * max(p, MIN_FREQUENCY))

            # Full sibling: KI = (1 + p)² / (4p²)
            ki_fs = ((1.0 + p) ** 2) / (4.0 * max(p ** 2, MIN_FREQUENCY ** 2))

            # Half sibling: KI = (2 + 2p) / (4p) = (1+p)/(2p)
            ki_hs = (1.0 + p) / (2.0 * max(p, MIN_FREQUENCY))
        else:
            # Heterozygous match
            # Parent-child: KI = 1 / (2 * min(p, q))
            min_freq = min(p, q)
            ki_pc = 1.0 / (2.0 * max(min_freq, MIN_FREQUENCY))

            # Full sibling: KI = (1 + p + q) / (4pq)
            ki_fs = (1.0 + p + q) / (4.0 * max(p * q, MIN_FREQUENCY ** 2))

            # Half sibling: KI = (2 + p + q) / (4(p + q))
            ki_hs = (2.0 + p + q) / (4.0 * max(p + q, 2 * MIN_FREQUENCY))

    ibs_label = f"IBS{ibs}"

    return LocusKIDetail(
        marker=marker,
        alleles_a=alleles_a,
        alleles_b=alleles_b,
        shared_alleles=ibs,
        ibs_pattern=ibs_label,
        ki_parent_child=ki_pc,
        ki_full_sibling=ki_fs,
        ki_half_sibling=ki_hs,
        excluded_parent_child=excluded_pc,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMBINED KINSHIP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_kinship(
    profile_a: Dict[str, Tuple[float, float]],
    profile_b: Dict[str, Tuple[float, float]],
    population: str = "European",
) -> KinshipResult:
    """
    Compute the Combined Kinship Index across all shared loci between
    two STR profiles.

    CKI = Π (per-locus KI)

    Classifies the most likely biological relationship based on
    which hypothesis yields the highest CKI.

    Args:
        profile_a: Query profile {marker: (allele1, allele2)}.
        profile_b: Reference profile {marker: (allele1, allele2)}.
        population: Population group for frequency lookup.

    Returns:
        KinshipResult with relationship classification and full breakdown.
    """
    per_locus: List[LocusKIDetail] = []
    exclusion_count = 0

    # IBS distribution counters
    ibs_counts = {0: 0, 1: 0, 2: 0}

    # Find overlapping loci (excluding AMEL)
    shared_loci = set(profile_a.keys()) & set(profile_b.keys()) - {"AMEL"}

    for marker in sorted(shared_loci):
        alleles_a = profile_a[marker]
        alleles_b = profile_b[marker]

        detail = compute_locus_ki(alleles_a, alleles_b, marker, population)
        per_locus.append(detail)
        ibs_counts[detail.shared_alleles] += 1

        if detail.excluded_parent_child:
            exclusion_count += 1

    if not per_locus:
        return KinshipResult(population_used=population, reasoning="No overlapping loci found.")

    n_loci = len(per_locus)

    # ── Compute Combined KIs ──
    cki_pc = 1.0
    cki_fs = 1.0
    cki_hs = 1.0

    for detail in per_locus:
        cki_pc *= max(detail.ki_parent_child, 1e-30)
        cki_fs *= max(detail.ki_full_sibling, 1e-30)
        cki_hs *= max(detail.ki_half_sibling, 1e-30)

    log_pc = math.log10(max(cki_pc, 1e-300))
    log_fs = math.log10(max(cki_fs, 1e-300))
    log_hs = math.log10(max(cki_hs, 1e-300))

    # ── IBD summary (observed proportions) ──
    ibd_summary = {
        "ibs0_proportion": ibs_counts[0] / n_loci,
        "ibs1_proportion": ibs_counts[1] / n_loci,
        "ibs2_proportion": ibs_counts[2] / n_loci,
        "ibs0_count": ibs_counts[0],
        "ibs1_count": ibs_counts[1],
        "ibs2_count": ibs_counts[2],
    }

    # ── Classify relationship ──
    relationship, confidence, reasoning = _classify_relationship(
        cki_pc, cki_fs, cki_hs,
        log_pc, log_fs, log_hs,
        exclusion_count, n_loci,
        ibs_counts,
    )

    result = KinshipResult(
        relationship_type=relationship,
        confidence=confidence,
        kinship_index_parent_child=cki_pc,
        kinship_index_full_sibling=cki_fs,
        kinship_index_half_sibling=cki_hs,
        log10_ki_parent_child=log_pc,
        log10_ki_full_sibling=log_fs,
        log10_ki_half_sibling=log_hs,
        exclusion_count=exclusion_count,
        loci_analyzed=n_loci,
        ibd_summary=ibd_summary,
        per_locus_details=per_locus,
        population_used=population,
        reasoning=reasoning,
    )

    logger.info(
        f"[KINSHIP] ═══ Analysis Complete ═══\n"
        f"  Loci: {n_loci} | Exclusions: {exclusion_count}\n"
        f"  CKI Parent-Child: {cki_pc:.2e} (log10 = {log_pc:.2f})\n"
        f"  CKI Full Sibling: {cki_fs:.2e} (log10 = {log_fs:.2f})\n"
        f"  CKI Half Sibling:  {cki_hs:.2e} (log10 = {log_hs:.2f})\n"
        f"  Result: {relationship.value} (confidence: {confidence:.4f})"
    )

    return result


def _classify_relationship(
    cki_pc: float, cki_fs: float, cki_hs: float,
    log_pc: float, log_fs: float, log_hs: float,
    exclusion_count: int, n_loci: int,
    ibs_counts: Dict[int, int],
) -> Tuple[RelationshipType, float, str]:
    """
    Classify the biological relationship based on Combined Kinship Indices.

    Decision logic:
        1. If exclusion_count > 3 → parent-child excluded
        2. Compare CKIs to determine best-fit relationship
        3. CKI > 10⁴ for any hypothesis → strong support
        4. All CKI < 1.0 → unrelated
    """
    # ── Rule 1: Parent-child exclusion ──
    pc_excluded = exclusion_count > 3

    # ── Rule 2: All weak → unrelated ──
    if cki_pc < 1.0 and cki_fs < 1.0 and cki_hs < 1.0:
        return (
            RelationshipType.UNRELATED,
            0.95,
            f"All kinship indices below 1.0. No biological relationship supported. "
            f"IBS distribution: IBS0={ibs_counts[0]}, IBS1={ibs_counts[1]}, IBS2={ibs_counts[2]}."
        )

    # ── Rule 3: Self / identical twins ──
    if ibs_counts[0] == 0 and ibs_counts[2] == n_loci:
        return (
            RelationshipType.SELF,
            0.9999,
            f"All {n_loci} loci show IBS2 (complete genotype match). "
            f"Profiles are from the same individual or identical twins."
        )

    # ── Determine best-fit hypothesis ──
    candidates = []

    if not pc_excluded and cki_pc > 1.0:
        candidates.append((cki_pc, log_pc, RelationshipType.PARENT_CHILD))

    if cki_fs > 1.0:
        candidates.append((cki_fs, log_fs, RelationshipType.FULL_SIBLING))

    if cki_hs > 1.0:
        candidates.append((cki_hs, log_hs, RelationshipType.HALF_SIBLING))

    if not candidates:
        return (
            RelationshipType.INCONCLUSIVE,
            0.3,
            f"Mixed signals. Exclusions: {exclusion_count}/{n_loci}. "
            f"No hypothesis achieves CKI > 1.0 after exclusion filtering."
        )

    # Sort by CKI descending → best hypothesis first
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_cki, best_log, best_type = candidates[0]

    # Confidence from log10 of best CKI
    if best_log >= 6:
        confidence = 0.9999
    elif best_log >= 4:
        confidence = 0.99
    elif best_log >= 2:
        confidence = 0.95
    elif best_log >= 1:
        confidence = 0.80
    else:
        confidence = 0.60

    # Build reasoning
    pc_status = f"EXCLUDED ({exclusion_count} exclusions)" if pc_excluded else f"{cki_pc:.2e}"

    reasoning = (
        f"Best-fit relationship: {best_type.value}. "
        f"CKI(Parent-Child)={pc_status}, "
        f"CKI(Full Sibling)={cki_fs:.2e}, "
        f"CKI(Half Sibling)={cki_hs:.2e}. "
        f"IBS distribution: IBS0={ibs_counts[0]}, IBS1={ibs_counts[1]}, IBS2={ibs_counts[2]} "
        f"across {n_loci} loci."
    )

    # Additional context for parent-child
    if best_type == RelationshipType.PARENT_CHILD and ibs_counts[0] == 0:
        reasoning += " Mandatory allele sharing confirmed at all loci (obligate allele criterion met)."

    return best_type, confidence, reasoning
