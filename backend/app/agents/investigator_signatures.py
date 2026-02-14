"""
Forensic Investigator DSPy Signatures — VANTAGE-STR Phase 3.1.

Defines the input/output contracts for the Forensic Investigator Agent.
These signatures guide the LLM's reasoning when analyzing match results
from global broadcast queries and producing ISO 17025-compliant
forensic reports.

Signatures:
    GenomicReasoningSignature  — Core match analysis + hypothesis generation.
    LociDiscriminationSignature — Per-locus discriminative power analysis.
    FamilialRelationSignature — Kinship vs. direct match distinction.
    CertaintyReportSignature — Final human-readable certainty report.
"""

import dspy


# ═══════════════════════════════════════════════════════════════════════════════
# CORE REASONING SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════════

class GenomicReasoningSignature(dspy.Signature):
    """
    Evaluate forensic STR match results across sovereign nodes and generate
    a probabilistic hypothesis about the identity of the matched individual.

    You are a senior forensic geneticist operating under ISO 17025 standards.
    You must distinguish between direct identity matches, familial associations
    (parent-child, siblings, cousins), and coincidental statistical hits.

    Apply Likelihood Ratio (LR) methodology as recommended by the International
    Society for Forensic Genetics (ISFG). Consider allele frequencies from
    the population databases of each responding node's jurisdiction.
    """

    # ── Inputs ──
    match_results: str = dspy.InputField(
        desc=(
            "JSON array of match results from global broadcast query. Each entry "
            "contains: node_id, country_code, match_score, local_reference_token. "
            "Higher match_score indicates greater similarity."
        )
    )
    case_context: str = dspy.InputField(
        desc=(
            "Narrative context: crime scene type, evidence type, DNA degradation level, "
            "STR panel used, and circumstantial evidence."
        )
    )
    loci_detail: str = dspy.InputField(
        desc=(
            "Per-locus comparison data. Format: 'LOCUS: query → match | shared/total'."
        )
    )
    quantitative_metrics: str = dspy.InputField(
        desc=(
            "Precise statistical metrics computed by the LR Engine. Includes: "
            "Combined Likelihood Ratio (CLR), Random Match Probability (RMP), "
            "Posterior P(Hp|E), and 95% HPD Confidence Interval. "
            "YOU MUST DATA-CITE THESE EXACT VALUES in your reasoning."
        )
    )
    rarity_context: str = dspy.InputField(
        desc=(
            "Contextual flags for rare/common alleles. Example: "
            "'D7S820: 10,11 (Rare - Top 1%)'. Use this to strengthen or weaken "
            "the evidential weight narrative."
        )
    )

    # ── Outputs ──
    forensic_hypothesis: str = dspy.OutputField(
        desc=(
            "Detailed Chain-of-Thought forensic hypothesis. MUST: "
            "(1) Explicitly cite the CLR and RMP from quantitative_metrics. "
            "(2) Discuss the significance of any rare alleles mentioned in rarity_context. "
            "(3) Classify the match (Identity vs Familial vs Coincidental). "
            "(4) Provide a verdict using the ISFG verbal scale. "
            "Tone: Formal, scientific, objective, ISO 17025 compliant."
        )
    )
    match_classification: str = dspy.OutputField(
        desc="One of: 'DIRECT_IDENTITY', 'PARENT_CHILD', 'FULL_SIBLING', 'HALF_SIBLING', 'COINCIDENTAL', 'INCONCLUSIVE'."
    )
    recommended_action: str = dspy.OutputField(
        desc="One of: 'CONFIRM_MATCH', 'EXPAND_SEARCH', 'REQUEST_BUCCAL', 'ESCALATE_FAMILIAL', 'INSUFFICIENT_DATA'."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# LOCI DISCRIMINATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class LociDiscriminationSignature(dspy.Signature):
    """
    Evaluate the discriminative power of individual STR loci to determine
    which markers provide the strongest evidence for or against a match.

    Used by the adaptive query optimizer to decide which loci to focus on
    when re-querying nodes with a wider similarity radius.
    """

    loci_comparison: str = dspy.InputField(
        desc=(
            "Full per-locus comparison between the query and candidate profiles. "
            "Format: 'LOCUS: query_a1/a2 → match_a1/a2 | pop_freq_a1, pop_freq_a2'. "
            "Each locus includes population frequency data for allele weight calculation."
        )
    )
    population_database: str = dspy.InputField(
        desc=(
            "Population allele frequency database identifier and summary stats. "
            "Example: 'NIST_Caucasian (n=361), NIST_Hispanic (n=238)'. "
            "Used to compute locus-specific likelihood ratios."
        )
    )

    per_locus_lr: str = dspy.OutputField(
        desc=(
            "Per-locus Likelihood Ratio breakdown. Format: 'LOCUS: LR=X.XX "
            "(discriminative_power=HIGH/MEDIUM/LOW); alleles_shared=N/2'. "
            "Sorted by discriminative power descending."
        )
    )
    focus_loci: str = dspy.OutputField(
        desc=(
            "Comma-separated list of loci that should be prioritized for "
            "re-query optimization. These are loci with HIGH discriminative power "
            "but incomplete or ambiguous matches in the current results."
        )
    )
    widen_radius_recommendation: str = dspy.OutputField(
        desc=(
            "If the current similarity radius is too narrow, specify the "
            "recommended new threshold (float 0–1) and which loci to relax. "
            "Format: 'WIDEN to threshold=0.XX on loci: [list]'. "
            "Or 'NO_CHANGE' if the current radius is adequate."
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FAMILIAL RELATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

class FamilialRelationSignature(dspy.Signature):
    """
    Determine the most likely biological relationship between the query
    and matched profiles using kinship index calculations.

    Apply the formulae from ISFG guidelines:
    - Parent-child: At least one shared allele at every locus (IBS ≥ 1).
    - Full siblings: Average shared alleles ≈ 1.5 per locus.
    - Half siblings: Average shared alleles ≈ 1.25 per locus.
    - Unrelated: Average shared alleles ≈ 1.0 per locus (frequency-dependent).
    """

    allele_sharing_profile: str = dspy.InputField(
        desc=(
            "Per-locus Identical By State (IBS) counts. Format: "
            "'LOCUS: IBS=N (query: a1/a2, match: a3/a4)'. "
            "IBS=0 (no shared alleles), IBS=1 (one shared), IBS=2 (both shared)."
        )
    )
    total_loci_compared: int = dspy.InputField(
        desc="Total number of STR loci used in the comparison."
    )
    population_allele_frequencies: str = dspy.InputField(
        desc="Relevant population allele frequencies for kinship index calculation."
    )

    kinship_index: str = dspy.OutputField(
        desc=(
            "Combined Kinship Index (CKI) value and interpretation. Format: "
            "'CKI = X.XXX; Interpretation: [STRONGLY_SUPPORTS / SUPPORTS / "
            "INCONCLUSIVE / EXCLUDES] kinship hypothesis'. "
            "Include per-locus kinship indices for audit trail."
        )
    )
    relationship_probability: str = dspy.OutputField(
        desc=(
            "Posterior probability of each relationship type. Format: "
            "'Parent-Child: X.XX%; Full Sibling: X.XX%; Half Sibling: X.XX%; "
            "Unrelated: X.XX%'. Must sum to ~100%."
        )
    )
    ibs_summary: str = dspy.OutputField(
        desc=(
            "Summary of IBS distribution. Format: 'IBS=2: N loci; IBS=1: N loci; "
            "IBS=0: N loci. Average IBS = X.XX'. Used to validate against "
            "expected distributions for each relationship type."
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CERTAINTY REPORT SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════════

class CertaintyReportSignature(dspy.Signature):
    """
    Generate a final, human-readable Certainty Report for the forensic
    operator. This report must be suitable for inclusion in legal proceedings
    and adhere to ISO 17025 reporting standards.

    The report must be written in clear, unambiguous language that a
    non-geneticist (judge, attorney, jury member) can understand while
    maintaining scientific precision.
    """

    forensic_hypothesis: str = dspy.InputField(
        desc="The forensic hypothesis from the GenomicReasoningSignature."
    )
    confidence_interval: str = dspy.InputField(
        desc="The confidence interval and LR from the GenomicReasoningSignature."
    )
    kinship_analysis: str = dspy.InputField(
        desc="The kinship index and relationship probabilities, if computed."
    )
    case_context: str = dspy.InputField(
        desc="Original case context for the investigation."
    )

    certainty_report: str = dspy.OutputField(
        desc=(
            "Formal forensic report with sections: (1) EXECUTIVE SUMMARY — "
            "one-paragraph conclusion. (2) METHODOLOGY — statistical approach used. "
            "(3) FINDINGS — detailed locus-by-locus analysis. (4) STATISTICAL "
            "WEIGHT — likelihood ratios and random match probability. "
            "(5) CONCLUSION — verbal equivalence scale (EXTREMELY STRONG SUPPORT / "
            "STRONG SUPPORT / MODERATE SUPPORT / LIMITED SUPPORT / INCONCLUSIVE). "
            "(6) LIMITATIONS — factors that may affect reliability."
        )
    )
    verbal_equivalence: str = dspy.OutputField(
        desc=(
            "One of the ISFG recommended verbal equivalence scale terms: "
            "'EXTREMELY_STRONG_SUPPORT', 'VERY_STRONG_SUPPORT', 'STRONG_SUPPORT', "
            "'MODERATE_SUPPORT', 'LIMITED_SUPPORT', 'INCONCLUSIVE', 'EXCLUSION'."
        )
    )
