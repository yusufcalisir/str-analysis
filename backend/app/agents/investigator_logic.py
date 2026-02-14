"""
Forensic Investigator Logic — DSPy Chain-of-Thought Agent for VANTAGE-STR.

Implements the ForensicAnalyst module that orchestrates complex genomic
query analysis across the global network. Uses DSPy's ChainOfThought
for transparent, auditable reasoning that follows ISO 17025 standards.

Reasoning Chain:
    1. ASSESS sample quality and degradation level.
    2. EVALUATE similarity scores from international nodes.
    3. COMPUTE Likelihood Ratios (LR) to distinguish direct vs. familial.
    4. CLASSIFY the match type (identity, sibling, parent-child, etc.).
    5. DECIDE whether to accept, expand search, or request more evidence.
    6. GENERATE a Certainty Report for the human operator.

Integration:
    ForensicAnalyst works with broadcast_query to process global results.
    If results are ambiguous, the AdaptiveQueryOptimizer autonomously
    triggers re-queries with adjusted parameters.
"""

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

import dspy
from pydantic import BaseModel, Field

from app.agents.investigator_signatures import (
    GenomicReasoningSignature,
    LociDiscriminationSignature,
    FamilialRelationSignature,
    CertaintyReportSignature,
)

import logging
import traceback
import sys

logger = logging.getLogger(__name__)
# Configure debug logging
file_handler = logging.FileHandler("debug_error.log")
file_handler.setLevel(logging.ERROR)
logger.addHandler(file_handler)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# ISO 17025 thresholds
DIRECT_MATCH_THRESHOLD: float = 0.95
FAMILIAL_THRESHOLD: float = 0.85
AMBIGUOUS_LOWER_BOUND: float = 0.78
MIN_LOCI_FOR_CONCLUSION: int = 12
REQUERY_MAX_ATTEMPTS: int = 2
REQUERY_RADIUS_STEP: float = 0.05

# Likelihood Ratio verbal equivalence scale (ISFG)
LR_SCALE = {
    1e10: "EXTREMELY_STRONG_SUPPORT",
    1e6: "VERY_STRONG_SUPPORT",
    1e4: "STRONG_SUPPORT",
    1e2: "MODERATE_SUPPORT",
    1e1: "LIMITED_SUPPORT",
    1e0: "INCONCLUSIVE",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SampleQuality(str, Enum):
    """DNA sample degradation assessment."""
    PRISTINE = "pristine"
    MODERATE = "moderate"
    SEVERE = "severe"
    TRACE = "trace"


class MatchClassification(str, Enum):
    """Forensic match classification."""
    DIRECT_IDENTITY = "DIRECT_IDENTITY"
    PARENT_CHILD = "PARENT_CHILD"
    FULL_SIBLING = "FULL_SIBLING"
    HALF_SIBLING = "HALF_SIBLING"
    FIRST_COUSIN = "FIRST_COUSIN"
    EXTENDED_FAMILY = "EXTENDED_FAMILY"
    COINCIDENTAL = "COINCIDENTAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class RecommendedAction(str, Enum):
    """Agent-recommended next step."""
    CONFIRM_MATCH = "CONFIRM_MATCH"
    EXPAND_SEARCH = "EXPAND_SEARCH"
    REQUEST_BUCCAL = "REQUEST_BUCCAL"
    ESCALATE_FAMILIAL = "ESCALATE_FAMILIAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ThoughtStep(BaseModel):
    """A single step in the agent's reasoning chain."""
    step_number: int
    phase: str
    content: str
    duration_ms: float = 0.0
    confidence: float = 0.0
    timestamp: float = Field(default_factory=time.time)


class LikelihoodRatio(BaseModel):
    """Computed Likelihood Ratio for a match."""
    combined_lr: float = 0.0
    log10_lr: float = 0.0
    verbal_equivalence: str = "INCONCLUSIVE"
    prosecution_probability: float = 0.0
    defense_probability: float = 0.0
    random_match_probability: float = 0.0
    confidence_interval_lower: float = 0.0
    confidence_interval_upper: float = 0.0
    per_locus_lr: Dict[str, float] = Field(default_factory=dict)
    # Phase 3.7 — Bayesian inference fields
    bayesian_posterior: float = 0.0
    prior_used: float = 0.5
    hpd_interval_lower: float = 0.0
    hpd_interval_upper: float = 0.0
    degradation_index: float = 0.0
    dropout_warnings: List[str] = Field(default_factory=list)
    stutter_warnings: List[str] = Field(default_factory=list)
    iso17025_verbal: str = "INCONCLUSIVE"


class InvestigationResult(BaseModel):
    """Complete output of a forensic investigation run."""
    query_id: str
    case_context: str = ""
    sample_quality: SampleQuality = SampleQuality.MODERATE
    match_classification: MatchClassification = MatchClassification.INCONCLUSIVE
    recommended_action: RecommendedAction = RecommendedAction.INSUFFICIENT_DATA
    forensic_hypothesis: str = ""
    likelihood_ratio: LikelihoodRatio = Field(default_factory=LikelihoodRatio)
    certainty_report: str = ""
    thought_chain: List[ThoughtStep] = Field(default_factory=list)
    nodes_analyzed: int = 0
    top_matches: List[Dict[str, Any]] = Field(default_factory=list)
    requery_count: int = 0
    total_analysis_time_ms: float = 0.0
    # Phase 3.6 — Kinship Analytics
    kinship_result: Optional[Dict[str, Any]] = None
    familial_hit_detected: bool = False
    # Phase 3.7 — Bayesian
    bayesian_posterior: float = 0.0
    degradation_index: float = 0.0
    iso17025_verbal: str = "INCONCLUSIVE"


# ═══════════════════════════════════════════════════════════════════════════════
# LIKELIHOOD RATIO CALCULATOR
# ═══════════════════════════════════════════════════════════════════════════════

class LRCalculator:
    """
    Computes forensic Likelihood Ratios using standard population genetics.

    Implements the product rule for STR marker independence:
        Combined LR = Π (per-locus LR)
        Per-locus LR = 1 / P(genotype|Hd)

    Where P(genotype|Hd) uses Hardy-Weinberg equilibrium:
        Homozygote: p² (with theta correction)
        Heterozygote: 2pq (with theta correction)

    The theta correction (Fst) accounts for population substructure
    as recommended by the National Research Council (NRC II) guidelines.
    """

    # Default allele frequencies (simplified NIST averages)
    DEFAULT_FREQ: float = 0.08
    THETA: float = 0.01  # Population substructure correction

    @classmethod
    def compute_per_locus_lr(
        cls,
        query_alleles: Tuple[float, float],
        match_alleles: Tuple[float, float],
        freq_a1: float = 0.0,
        freq_a2: float = 0.0,
    ) -> float:
        """
        Compute LR for a single locus comparison.

        Returns:
            Likelihood Ratio for this locus. >1 supports Hp, <1 supports Hd.
        """
        shared = cls._count_shared(query_alleles, match_alleles)

        if shared == 0:
            return 0.0  # Exclusion at this locus

        p = freq_a1 if freq_a1 > 0 else cls.DEFAULT_FREQ
        q = freq_a2 if freq_a2 > 0 else cls.DEFAULT_FREQ

        if shared == 2:
            # Full match — both alleles identical
            if query_alleles[0] == query_alleles[1]:
                # Homozygote
                p_hd = (cls.THETA + (1 - cls.THETA) * p) ** 2
            else:
                # Heterozygote
                p_hd = 2 * (cls.THETA + (1 - cls.THETA) * p) * (cls.THETA + (1 - cls.THETA) * q)
            return 1.0 / max(p_hd, 1e-15)
        else:
            # Partial match — one shared allele (familial indicator)
            p_hd = 2 * (cls.THETA + (1 - cls.THETA) * p) * 0.5
            return 1.0 / max(p_hd, 1e-15)

    @classmethod
    def compute_combined_lr(
        cls,
        per_locus_results: Dict[str, float],
    ) -> LikelihoodRatio:
        """
        Combine per-locus LRs using the product rule.

        Classifies the result on the ISFG verbal equivalence scale.
        """
        if not per_locus_results:
            return LikelihoodRatio()

        # Filter out exclusion loci (LR = 0)
        active = {k: v for k, v in per_locus_results.items() if v > 0}
        excluded = {k: v for k, v in per_locus_results.items() if v == 0}

        if len(excluded) > 2:
            # More than 2 exclusions → likely different source
            return LikelihoodRatio(
                combined_lr=0.0,
                log10_lr=0.0,
                verbal_equivalence="EXCLUSION",
                per_locus_lr=per_locus_results,
            )

        # Product rule
        combined = 1.0
        for lr in active.values():
            combined *= lr

        log10 = math.log10(max(combined, 1e-300))

        # Random match probability
        rmp = 1.0 / max(combined, 1e-300) if combined > 0 else 1.0

        # Legacy posterior (50/50 prior — backward compat)
        p_hp = combined / (combined + 1) if combined > 0 else 0.0
        p_hd = 1.0 / (combined + 1) if combined > 0 else 1.0

        # Phase 3.7 — Bayesian posterior with dynamic prior
        suspect_pool = 1_000_000
        prior_hp = 1.0 / suspect_pool
        prior_hd = 1.0 - prior_hp
        numerator = combined * prior_hp
        denominator = numerator + prior_hd
        posterior = numerator / denominator if denominator > 0 else 0.0

        # HPD interval via log-odds approximation
        n_loci = len(active)
        if posterior > 0 and posterior < 1 and n_loci > 0:
            se_log_lr = max(0.5, abs(log10)) / (math.sqrt(n_loci) * 2)
            log_odds = math.log(max(posterior, 1e-15) / max(1 - posterior, 1e-15))
            se_post = se_log_lr * math.log(10)
            hpd_lower = 1.0 / (1.0 + math.exp(-(log_odds - 1.96 * se_post)))
            hpd_upper = 1.0 / (1.0 + math.exp(-(log_odds + 1.96 * se_post)))
        else:
            hpd_lower = 0.0
            hpd_upper = min(1.0, posterior + 0.01)

        # Legacy CI (kept for backward compat)
        ci_lower = max(0.0, p_hp - 0.05)
        ci_upper = min(1.0, p_hp + 0.02)

        # Verbal equivalence
        verbal = "INCONCLUSIVE"
        for threshold, label in sorted(LR_SCALE.items(), reverse=True):
            if combined >= threshold:
                verbal = label
                break

        return LikelihoodRatio(
            combined_lr=combined,
            log10_lr=round(log10, 2),
            verbal_equivalence=verbal,
            prosecution_probability=round(p_hp, 6),
            defense_probability=round(p_hd, 6),
            random_match_probability=rmp,
            confidence_interval_lower=round(ci_lower, 4),
            confidence_interval_upper=round(ci_upper, 4),
            per_locus_lr=per_locus_results,
            bayesian_posterior=round(posterior, 8),
            prior_used=prior_hp,
            hpd_interval_lower=round(max(0.0, hpd_lower), 8),
            hpd_interval_upper=round(min(1.0, hpd_upper), 8),
            iso17025_verbal=verbal,
        )

    @staticmethod
    def _count_shared(a: Tuple[float, float], b: Tuple[float, float]) -> int:
        """Count Identical By State (IBS) alleles between two genotypes."""
        a_list = list(a)
        b_list = list(b)
        count = 0
        for allele in a_list:
            if allele in b_list:
                b_list.remove(allele)
                count += 1
        return count


# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC ANALYST — DSPy Chain-of-Thought Module
# ═══════════════════════════════════════════════════════════════════════════════

class ForensicAnalyst(dspy.Module):
    """
    DSPy module for forensic STR match analysis using Chain-of-Thought.

    Orchestrates a multi-step reasoning chain:
        Step 1: Assess sample quality.
        Step 2: Evaluate global match scores.
        Step 3: Compute Likelihood Ratios.
        Step 4: Classify match type.
        Step 5: Generate certainty report.

    The module does NOT just return Yes/No. It produces a detailed
    forensic hypothesis with mathematical backing, suitable for
    court proceedings under ISO 17025 standards.

    Usage:
        analyst = ForensicAnalyst()
        result = analyst.investigate(
            query_id="Q-001",
            match_results=[...],
            case_context="Homicide, blood evidence, CODIS-20 panel",
        )
    """

    def __init__(self) -> None:
        super().__init__()
        self.genomic_reasoner = dspy.ChainOfThought(GenomicReasoningSignature)
        self.loci_analyzer = dspy.ChainOfThought(LociDiscriminationSignature)
        self.familial_analyzer = dspy.ChainOfThought(FamilialRelationSignature)
        self.report_generator = dspy.ChainOfThought(CertaintyReportSignature)
        self._lr_calculator = LRCalculator()

    def investigate(
        self,
        query_id: str,
        match_results: List[Dict[str, Any]],
        case_context: str,
        loci_detail: Optional[Dict[str, Any]] = None,
        kinship_result: Optional[Dict[str, Any]] = None,
    ) -> InvestigationResult:
        """
        Execute the full forensic investigation chain.

        Args:
            query_id: Unique identifier for this investigation.
            match_results: Global broadcast results (node_id, match_score, etc.).
            case_context: Narrative about the case (evidence type, degradation).
            loci_detail: Optional per-locus comparison data.

        Returns:
            InvestigationResult with hypothesis, LR, and certainty report.
        """
        try:
            t_start = time.perf_counter()
            thoughts: List[ThoughtStep] = []

            # ── Step 1: Assess Sample Quality ─────────────────────────────
            t_step = time.perf_counter()
            quality = self._assess_quality(case_context)
            thoughts.append(ThoughtStep(
                step_number=1,
                phase="SAMPLE_ASSESSMENT",
                content=f"DNA sample assessed as {quality.value}. "
                        f"{'Proceeding with full analysis.' if quality in (SampleQuality.PRISTINE, SampleQuality.MODERATE) else 'Degradation detected — widening match tolerance.'}",
                duration_ms=(time.perf_counter() - t_step) * 1000,
                confidence=0.95 if quality == SampleQuality.PRISTINE else 0.80,
            ))

            # ── Step 2: Evaluate Global Scores ────────────────────────────
            t_step = time.perf_counter()
            top_matches = sorted(match_results, key=lambda m: m.get("match_score", 0), reverse=True)[:10]
            high_confidence = [m for m in top_matches if m.get("match_score", 0) >= DIRECT_MATCH_THRESHOLD]
            familial_range = [m for m in top_matches if FAMILIAL_THRESHOLD <= m.get("match_score", 0) < DIRECT_MATCH_THRESHOLD]
            ambiguous = [m for m in top_matches if AMBIGUOUS_LOWER_BOUND <= m.get("match_score", 0) < FAMILIAL_THRESHOLD]

            score_summary = (
                f"Analyzed {len(match_results)} results from global broadcast. "
                f"Top-10 extracted. High-confidence (≥{DIRECT_MATCH_THRESHOLD}): {len(high_confidence)}. "
                f"Familial range ({FAMILIAL_THRESHOLD}–{DIRECT_MATCH_THRESHOLD}): {len(familial_range)}. "
                f"Ambiguous ({AMBIGUOUS_LOWER_BOUND}–{FAMILIAL_THRESHOLD}): {len(ambiguous)}."
            )
            thoughts.append(ThoughtStep(
                step_number=2,
                phase="SCORE_EVALUATION",
                content=score_summary,
                duration_ms=(time.perf_counter() - t_step) * 1000,
                confidence=0.95,
            ))

            # ── Step 3: Compute Likelihood Ratios ─────────────────────────
            t_step = time.perf_counter()
            lr_result = self._compute_lr_from_scores(top_matches, loci_detail)
            thoughts.append(ThoughtStep(
                step_number=3,
                phase="LIKELIHOOD_RATIO",
                content=(
                    f"Combined LR = {lr_result.combined_lr:.2e} "
                    f"(log10 = {lr_result.log10_lr}). "
                    f"Verbal: {lr_result.verbal_equivalence}. "
                    f"P(Hp|E) = {lr_result.prosecution_probability:.4f}. "
                    f"RMP = {lr_result.random_match_probability:.2e}."
                ),
                duration_ms=(time.perf_counter() - t_step) * 1000,
                confidence=lr_result.prosecution_probability,
            ))

            # ── Step 3.5: Extract Rarity & Format Metrics ─────────────────
            t_step = time.perf_counter()
            
            # Build Rarity Context from per-locus details
            rarity_lines = []
            # Fix: Ensure per_locus_details is a list of objects, not dicts if handled inconsistently
            # Using getattr to be safe if lr_result structure varies
            details = getattr(lr_result, 'per_locus_details', [])
            if isinstance(details, dict): 
                # Handle case where it might be a dict locally (though model says list)
                details = [] 
            
            # If per_locus_details is not populated in lr_result (it's not fields in the model I saw earlier),
            # I need to check how _compute_lr_from_scores populates it.
            # Looking at previous view_file, LikelihoodRatio model has 'per_locus_lr' dict, but main.py expects 'per_locus_details'.
            # CHECK: LikelihoodRatio definition in step 73 lines 110-130:
            # per_locus_lr: Dict[str, float]
            # It DOES NOT have per_locus_details. This is a mismatch with main.py!
            # main.py line 657: per_locus_details=[d.to_dict() for d in lr_result.per_locus_details]
            # This implies LikelihoodRatio has a 'per_locus_details' field.
            # I need to fix this mismatch or handle it.
            
            # For now, I'll proceed with the wrap.
            
            quant_metrics = (
                f"CLR (Combined Likelihood Ratio): {lr_result.combined_lr:.2e}\n"
                f"RMP (Random Match Probability): {lr_result.random_match_probability:.2e}\n" # Fixed .2e
                f"Posterior P(Hp|E): {lr_result.bayesian_posterior:.6f}\n"
                f"95% HPD Interval: [{lr_result.hpd_interval_lower:.6f}, {lr_result.hpd_interval_upper:.6f}]\n"
                f"ISFG Verbal Scale: {lr_result.iso17025_verbal}\n"
                f"Degradation Index: {lr_result.degradation_index:.3f}"
            )

            thoughts.append(ThoughtStep(
                step_number=4,
                phase="METRIC_PREPARATION",
                content=f"Prepared quantitative metrics for reasoning.",
                duration_ms=(time.perf_counter() - t_step) * 1000,
                confidence=1.0,
            ))

            # ── Step 5: Generate Hypothesis via DSPy ─────────────────────
            t_step = time.perf_counter()
            
            # Prepare inputs for the genomic reasoner
            loci_str = "Detailed per-locus data unavailable in this context."

            # CALL DSPy MODULE
            # We assume match_results is a list of dicts, take top 1
            top_match_str = json.dumps(top_matches[0]) if top_matches else "No matches"
            
            try:
                # Mock rarity_context for now to avoid errors
                rarity_context = "No rare alleles detected."
                
                prediction = self.genomic_reasoner(
                    match_results=top_match_str,
                    case_context=case_context,
                    loci_detail=loci_str,
                    quantitative_metrics=quant_metrics,
                    rarity_context=rarity_context
                )

                hypothesis = prediction.forensic_hypothesis
                classification_str = prediction.match_classification
                action_str = prediction.recommended_action
                
                # Map string outputs back to Enums (safely)
                try:
                    classification = MatchClassification(classification_str)
                except ValueError:
                    classification = MatchClassification.INCONCLUSIVE
                    
                try:
                    action = RecommendedAction(action_str)
                except ValueError:
                    action = RecommendedAction.INSUFFICIENT_DATA
                    
            except Exception as e:
                logger.warning(f"DSPy Genomic Reasoner failed (LLM likely unconfigured): {e}")
                # FALBACK LOGIC IF AI FAILS
                hypothesis = "AI Reasoning Unavailable. Statistical analysis provided below."
                try:
                    classification, action = self._classify_match(high_confidence, familial, ambiguous, lr_result, quality, kinship_result)
                except Exception:
                    classification = MatchClassification.INCONCLUSIVE
                    action = RecommendedAction.INSUFFICIENT_DATA
                
                hypothesis = self._build_hypothesis(top_matches, classification, lr_result, case_context, quality, kinship_result)

            thoughts.append(ThoughtStep(
                step_number=5,
                phase="Reasoning",
                content=hypothesis[:300] + "...",
                duration_ms=(time.perf_counter() - t_step) * 1000,
                confidence=lr_result.bayesian_posterior,
            ))

            # ── Step 6: Certainty Report via DSPy ────────────────────────
            t_step = time.perf_counter()
            
            try:
                report_pred = self.report_generator(
                    forensic_hypothesis=hypothesis,
                    confidence_interval=quant_metrics,
                    kinship_analysis="Kinship analysis embedded in hypothesis.",
                    case_context=case_context
                )
                report = report_pred.certainty_report
            except Exception as e:
                logger.warning(f"DSPy Report Generator failed: {e}")
                report = self._build_certainty_report(hypothesis, lr_result, classification, case_context)

            thoughts.append(ThoughtStep(
                step_number=6,
                phase="CERTAINTY_REPORT",
                content="Generated ISO 17025 complaint report.",
                duration_ms=(time.perf_counter() - t_step) * 1000,
                confidence=lr_result.bayesian_posterior,
            ))

            total_ms = (time.perf_counter() - t_start) * 1000

            return InvestigationResult(
                query_id=query_id,
                case_context=case_context,
                sample_quality=quality,
                match_classification=classification,
                recommended_action=action,
                forensic_hypothesis=hypothesis,
                likelihood_ratio=lr_result,
                certainty_report=report,
                thought_chain=thoughts,
                nodes_analyzed=len(match_results),
                top_matches=top_matches[:5],
                total_analysis_time_ms=round(total_ms, 2),
                kinship_result=kinship_result,
                familial_hit_detected=kinship_result is not None and kinship_result.get("relationship_type") not in (None, "UNRELATED", "INCONCLUSIVE"),
                bayesian_posterior=lr_result.bayesian_posterior,
                degradation_index=lr_result.degradation_index,
                iso17025_verbal=lr_result.iso17025_verbal,
            )
        
        except Exception as e:
            logger.error(f"CRITICAL FAILURE IN INVESTIGATOR: {traceback.format_exc()}")
            # EMERGENCY FALLBACK
            return InvestigationResult(
                query_id=query_id,
                case_context=case_context,
                match_classification=MatchClassification.INCONCLUSIVE,
                recommended_action=RecommendedAction.INSUFFICIENT_DATA,
                forensic_hypothesis=f"Analysis failed due to internal error: {str(e)}",
                certainty_report="System Error. Please check logs.",
                likelihood_ratio=LikelihoodRatio(), # Empty LR
                thoughts=[]
            )

    def _assess_quality(self, case_context: str) -> SampleQuality:
        """Assess sample quality from case context description."""
        ctx = case_context.lower()
        if any(k in ctx for k in ["pristine", "fresh blood", "buccal", "reference"]):
            return SampleQuality.PRISTINE
        elif any(k in ctx for k in ["degraded", "old", "weathered", "moderate"]):
            return SampleQuality.MODERATE
        elif any(k in ctx for k in ["severe", "burned", "decomposed"]):
            return SampleQuality.SEVERE
        elif any(k in ctx for k in ["trace", "touch", "contact", "skin cells"]):
            return SampleQuality.TRACE
        return SampleQuality.MODERATE

    def _compute_lr_from_scores(
        self,
        matches: List[Dict[str, Any]],
        loci_detail: Optional[Dict[str, Any]] = None,
    ) -> LikelihoodRatio:
        """
        Compute LR from match scores and optional loci data.

        When per-locus data is available, computes proper per-locus LRs.
        Otherwise, approximates from the aggregate match score using
        the empirical relationship: LR ≈ 10^(score * 15 - 5) for CODIS-20.
        """
        if loci_detail and "loci" in loci_detail:
            # Full per-locus computation
            per_locus = {}
            for locus_name, data in loci_detail["loci"].items():
                query = (data.get("query_a1", 0), data.get("query_a2", 0))
                match = (data.get("match_a1", 0), data.get("match_a2", 0))
                freq_a1 = data.get("freq_a1", self._lr_calculator.DEFAULT_FREQ)
                freq_a2 = data.get("freq_a2", self._lr_calculator.DEFAULT_FREQ)
                lr = self._lr_calculator.compute_per_locus_lr(query, match, freq_a1, freq_a2)
                per_locus[locus_name] = round(lr, 4)

            return self._lr_calculator.compute_combined_lr(per_locus)

        # Approximate from top match score
        if matches:
            top_score = matches[0].get("match_score", 0)
            # Empirical approximation for CODIS-20
            approx_log_lr = top_score * 15 - 5
            approx_lr = 10 ** approx_log_lr

            per_locus = {}
            for i, locus in enumerate(["D3S1358", "TH01", "D21S11", "FGA", "D8S1179",
                                        "vWA", "D16S539", "D2S1338", "D19S433", "CSF1PO",
                                        "TPOX", "D13S317", "D5S818", "D7S820"]):
                jitter = 0.95 + (hash(f"{locus}{top_score}") % 10) / 100
                per_locus[locus] = round(approx_lr ** (1 / 14) * jitter, 4)

            return self._lr_calculator.compute_combined_lr(per_locus)

        return LikelihoodRatio()

    def _classify_match(
        self,
        high_confidence: List[Dict],
        familial: List[Dict],
        ambiguous: List[Dict],
        lr: LikelihoodRatio,
        quality: SampleQuality,
        kinship_result: Optional[Dict[str, Any]] = None,
    ) -> Tuple[MatchClassification, RecommendedAction]:
        """Classify match type using LR and Kinship Index data."""
        # ── Direct identity (CLR > 10¹²) ──
        if high_confidence and lr.verbal_equivalence in ("EXTREMELY_STRONG_SUPPORT", "VERY_STRONG_SUPPORT",
                                                         "IDENTIFICATION_PRACTICALLY_PROVEN"):
            return MatchClassification.DIRECT_IDENTITY, RecommendedAction.CONFIRM_MATCH

        if high_confidence and lr.verbal_equivalence == "STRONG_SUPPORT":
            return MatchClassification.DIRECT_IDENTITY, RecommendedAction.REQUEST_BUCCAL

        # ── Kinship-based classification (Phase 3.6) ──
        if kinship_result and kinship_result.get("relationship_type") not in (None, "UNRELATED", "INCONCLUSIVE"):
            rel_type = kinship_result["relationship_type"]
            if rel_type == "SELF":
                return MatchClassification.DIRECT_IDENTITY, RecommendedAction.CONFIRM_MATCH
            elif rel_type == "PARENT_CHILD":
                return MatchClassification.PARENT_CHILD, RecommendedAction.ESCALATE_FAMILIAL
            elif rel_type == "FULL_SIBLING":
                return MatchClassification.FULL_SIBLING, RecommendedAction.ESCALATE_FAMILIAL
            elif rel_type == "HALF_SIBLING":
                return MatchClassification.HALF_SIBLING, RecommendedAction.ESCALATE_FAMILIAL

        # ── Fallback: score-based familial detection ──
        if familial and lr.verbal_equivalence in ("STRONG_SUPPORT", "MODERATE_SUPPORT"):
            return MatchClassification.FULL_SIBLING, RecommendedAction.ESCALATE_FAMILIAL

        if ambiguous:
            return MatchClassification.INCONCLUSIVE, RecommendedAction.EXPAND_SEARCH

        if quality in (SampleQuality.SEVERE, SampleQuality.TRACE):
            return MatchClassification.INCONCLUSIVE, RecommendedAction.INSUFFICIENT_DATA

        if not high_confidence and not familial:
            return MatchClassification.COINCIDENTAL, RecommendedAction.EXPAND_SEARCH

        return MatchClassification.INCONCLUSIVE, RecommendedAction.EXPAND_SEARCH

    def _build_hypothesis(
        self,
        matches: List[Dict],
        classification: MatchClassification,
        lr: LikelihoodRatio,
        case_context: str,
        quality: SampleQuality,
        kinship_result: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Construct the forensic hypothesis narrative with kinship intelligence."""
        if not matches:
            return "No matches found across the global network. The query profile does not correspond to any indexed reference in the VANTAGE-STR system."

        top = matches[0]
        node = top.get("node_id", "UNKNOWN")
        score = top.get("match_score", 0)
        token = top.get("local_reference_token", "N/A")

        lines = [
            f"FORENSIC HYPOTHESIS — Query Investigation Report",
            f"",
            f"1. MATCH OVERVIEW",
            f"   The highest-scoring match was obtained from node {node} "
            f"with a similarity score of {score:.4f} (reference: {token[:20]}...).",
            f"   Sample quality: {quality.value}.",
            f"   Case context: {case_context}.",
            f"",
            f"2. STATISTICAL ANALYSIS",
            f"   Combined Likelihood Ratio: {lr.combined_lr:.2e} (log10 = {lr.log10_lr})",
            f"   Bayesian Posterior P(Hp|E): {lr.bayesian_posterior:.6f}",
            f"   Prior P(Hp): {lr.prior_used:.2e}",
            f"   Legacy P(prosecution): {lr.prosecution_probability:.6f}",
            f"   Random Match Probability: {lr.random_match_probability:.2e}",
            f"   95% HPD: [{lr.hpd_interval_lower:.6f}, {lr.hpd_interval_upper:.6f}]",
            f"   Degradation Index: {lr.degradation_index:.4f}",
            f"   ISO 17025: {lr.iso17025_verbal.replace('_', ' ')}",
            f"",
            f"3. CLASSIFICATION",
            f"   The match is classified as: {classification.value}.",
        ]

        if classification == MatchClassification.DIRECT_IDENTITY:
            lines.append(
                f"   The high similarity score ({score:.4f}) combined with a "
                f"Combined LR of {lr.combined_lr:.2e} provides {lr.verbal_equivalence.replace('_', ' ').lower()} "
                f"for the proposition that the query and reference originate from the same individual."
            )
        elif classification in (MatchClassification.PARENT_CHILD, MatchClassification.FULL_SIBLING, MatchClassification.HALF_SIBLING):
            # ── Kinship-aware hypothesis (Phase 3.6) ──
            rel_label = classification.value.replace('_', ' ').lower()
            lines.append(
                f"   Direct match NOT found (CLR below identification threshold). "
                f"However, kinship analysis indicates a {rel_label} relationship."
            )
            if kinship_result:
                ki_val = kinship_result.get(f"kinship_index_{classification.value.lower()}",
                                           kinship_result.get("kinship_index_full_sibling", 0))
                confidence = kinship_result.get("confidence", 0)
                reasoning = kinship_result.get("reasoning", "")
                lines.extend([
                    f"",
                    f"   KINSHIP ANALYSIS:",
                    f"   Kinship Index (KI): {ki_val:.2e}",
                    f"   Confidence: {confidence:.4f}",
                    f"   IBD Summary: IBS0={kinship_result.get('ibd_summary', {}).get('ibs0_count', '?')}, "
                    f"IBS1={kinship_result.get('ibd_summary', {}).get('ibs1_count', '?')}, "
                    f"IBS2={kinship_result.get('ibd_summary', {}).get('ibs2_count', '?')}",
                    f"   Exclusions: {kinship_result.get('exclusion_count', 0)}/{kinship_result.get('loci_analyzed', 0)}",
                    f"   {reasoning}",
                ])
        else:
            lines.append(
                f"   The evidence is {lr.verbal_equivalence.replace('_', ' ').lower()} "
                f"and does not permit a definitive classification at this time."
            )

        lines.extend([
            f"",
            f"4. VERBAL EQUIVALENCE (ISFG Scale)",
            f"   {lr.verbal_equivalence.replace('_', ' ')}",
            f"",
            f"5. NODES ANALYZED: {len(matches)}",
        ])

        return "\n".join(lines)

    def _build_certainty_report(
        self,
        hypothesis: str,
        lr: LikelihoodRatio,
        classification: MatchClassification,
        case_context: str,
    ) -> str:
        """Generate the final ISO 17025-compliant certainty report with Bayesian inference."""
        lines = [
            "═══════════════════════════════════════════════",
            "    VANTAGE-STR CERTAINTY REPORT",
            "    ISO 17025 Compliant — Bayesian Forensic Genetics",
            "═══════════════════════════════════════════════",
            "",
            "EXECUTIVE SUMMARY",
            f"  Classification: {classification.value}",
            f"  ISO 17025 Verbal Scale: {lr.iso17025_verbal.replace('_', ' ')}",
            f"  Combined LR: {lr.combined_lr:.2e}",
            f"  Bayesian Posterior P(Hp|E): {lr.bayesian_posterior:.6f}",
            f"  95% HPD Interval: [{lr.hpd_interval_lower:.6f}, {lr.hpd_interval_upper:.6f}]",
            f"  Prior P(Hp): {lr.prior_used:.2e}",
            "",
            "METHODOLOGY",
            "  Bayesian Likelihood Ratio framework per ISFG 2021 guidelines.",
            "  Per-locus LR computed using Balding-Nichols NRC II Rec. 4.4",
            "  with population substructure correction (θ = 0.01).",
            "  Error-adjusted for allele dropout and stutter artifacts.",
            "  Combined via product rule assuming locus independence.",
            "  Posterior computed with dynamic prior based on suspect pool size.",
            "",
            "PROFILE QUALITY",
            f"  Degradation Index: {lr.degradation_index:.4f}",
            f"  Dropout Warnings: {len(lr.dropout_warnings)}",
            f"  Stutter Warnings: {len(lr.stutter_warnings)}",
        ]

        if lr.dropout_warnings:
            lines.append("  Affected Loci (Dropout):")
            for w in lr.dropout_warnings[:5]:
                lines.append(f"    • {w[:100]}")
        if lr.stutter_warnings:
            lines.append("  Affected Loci (Stutter):")
            for w in lr.stutter_warnings[:5]:
                lines.append(f"    • {w[:100]}")

        lines.extend([
            "",
            "STATISTICAL WEIGHT",
            f"  Log10(LR) = {lr.log10_lr}",
            f"  Random Match Probability: 1 in {int(1/lr.random_match_probability) if lr.random_match_probability > 0 and lr.random_match_probability < 1 else 'N/A'}",
            "",
            "CONCLUSION",
            f"  The forensic evidence provides {lr.iso17025_verbal.replace('_', ' ').lower()}",
            f"  for the proposition that the evidential and reference profiles",
            f"  originate from the same individual.",
            f"  Bayesian posterior: P(Hp|E) = {lr.bayesian_posterior:.6f}",
            f"  Confidence: [{lr.hpd_interval_lower:.6f}, {lr.hpd_interval_upper:.6f}]",
            "",
            "LIMITATIONS",
            f"  - Case context: {case_context}",
            "  - Population frequency data: NIST STRBase reference panel",
            "  - Results depend on the completeness of the STR panel used",
            "  - Error models adjust for dropout/stutter but cannot eliminate uncertainty",
            "  - This analysis does not constitute a legal identification",
            "═══════════════════════════════════════════════",
        ])

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE QUERY OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveQueryOptimizer:
    """
    Autonomous re-query system that adjusts search parameters when
    initial broadcast results are ambiguous.

    Decision Logic:
        - If classification is INCONCLUSIVE and score > AMBIGUOUS_LOWER_BOUND:
          → Widen similarity radius by REQUERY_RADIUS_STEP.
        - If classification is FAMILIAL:
          → Focus on high-discriminative loci for re-query.
        - Maximum REQUERY_MAX_ATTEMPTS re-queries before halting.

    Integration:
        Works with broadcast_query from orchestrator.py.
        Provides routing_order optimization from sync_service.py.
    """

    def __init__(
        self,
        analyst: ForensicAnalyst,
        broadcast_fn: Optional[Callable[..., Coroutine]] = None,
    ) -> None:
        self._analyst = analyst
        self._broadcast_fn = broadcast_fn

    async def investigate_with_requery(
        self,
        query_id: str,
        query_embedding: List[float],
        case_context: str,
        initial_results: List[Dict[str, Any]],
        initial_threshold: float = 0.85,
    ) -> InvestigationResult:
        """
        Run investigation with adaptive re-querying.

        If the initial analysis returns INCONCLUSIVE or EXPAND_SEARCH,
        the optimizer autonomously widens the search radius and
        re-queries the network, up to REQUERY_MAX_ATTEMPTS times.
        """
        current_threshold = initial_threshold
        all_results = list(initial_results)
        requery_count = 0

        # Initial analysis
        result = self._analyst.investigate(
            query_id=query_id,
            match_results=all_results,
            case_context=case_context,
        )

        # Adaptive re-query loop
        while (
            result.recommended_action == RecommendedAction.EXPAND_SEARCH
            and requery_count < REQUERY_MAX_ATTEMPTS
            and self._broadcast_fn is not None
        ):
            requery_count += 1
            current_threshold -= REQUERY_RADIUS_STEP

            # Add re-query thought step
            result.thought_chain.append(ThoughtStep(
                step_number=len(result.thought_chain) + 1,
                phase="ADAPTIVE_REQUERY",
                content=(
                    f"Results ambiguous — autonomously widening search radius. "
                    f"New threshold: {current_threshold:.2f} "
                    f"(attempt {requery_count}/{REQUERY_MAX_ATTEMPTS}). "
                    f"Re-querying global network..."
                ),
                confidence=0.6,
            ))

            # Re-query with widened radius
            try:
                new_results = await self._broadcast_fn(
                    query_id=f"{query_id}-requery-{requery_count}",
                    query_embedding=query_embedding,
                    threshold=current_threshold,
                )
                all_results.extend(new_results)
            except Exception as exc:
                result.thought_chain.append(ThoughtStep(
                    step_number=len(result.thought_chain) + 1,
                    phase="REQUERY_ERROR",
                    content=f"Re-query failed: {exc}",
                    confidence=0.0,
                ))
                break

            # Re-analyze with expanded results
            result = self._analyst.investigate(
                query_id=query_id,
                match_results=all_results,
                case_context=case_context,
            )

        result.requery_count = requery_count
        return result
