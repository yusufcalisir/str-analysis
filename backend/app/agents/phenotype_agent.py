"""
PhenotypeAnalyst — DSPy Module for SNP-Based Phenotype Reasoning.

Phase 3.3: Predictive Intelligence for VANTAGE-STR.

This module wraps the PhenotypePredictor engine output with DSPy
Chain-of-Thought reasoning to provide biologically grounded explanations
for each phenotype prediction. The agent explains molecular mechanisms,
epistatic interactions, and population-specific considerations.

Architecture:
    1. PhenotypePredictor (engine) → raw probability distributions
    2. PhenotypeAnalyst (DSPy) → biological reasoning + structured report
    3. Combined output → PhenotypeReport (API response)

The agent follows the same lazy-loading pattern as ForensicValidator:
if the LLM backend is unavailable, it degrades gracefully to
engine-only predictions without reasoning.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from app.agents.signatures import PhenotypeAnalysisSignature
from app.core.engine.phenotype_engine import PhenotypePredictor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy PHENOTYPE ANALYST MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class PhenotypeAnalyst(dspy.Module):
    """
    DSPy-powered phenotype analysis module with Chain-of-Thought reasoning.

    Combines the deterministic HIrisPlex-S PhenotypePredictor with LLM-driven
    biological explanation to produce a forensically defensible phenotype report.

    The module uses dspy.ChainOfThought to ensure the AI agent explains
    each prediction step-by-step, providing an auditable reasoning chain
    grounded in molecular genetics.

    Usage:
        analyst = PhenotypeAnalyst()
        report = analyst.analyze(
            profile_id="PRF-001",
            snp_map={"rs12913832": "GG", "rs16891982": "GG", ...},
            node_id="EUROPOL-NL",
        )
    """

    def __init__(self) -> None:
        """Initialize the PhenotypeAnalyst with engine and ChainOfThought."""
        super().__init__()
        self.predictor = PhenotypePredictor()
        self.chain_of_thought = dspy.ChainOfThought(PhenotypeAnalysisSignature)

    def _serialize_snps(self, snp_map: Dict[str, str]) -> str:
        """
        Serialize SNP genotypes into a human-readable string for the LLM.

        Format: 'rsID: GENOTYPE; rsID: GENOTYPE; ...'
        Sorted by rsID for deterministic ordering.
        """
        parts = [f"{rsid}: {genotype}" for rsid, genotype in sorted(snp_map.items())]
        return "; ".join(parts)

    def _build_population_context(
        self,
        node_id: str,
        snp_count: int,
        ancestry: Dict[str, float],
    ) -> str:
        """
        Build population context string for the LLM.

        Includes node metadata and ancestry-informative estimates
        from the engine to guide population-aware reasoning.
        """
        ancestry_str = ", ".join(f"{k}: {v:.1%}" for k, v in sorted(ancestry.items(), key=lambda x: -x[1]))
        return (
            f"Node: {node_id} | "
            f"SNPs submitted: {snp_count} | "
            f"Ancestry estimates (from pigmentation AIMs): {ancestry_str} | "
            f"Note: Ancestry estimates are indicative only, based on a limited "
            f"set of pigmentation-associated markers, not a full AIM panel."
        )

    async def analyze(
        self,
        profile_id: str,
        snp_map: Dict[str, str],
        node_id: str = "UNKNOWN",
        skip_ai: bool = False,
    ) -> Dict:
        """
        Execute full phenotype analysis pipeline (Async).

        Pipeline:
            1. Run PhenotypePredictor engine for probability distributions.
            2. If AI is available and not skipped, invoke ChainOfThought
               for biological reasoning.
            3. Generate forensic visual reconstruction (GenAI).
            4. Merge engine probabilities, AI reasoning, and image URL.

        Args:
            profile_id: Unique profile identifier.
            snp_map: Dict of rsID → genotype.
            node_id: Originating node identifier for population context.
            skip_ai: If True, skip DSPy inference.

        Returns:
            Dict containing trait probabilities, ancestry, reasoning, and image_url.
        """
        # ── Stage 1: Engine prediction ──
        engine_result = self.predictor.predict(profile_id, snp_map)
        ai_reasoning: Optional[str] = None

        # ── Stage 2: DSPy reasoning (conditional) ──
        if not skip_ai:
            try:
                snp_data = self._serialize_snps(snp_map)
                trait_probs_json = json.dumps(
                    {t["trait"]: t["predictions"] for t in engine_result["traits"]},
                    indent=2,
                )
                population_context = self._build_population_context(
                    node_id,
                    len(snp_map),
                    engine_result.get("ancestry_indicators", {}),
                )

                # DSPy calls are synchronous, so we run them directly
                prediction = self.chain_of_thought(
                    snp_data=snp_data,
                    trait_probabilities=trait_probs_json,
                    population_context=population_context,
                )

                # Extract reasoning — may be a JSON string or plain text
                raw_report = str(getattr(prediction, "phenotype_report", ""))
                reasoning_chain = str(getattr(prediction, "reasoning", ""))

                if reasoning_chain and raw_report:
                    ai_reasoning = f"[REASONING]\n{reasoning_chain}\n\n[REPORT]\n{raw_report}"
                elif raw_report:
                    ai_reasoning = raw_report
                elif reasoning_chain:
                    ai_reasoning = reasoning_chain

                logger.info(
                    f"[PHENOTYPE-AI] Analysis complete for {profile_id}: "
                    f"reasoning length={len(ai_reasoning or '')} chars"
                )

            except Exception as exc:
                logger.warning(
                    f"[PHENOTYPE-AI] DSPy inference failed for {profile_id}: {exc}. "
                    "Falling back to engine-only predictions."
                )
                ai_reasoning = f"AI reasoning unavailable: {exc}"

        # ── Stage 3: Deterministic Phenotype Mapping (No GenAI) ──
        # Replaces visual reconstruction with scientific trait prediction
        
        # Determine top ancestry region for context-aware fallbacks
        ancestry_probs = engine_result.get("ancestry_indicators", {})
        
        # Run the new deterministic engine with Bayesian Sync
        from app.agents.phenotype_engine import PhenotypeEngine
        engine = PhenotypeEngine()
        phenotype_data = engine.predict_phenotype(snp_map, ancestry_probabilities=ancestry_probs)

        # Merge results
        # We replace the old 'traits' list with our structured forensic traits
        engine_result["traits"] = phenotype_data["traits"]
        engine_result["forensic_traits"] = phenotype_data["traits"]
        engine_result["phenotype_reliability"] = phenotype_data["reliability_score"]
        engine_result["phenotype_coherence"] = phenotype_data["coherence_score"]
        engine_result["coherence_status"] = phenotype_data["coherence_status"]
        engine_result["snps_analyzed"] = phenotype_data["snps_analyzed"]
        
        # Remove GenAI fields to prevent frontend confusion
        engine_result["image_url"] = None
        engine_result["genai_model_id"] = None
        
        # ── Stage 4: Merge results ──
        engine_result["ai_reasoning"] = ai_reasoning
        engine_result["model_version"] = "VANTAGE-STR Phenotype v2.0"
        
        logger.info(f"[PHENOTYPE-API] Analysis complete for {profile_id}. Traits: {phenotype_data['traits']}")
        return engine_result

    def forward(
        self,
        snp_data: str,
        trait_probabilities: str,
        population_context: str,
    ) -> dspy.Prediction:
        """
        DSPy Module forward pass — required by the dspy.Module interface.

        Used internally by DSPy optimizers during training.
        """
        return self.chain_of_thought(
            snp_data=snp_data,
            trait_probabilities=trait_probabilities,
            population_context=population_context,
        )
