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

        # ── Stage 3: Visual Reconstruction (Phase 3.4) ──
        image_url: Optional[str] = None
        
        # Generate forensic sketch/photo based on predicted phenotype
        from app.core.engine.prompt_architect import SuspectPromptGenerator
        from app.infrastructure.gen_ai_client import get_gen_ai_client

        try:
            # 1. Generate Prompt
            prompt_gen = SuspectPromptGenerator()
            # We use a default sex hint "male" for now, or infer from Y-chromosome markers if available
            generated_prompt = prompt_gen.generate(engine_result, sex_hint="male")
            
            logger.info(f"[PHENOTYPE-VISUAL] Visual Prompt Created: {generated_prompt.positive[:60]}...")

            # 2. Call GenAI Client (Async)
            logger.info(f"[PHENOTYPE-VISUAL] Calling GenAI API for {profile_id}...")
            
            # Default to mock for stability during development, can switch to "replicate" via env var
            client = get_gen_ai_client(provider="mock") 
            gen_result = await client.generate_suspect_visual(
                prompt=generated_prompt.positive,
                negative_prompt=generated_prompt.negative,
                seed=generated_prompt.seed,
            )

            image_url = gen_result.image_url
            logger.info(f"[GenAI_SUCCESS] Image stored at: {image_url}")

            # Add GenAI metadata to result
            engine_result["image_url"] = image_url
            engine_result["positive_prompt"] = generated_prompt.positive
            engine_result["negative_prompt"] = generated_prompt.negative
            engine_result["seed"] = generated_prompt.seed
            engine_result["genai_model_id"] = gen_result.model_id
            engine_result["trait_summary"] = generated_prompt.trait_summary

        except Exception as exc:
            logger.warning(f"[PHENOTYPE-VISUAL] Visualization failed: {exc}")
            # Fallback to a reliable placeholder
            # Using a generic avatar from UI Avatars as a safe fallback
            # In a real production system, this would be a local static asset like "/static/silhouette.png"
            engine_result["image_url"] = "https://ui-avatars.com/api/?name=Unknown+Suspect&background=0D8ABC&color=fff&size=512"
            engine_result["genai_model_id"] = "fallback-silhouette"

        # ── Stage 4: Merge results ──
        engine_result["ai_reasoning"] = ai_reasoning
        engine_result["model_version"] = "HIrisPlex-S v1.0"
        
        logger.info(f"[PHENOTYPE-API] API Response Sent with Image URL for {profile_id}")
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
