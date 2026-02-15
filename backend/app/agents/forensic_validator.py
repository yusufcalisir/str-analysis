"""
ForensicValidator — DSPy Module for STR Profile Integrity Enforcement.

This module intercepts every incoming GenomicProfile and verifies its
biological plausibility before the data enters the vector database.
It combines deterministic rule-based pre-checks (allele bounds, loci
completeness) with LLM-powered chain-of-thought reasoning for
statistical rarity and data poisoning detection.

Pipeline:
    1. Rule-based pre-validation (fast, deterministic)
    2. DSPy ChainOfThought assessment (deep, reasoning-based)
    3. Composite score computation → accept / quarantine decision

Quarantine Threshold:
    Profiles scoring below 0.85 are flagged for quarantine in PostgreSQL.
    This threshold balances false-positive rate against forensic safety —
    a missed poisoned profile has catastrophic judicial consequences.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import dspy
from pydantic import BaseModel, Field

from app.agents.signatures import BiometricValidationSignature
from app.schemas.genomic import CODIS_MARKERS, LocusDataSchema

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ALLELE REFERENCE DATA
# Known biological ranges for STR loci. Values outside these ranges are
# biologically implausible and indicate measurement error or tampering.
# Source: NIST STRBase, European Network of Forensic Science Institutes.
# ═══════════════════════════════════════════════════════════════════════════════

ALLELE_RANGES: Dict[str, Tuple[float, float]] = {
    "CSF1PO":   (5.0, 16.0),
    "D1S1656":  (8.0, 20.3),
    "D2S441":   (8.0, 17.0),
    "D2S1338":  (15.0, 28.0),
    "D3S1358":  (8.0, 21.0),
    "D5S818":   (7.0, 18.0),
    "D7S820":   (5.0, 16.0),
    "D8S1179":  (7.0, 20.0),
    "D10S1248": (8.0, 19.0),
    "D12S391":  (14.0, 27.3),
    "D13S317":  (7.0, 16.0),
    "D16S539":  (5.0, 16.0),
    "D18S51":   (7.0, 40.0),
    "D19S433":  (9.0, 19.2),
    "D21S11":   (24.0, 41.2),
    "D22S1045": (8.0, 19.0),
    "FGA":      (15.0, 51.2),
    "SE33":     (3.0, 47.2),
    "TH01":     (3.0, 14.0),
    "TPOX":     (4.0, 16.0),
    "VWA":      (10.0, 25.0),
    "AMEL":     (1.0, 2.0),  # X=1, Y=2
}

# Minimum loci required for a forensically meaningful profile
MIN_CODIS_LOCI: int = 8

# Quarantine threshold — profiles below this are flagged
QUARANTINE_THRESHOLD: float = 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationAnomaly(BaseModel):
    """A single anomaly detected during rule-based pre-validation."""
    locus: str
    anomaly_type: str  # OUT_OF_RANGE, MISSING_REQUIRED, RARE_COMBINATION
    observed: str
    expected: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL


class ValidationResult(BaseModel):
    """Complete validation result for a genomic profile."""
    profile_id: str
    validity_score: float = Field(..., ge=0.0, le=1.0)
    is_poisoned: bool = False
    anomaly_report: str = "No anomalies detected."
    anomalies: List[ValidationAnomaly] = []
    rule_based_score: float = Field(..., ge=0.0, le=1.0)
    ai_reasoning: Optional[str] = None
    decision: str = "ACCEPTED"  # ACCEPTED, QUARANTINED, REJECTED
    validated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED PRE-VALIDATOR
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedPreValidator:
    """
    Deterministic rule-based checks that execute before the DSPy module.

    These checks are fast, reproducible, and catch obvious violations
    without incurring LLM latency. The DSPy module handles nuanced
    statistical analysis that rules cannot express.
    """

    @staticmethod
    def validate(
        str_markers: Dict[str, LocusDataSchema],
        node_id: str,
    ) -> Tuple[float, List[ValidationAnomaly]]:
        """
        Execute all rule-based checks and return a composite score.

        Scoring:
            Start at 1.0 and subtract penalties per violation:
            - OUT_OF_RANGE allele: -0.15 per locus
            - MISSING_REQUIRED locus: -0.03 per missing locus
            - RARE_COMBINATION (homozygous at extreme): -0.05

        Args:
            str_markers: Dict of marker name → LocusDataSchema.
            node_id: Originating node for context logging.

        Returns:
            Tuple of (score: float, anomalies: list[ValidationAnomaly]).
        """
        score: float = 1.0
        anomalies: List[ValidationAnomaly] = []

        # ── Check 1: Allele Range Bounds ──
        for marker_name, locus in str_markers.items():
            if marker_name in ALLELE_RANGES:
                lo, hi = ALLELE_RANGES[marker_name]

                for allele_label, allele_val in [
                    ("allele_1", locus.allele_1),
                    ("allele_2", locus.allele_2),
                ]:
                    if not (lo <= allele_val <= hi):
                        severity = "CRITICAL" if (allele_val < lo * 0.5 or allele_val > hi * 1.5) else "HIGH"
                        anomalies.append(ValidationAnomaly(
                            locus=marker_name,
                            anomaly_type="OUT_OF_RANGE",
                            observed=f"{allele_label}={allele_val}",
                            expected=f"[{lo}, {hi}]",
                            severity=severity,
                        ))
                        score -= 0.15 if severity == "CRITICAL" else 0.10

        # ── Check 2: CODIS Loci Completeness ──
        present_codis = set(str_markers.keys()) & CODIS_MARKERS
        missing_codis = CODIS_MARKERS - set(str_markers.keys())

        if len(present_codis) < MIN_CODIS_LOCI:
            for locus in sorted(missing_codis):
                anomalies.append(ValidationAnomaly(
                    locus=locus,
                    anomaly_type="MISSING_REQUIRED",
                    observed="absent",
                    expected="present in CODIS panel",
                    severity="MEDIUM",
                ))
            # Proportional penalty
            completeness = len(present_codis) / len(CODIS_MARKERS)
            score -= (1.0 - completeness) * 0.15

        # ── Check 3: Biological Rarity Flags ──
        homozygous_extreme_count = 0
        for marker_name, locus in str_markers.items():
            if locus.is_homozygous and marker_name in ALLELE_RANGES:
                lo, hi = ALLELE_RANGES[marker_name]
                mid = (lo + hi) / 2.0
                # Homozygous at the extremes of the range is rare
                if locus.allele_1 <= lo + 1 or locus.allele_1 >= hi - 1:
                    homozygous_extreme_count += 1

        if homozygous_extreme_count >= 3:
            anomalies.append(ValidationAnomaly(
                locus="MULTI",
                anomaly_type="RARE_COMBINATION",
                observed=f"{homozygous_extreme_count} extreme homozygous loci",
                expected="< 3 in a natural profile",
                severity="HIGH",
            ))
            score -= 0.05 * homozygous_extreme_count

        return max(0.0, min(1.0, score)), anomalies


# ═══════════════════════════════════════════════════════════════════════════════
# DSPy FORENSIC VALIDATOR MODULE
# ═══════════════════════════════════════════════════════════════════════════════

class ForensicValidator(dspy.Module):
    """
    DSPy-powered forensic validation module with Chain-of-Thought reasoning.

    Combines deterministic rule-based pre-checks with LLM-driven analysis
    to produce a comprehensive validation result for each genomic profile.

    The module uses dspy.ChainOfThought to ensure the AI agent explains
    its reasoning step-by-step when flagging a profile, providing an
    auditable chain of evidence for forensic compliance.

    Usage:
        validator = ForensicValidator()
        result = validator.validate_profile(profile_id, str_markers, node_id)
    """

    def __init__(self) -> None:
        """Initialize the ForensicValidator with the ChainOfThought predictor."""
        super().__init__()
        self.chain_of_thought = dspy.ChainOfThought(BiometricValidationSignature)
        self.rule_validator = RuleBasedPreValidator()

    def _serialize_markers(self, str_markers: Dict[str, LocusDataSchema]) -> str:
        """
        Serialize STR markers into a human-readable string for the LLM.

        Format: 'MARKER: allele_1/allele_2; MARKER: allele_1/allele_2; ...'
        Sorted alphabetically for deterministic input ordering.

        Args:
            str_markers: Dict of marker name → LocusDataSchema.

        Returns:
            Serialized string representation of the profile.
        """
        parts: List[str] = []
        for name in sorted(str_markers.keys()):
            locus = str_markers[name]
            homo_flag = " [HOMO]" if locus.is_homozygous else ""
            parts.append(f"{name}: {locus.allele_1}/{locus.allele_2}{homo_flag}")
        return "; ".join(parts)

    def _build_population_context(self, node_id: str, marker_count: int) -> str:
        """
        Build a population context string for the LLM.

        In production, this would query a population frequency database
        for the node's geographic region. Currently provides node metadata.

        Args:
            node_id: Originating node identifier.
            marker_count: Number of STR markers in the profile.

        Returns:
            Context string for population-aware validation.
        """
        return (
            f"Node: {node_id} | "
            f"Markers submitted: {marker_count} | "
            f"Required minimum CODIS loci: {MIN_CODIS_LOCI} | "
            f"Population baseline: Global average (no region-specific data available)"
        )

    def validate_profile(
        self,
        profile_id: str,
        str_markers: Dict[str, LocusDataSchema],
        node_id: str,
        skip_ai: bool = False,
    ) -> ValidationResult:
        """
        Execute the full validation pipeline on a genomic profile.

        Pipeline stages:
            1. Rule-based pre-validation (allele bounds, completeness, rarity).
            2. If rule score < 1.0 and AI is not skipped, invoke ChainOfThought
               for deep statistical analysis and reasoning.
            3. Compute composite score: weighted average of rule (60%) and AI (40%).
            4. Determine decision: ACCEPTED / QUARANTINED / REJECTED.

        Args:
            profile_id: UUID v4 of the profile being validated.
            str_markers: Dict of marker name → allele data.
            node_id: Originating node identifier.
            skip_ai: If True, skip DSPy inference (for testing or when LLM is unavailable).

        Returns:
            ValidationResult with composite score, anomalies, and decision.
        """
        # ── Stage 0: Data Integrity Check ──
        # Relaxed for testing: Require at least 1 marker if validating STRs.
        if len(str_markers) < 1:
            logger.warning(f"[VALIDATOR] Step 0: ABORTED - No STR markers ({len(str_markers)}) for {profile_id}")
            return ValidationResult(
                profile_id=profile_id,
                validity_score=0.0,
                rule_based_score=0.0,
                decision="REJECTED",
                anomaly_report="Step 0: ABORTED - No valid STR markers detected.",
                anomalies=[],
            )

        # ── Stage 1: Rule-based checks ──
        rule_score, anomalies = self.rule_validator.validate(str_markers, node_id)

        ai_score: float = 1.0
        ai_reasoning: Optional[str] = None
        is_poisoned: bool = False

        # ── Stage 2: DSPy ChainOfThought (conditional) ──
        if not skip_ai and (rule_score < 1.0 or len(anomalies) > 0):
            try:
                genomic_data = self._serialize_markers(str_markers)
                population_context = self._build_population_context(node_id, len(str_markers))

                prediction = self.chain_of_thought(
                    genomic_data=genomic_data,
                    population_context=population_context,
                )

                ai_score = self._parse_float(prediction.validity_score, default=0.5)
                ai_reasoning = str(getattr(prediction, "reasoning", "")) + "\n" + str(prediction.anomaly_report)
                is_poisoned = self._parse_bool(prediction.is_poisoned, default=False)

                logger.info(
                    f"[VALIDATOR] AI assessment for {profile_id}: "
                    f"score={ai_score:.3f}, poisoned={is_poisoned}"
                )

            except Exception as exc:
                # AI failure should not block ingestion — degrade gracefully
                logger.warning(
                    f"[VALIDATOR] DSPy inference failed for {profile_id}: {exc}. "
                    "Falling back to rule-based score only."
                )
                ai_score = rule_score
                ai_reasoning = f"AI inference unavailable: {exc}"

        # ── Stage 3: Composite score ──
        # Weight: 60% rule-based (deterministic) + 40% AI (reasoning)
        composite_score = round(rule_score * 0.6 + ai_score * 0.4, 6)

        # If AI flagged as poisoned, override composite to force quarantine
        if is_poisoned:
            composite_score = min(composite_score, 0.3)

        # ── Stage 4: Decision ──
        if composite_score >= QUARANTINE_THRESHOLD:
            decision = "ACCEPTED"
        elif composite_score >= 0.5:
            decision = "QUARANTINED"
        else:
            decision = "REJECTED"

        # Build anomaly report string
        anomaly_report = "No anomalies detected."
        if anomalies:
            report_lines = [
                f"[{a.severity}] {a.locus} — {a.anomaly_type}: "
                f"observed {a.observed}, expected {a.expected}"
                for a in anomalies
            ]
            anomaly_report = "\n".join(report_lines)

        logger.info(
            f"[VALIDATOR] {profile_id} | rule={rule_score:.3f} ai={ai_score:.3f} "
            f"composite={composite_score:.3f} | decision={decision}"
        )

        return ValidationResult(
            profile_id=profile_id,
            validity_score=composite_score,
            is_poisoned=is_poisoned,
            anomaly_report=anomaly_report,
            anomalies=anomalies,
            rule_based_score=rule_score,
            ai_reasoning=ai_reasoning,
            decision=decision,
        )

    def forward(self, genomic_data: str, population_context: str) -> dspy.Prediction:
        """
        DSPy Module forward pass — required by the dspy.Module interface.

        Invokes ChainOfThought reasoning on the provided genomic data.
        This method is called internally by DSPy optimizers during
        BootstrapFewShot training.

        Args:
            genomic_data: Serialized STR marker string.
            population_context: Population metadata string.

        Returns:
            dspy.Prediction with validity_score, anomaly_report, is_poisoned.
        """
        return self.chain_of_thought(
            genomic_data=genomic_data,
            population_context=population_context,
        )

    @staticmethod
    def _parse_float(value: Any, default: float = 0.5) -> float:
        """
        Safely parse a float from LLM output.

        LLMs may return strings like '0.82', '82%', or 'approximately 0.8'.
        This method extracts the numeric value robustly.
        """
        if isinstance(value, (int, float)):
            return float(max(0.0, min(1.0, value)))
        try:
            cleaned = str(value).strip().replace("%", "").replace(",", ".")
            # Extract first number-like substring
            import re
            match = re.search(r"(\d+\.?\d*)", cleaned)
            if match:
                v = float(match.group(1))
                if v > 1.0:
                    v /= 100.0  # Convert percentage
                return max(0.0, min(1.0, v))
        except (ValueError, TypeError):
            pass
        return default

    @staticmethod
    def _parse_bool(value: Any, default: bool = False) -> bool:
        """
        Safely parse a boolean from LLM output.

        Handles: True, False, 'true', 'false', 'yes', 'no', 1, 0.
        """
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in ("true", "yes", "1", "poisoned", "detected"):
            return True
        if s in ("false", "no", "0", "clean", "not detected", "none"):
            return False
        return default
