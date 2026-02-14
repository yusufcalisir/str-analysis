"""
DSPy Optimizer Stub — BootstrapFewShot for ForensicValidator.

This module provides the scaffolding for training the ForensicValidator
agent on historical forensic records. The optimizer learns from
"true positive" validated profiles (confirmed by human forensic
analysts) to improve the DSPy module's accuracy over time.

Architecture:
    BootstrapFewShot generates optimized few-shot demonstrations from
    a training set of (input, expected_output) pairs. These demonstrations
    are injected into the ChainOfThought prompt at inference time,
    teaching the LLM the forensic domain's validation patterns.

Usage (future):
    1. Collect validated profiles from PostgreSQL audit log.
    2. Build training examples with known validity scores.
    3. Run optimize() to generate optimized prompts.
    4. Save the optimized module for production deployment.
"""

import logging
from typing import Any, Dict, List, Optional

import dspy
from pydantic import BaseModel, Field

from app.agents.forensic_validator import ForensicValidator

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING DATA MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class TrainingExample(BaseModel):
    """A single training example for the optimizer."""
    genomic_data: str = Field(..., description="Serialized STR marker string")
    population_context: str = Field(..., description="Population metadata string")
    expected_validity_score: float = Field(..., ge=0.0, le=1.0)
    expected_is_poisoned: bool = False
    expected_anomaly_report: str = "No anomalies detected."
    source: str = "manual"  # manual, audit_log, synthetic


class OptimizationResult(BaseModel):
    """Result of an optimization run."""
    examples_used: int
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None
    optimized: bool = False
    save_path: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION METRIC
# ═══════════════════════════════════════════════════════════════════════════════

def forensic_validation_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace: Optional[Any] = None,
) -> float:
    """
    Custom metric for evaluating ForensicValidator predictions.

    Scoring criteria:
        - Validity score accuracy: absolute error < 0.1 → full marks (0.5 weight)
        - Poisoning detection: exact match → full marks (0.35 weight)
        - Anomaly report quality: non-empty when expected → full marks (0.15 weight)

    Args:
        example: Ground truth dspy.Example.
        prediction: Model prediction.
        trace: Optional trace for debugging.

    Returns:
        Composite metric score between 0.0 and 1.0.
    """
    score = 0.0

    # ── Validity score accuracy ──
    try:
        predicted_score = float(prediction.validity_score)
        expected_score = float(example.expected_validity_score)
        error = abs(predicted_score - expected_score)
        if error < 0.1:
            score += 0.5
        elif error < 0.2:
            score += 0.3
        elif error < 0.3:
            score += 0.1
    except (ValueError, AttributeError):
        pass

    # ── Poisoning detection accuracy ──
    try:
        predicted_poisoned = str(prediction.is_poisoned).lower() in ("true", "yes", "1")
        expected_poisoned = bool(example.expected_is_poisoned)
        if predicted_poisoned == expected_poisoned:
            score += 0.35
    except (ValueError, AttributeError):
        pass

    # ── Anomaly report quality ──
    try:
        report = str(prediction.anomaly_report).strip()
        expected_report = str(example.expected_anomaly_report).strip()
        if expected_report != "No anomalies detected.":
            if report and report != "No anomalies detected.":
                score += 0.15
        else:
            if report == "No anomalies detected." or report == "":
                score += 0.15
    except (ValueError, AttributeError):
        pass

    return score


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

class ForensicValidatorOptimizer:
    """
    BootstrapFewShot optimizer for the ForensicValidator DSPy module.

    Collects validated forensic records as training examples and uses
    DSPy's BootstrapFewShot teleprompter to generate optimized
    few-shot demonstrations that improve the module's accuracy.

    Lifecycle:
        1. add_example() — Collect training data from audit logs.
        2. optimize() — Run BootstrapFewShot to generate optimized prompts.
        3. save() / load() — Persist the optimized module for production use.
    """

    def __init__(self, max_bootstrapped_demos: int = 8, max_labeled_demos: int = 4) -> None:
        """
        Initialize the optimizer.

        Args:
            max_bootstrapped_demos: Maximum number of auto-generated demonstrations.
            max_labeled_demos: Maximum number of human-labeled demonstrations.
        """
        self._max_bootstrapped = max_bootstrapped_demos
        self._max_labeled = max_labeled_demos
        self._training_examples: List[dspy.Example] = []
        self._optimized_module: Optional[ForensicValidator] = None

    def add_example(self, example: TrainingExample) -> None:
        """
        Add a training example from validated forensic records.

        Converts the Pydantic model to a dspy.Example with input/output
        field annotations for the optimizer.

        Args:
            example: TrainingExample with ground truth values.
        """
        dspy_example = dspy.Example(
            genomic_data=example.genomic_data,
            population_context=example.population_context,
            expected_validity_score=example.expected_validity_score,
            expected_is_poisoned=example.expected_is_poisoned,
            expected_anomaly_report=example.expected_anomaly_report,
        ).with_inputs("genomic_data", "population_context")

        self._training_examples.append(dspy_example)
        logger.info(
            f"[OPTIMIZER] Added training example from '{example.source}'. "
            f"Total examples: {len(self._training_examples)}"
        )

    def add_examples_batch(self, examples: List[TrainingExample]) -> int:
        """
        Add multiple training examples at once.

        Args:
            examples: List of TrainingExample objects.

        Returns:
            Total number of training examples after addition.
        """
        for ex in examples:
            self.add_example(ex)
        return len(self._training_examples)

    def optimize(self, validator: Optional[ForensicValidator] = None) -> OptimizationResult:
        """
        Run BootstrapFewShot optimization on the ForensicValidator.

        Requires at least 4 training examples to generate meaningful
        demonstrations. The optimizer evaluates the module's predictions
        against ground truth using forensic_validation_metric.

        Args:
            validator: Optional pre-initialized ForensicValidator. If None,
                a new instance is created.

        Returns:
            OptimizationResult with before/after metrics.
        """
        if len(self._training_examples) < 4:
            logger.warning(
                f"[OPTIMIZER] Insufficient training data: {len(self._training_examples)}/4 minimum. "
                "Skipping optimization."
            )
            return OptimizationResult(
                examples_used=len(self._training_examples),
                optimized=False,
            )

        module = validator or ForensicValidator()

        try:
            teleprompter = dspy.BootstrapFewShot(
                metric=forensic_validation_metric,
                max_bootstrapped_demos=self._max_bootstrapped,
                max_labeled_demos=self._max_labeled,
            )

            self._optimized_module = teleprompter.compile(
                module,
                trainset=self._training_examples,
            )

            logger.info(
                f"[OPTIMIZER] Optimization complete. "
                f"Used {len(self._training_examples)} examples."
            )

            return OptimizationResult(
                examples_used=len(self._training_examples),
                optimized=True,
            )

        except Exception as exc:
            logger.error(f"[OPTIMIZER] Optimization failed: {exc}")
            return OptimizationResult(
                examples_used=len(self._training_examples),
                optimized=False,
            )

    def get_optimized_module(self) -> Optional[ForensicValidator]:
        """
        Return the optimized ForensicValidator module.

        Returns None if optimize() has not been successfully called.
        """
        return self._optimized_module

    def save(self, path: str) -> None:
        """
        Save the optimized module state to disk.

        Args:
            path: File path for the serialized module (JSON format).
        """
        if self._optimized_module is None:
            raise RuntimeError("No optimized module to save. Run optimize() first.")
        self._optimized_module.save(path)
        logger.info(f"[OPTIMIZER] Saved optimized module to {path}")

    def load(self, path: str) -> ForensicValidator:
        """
        Load a previously optimized module from disk.

        Args:
            path: File path to the serialized module.

        Returns:
            The loaded ForensicValidator module with optimized prompts.
        """
        module = ForensicValidator()
        module.load(path)
        self._optimized_module = module
        logger.info(f"[OPTIMIZER] Loaded optimized module from {path}")
        return module
