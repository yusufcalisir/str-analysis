"""
VANTAGE Flower Client — Federated Learning Client for Sovereign Nodes.

Implements the client-side Federated Learning protocol inspired by
Flower (flwr) and PySyft patterns. Connects to the Global Orchestrator
to participate in FL rounds without sharing raw DNA data.

FL Protocol:
    1. get_parameters() → Return current local model weights.
    2. fit()            → Train on local data, return updated weights.
    3. evaluate()       → Evaluate global model on local validation set.

The "model" here is the DSPy ForensicValidator from Phase 1.3.
Federated fine-tuning improves validation accuracy across diverse
population distributions without centralizing genomic data.
"""

import hashlib
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_LOCAL_EPOCHS: int = 3
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_LEARNING_RATE: float = 0.001
MODEL_DIM: int = 48  # Aligned with vector dimensionality


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ClientStatus(str, Enum):
    """Lifecycle state of the FL client."""
    IDLE = "idle"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    TRAINING = "training"
    EVALUATING = "evaluating"
    UPLOADING = "uploading"
    ERROR = "error"


class FitResult(BaseModel):
    """Result of a local training round."""
    parameters: List[float]
    num_examples: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    training_time_ms: float = 0.0
    round_id: int = 0


class EvaluateResult(BaseModel):
    """Result of a local evaluation."""
    loss: float
    accuracy: float
    num_examples: int
    metrics: Dict[str, Any] = Field(default_factory=dict)
    evaluation_time_ms: float = 0.0


class FLRoundInfo(BaseModel):
    """Metadata for a federated learning round."""
    round_id: int
    global_round: int
    server_round: int = 0
    status: str = "pending"
    started_at: float = 0.0
    completed_at: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL MODEL PROXY
# ═══════════════════════════════════════════════════════════════════════════════

class LocalModelProxy:
    """
    Proxy for the local ForensicValidator model weights.

    Wraps the DSPy validator's internal parameters into a flat
    float vector suitable for federated aggregation. In a full
    deployment, this maps to the ChainOfThought module's learned
    prompt parameters and scoring weights.

    For the current phase, we simulate model parameters as a
    weight vector that can be averaged across nodes (FedAvg).
    """

    def __init__(self, dim: int = MODEL_DIM) -> None:
        self._dim = dim
        # Initialize with small random weights (simulating learned params)
        self._weights: NDArray[np.float64] = np.random.randn(dim).astype(np.float64) * 0.01
        self._bias: float = 0.0
        self._version: int = 0

    def get_parameters(self) -> List[float]:
        """Flatten all learnable parameters into a single float list."""
        return list(self._weights) + [self._bias]

    def set_parameters(self, params: List[float]) -> None:
        """Load parameters from a flat float list (e.g., from global model)."""
        if len(params) != self._dim + 1:
            raise ValueError(
                f"Expected {self._dim + 1} params, got {len(params)}"
            )
        self._weights = np.array(params[:-1], dtype=np.float64)
        self._bias = params[-1]
        self._version += 1

    def train_step(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.float64],
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> float:
        """
        Single gradient descent step on local data.

        Uses a simple linear model: y_hat = X @ w + b
        Loss: MSE for regression-style validation scoring.

        Args:
            features: (N, dim) feature matrix.
            labels: (N,) target validation scores.
            learning_rate: Step size for gradient descent.

        Returns:
            Training loss (MSE) for this step.
        """
        n = features.shape[0]
        y_hat = features @ self._weights + self._bias
        error = y_hat - labels
        loss = float(np.mean(error ** 2))

        # Gradients
        grad_w = (2.0 / n) * (features.T @ error)
        grad_b = (2.0 / n) * np.sum(error)

        # Update
        self._weights -= learning_rate * grad_w
        self._bias -= learning_rate * grad_b

        return loss

    def evaluate(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.

        Returns:
            Tuple of (loss, accuracy). Accuracy is the fraction of
            predictions within 0.1 of the true label.
        """
        y_hat = features @ self._weights + self._bias
        error = y_hat - labels
        loss = float(np.mean(error ** 2))

        # "Accuracy" = predictions within tolerance of true score
        tolerance = 0.1
        correct = np.sum(np.abs(error) < tolerance)
        accuracy = float(correct / len(labels))

        return loss, accuracy

    @property
    def version(self) -> int:
        return self._version

    def parameter_hash(self) -> str:
        """SHA-256 hash of current parameters for integrity checking."""
        raw = self._weights.tobytes() + np.float64(self._bias).tobytes()
        return hashlib.sha256(raw).hexdigest()[:16]


# ═══════════════════════════════════════════════════════════════════════════════
# VANTAGE FLOWER CLIENT
# ═══════════════════════════════════════════════════════════════════════════════

class VantageFlowerClient:
    """
    Federated Learning client for the VANTAGE-STR sovereign node.

    Implements the Flower client protocol (get_parameters, fit, evaluate)
    for participating in global FL rounds. The local "model" is the
    ForensicValidator's scoring parameters.

    Privacy Guarantee:
        Only model parameters (weight vectors) are transmitted.
        No raw STR data, profile_ids, or validation results leave the node.

    Usage:
        client = VantageFlowerClient(node_id="TR-NODE-01")
        client.connect(orchestrator_url)
        params = client.get_parameters()
        fit_result = client.fit(global_params, round_config)
        eval_result = client.evaluate(global_params)
    """

    def __init__(
        self,
        node_id: str,
        local_epochs: int = DEFAULT_LOCAL_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
    ) -> None:
        self._node_id = node_id
        self._model = LocalModelProxy()
        self._status = ClientStatus.IDLE
        self._local_epochs = local_epochs
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._round_history: List[FLRoundInfo] = []
        self._current_round: int = 0
        self._orchestrator_url: str = ""
        self._training_data: Optional[Tuple[NDArray, NDArray]] = None

    @property
    def status(self) -> ClientStatus:
        return self._status

    @property
    def node_id(self) -> str:
        return self._node_id

    def connect(self, orchestrator_url: str) -> bool:
        """
        Connect to the Global Orchestrator.

        In production, this establishes the mTLS gRPC channel.
        Currently sets the connection state for protocol readiness.

        Args:
            orchestrator_url: gRPC endpoint of the orchestrator.

        Returns:
            True if connection was established.
        """
        self._status = ClientStatus.CONNECTING
        self._orchestrator_url = orchestrator_url

        # In production: establish gRPC channel with mTLS
        # channel = grpc.secure_channel(url, credentials)

        self._status = ClientStatus.CONNECTED
        logger.info(
            f"[FL-CLIENT] {self._node_id} connected to {orchestrator_url} | "
            f"model_v={self._model.version} | hash={self._model.parameter_hash()}"
        )
        return True

    def disconnect(self) -> None:
        """Disconnect from the orchestrator."""
        self._status = ClientStatus.IDLE
        self._orchestrator_url = ""
        logger.info(f"[FL-CLIENT] {self._node_id} disconnected")

    def load_training_data(
        self,
        features: NDArray[np.float64],
        labels: NDArray[np.float64],
    ) -> None:
        """
        Load local training data for FL rounds.

        In production, this is derived from historical validation
        results stored locally. Features are 48-dim embeddings,
        labels are DSPy validity scores.
        """
        assert features.shape[1] == MODEL_DIM, (
            f"Features must have {MODEL_DIM} columns, got {features.shape[1]}"
        )
        assert len(features) == len(labels), "Feature/label count mismatch"
        self._training_data = (features, labels)
        logger.info(
            f"[FL-CLIENT] Loaded {len(labels)} training examples for {self._node_id}"
        )

    def get_parameters(self) -> List[float]:
        """
        Return the current local model parameters.

        Called by the orchestrator at the start of each FL round
        to collect the baseline before training.

        Returns:
            Flat list of model weights + bias.
        """
        params = self._model.get_parameters()
        logger.info(
            f"[FL-CLIENT] get_parameters() → {len(params)} params | "
            f"hash={self._model.parameter_hash()}"
        )
        return params

    def fit(
        self,
        global_parameters: List[float],
        config: Optional[Dict[str, Any]] = None,
    ) -> FitResult:
        """
        Train the local model on local data, starting from global parameters.

        Flower-protocol fit():
            1. Load global parameters into local model.
            2. Train for `local_epochs` on local data.
            3. Return updated parameters + metrics.

        Args:
            global_parameters: Current global model parameters from FedAvg.
            config: Round configuration (epochs, lr, etc.).

        Returns:
            FitResult with updated local parameters.
        """
        self._status = ClientStatus.TRAINING
        self._current_round += 1
        t_start = time.perf_counter()

        # Apply round config overrides
        epochs = (config or {}).get("local_epochs", self._local_epochs)
        lr = (config or {}).get("learning_rate", self._learning_rate)

        # Load global model
        self._model.set_parameters(global_parameters)

        # Generate synthetic training data if none loaded
        if self._training_data is None:
            n_samples = 200
            features = np.random.randn(n_samples, MODEL_DIM).astype(np.float64) * 0.1
            labels = np.random.rand(n_samples).astype(np.float64)
            self._training_data = (features, labels)

        features, labels = self._training_data
        n_samples = len(labels)

        # Training loop
        total_loss = 0.0
        for epoch in range(epochs):
            # Shuffle
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for batch_start in range(0, n_samples, self._batch_size):
                batch_end = min(batch_start + self._batch_size, n_samples)
                batch_idx = indices[batch_start:batch_end]

                batch_features = features[batch_idx]
                batch_labels = labels[batch_idx]

                loss = self._model.train_step(batch_features, batch_labels, lr)
                epoch_loss += loss
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            total_loss += avg_epoch_loss

        t_end = time.perf_counter()
        training_ms = (t_end - t_start) * 1000
        avg_loss = total_loss / max(epochs, 1)

        self._status = ClientStatus.CONNECTED

        # Record round
        round_info = FLRoundInfo(
            round_id=self._current_round,
            global_round=self._current_round,
            status="completed",
            started_at=t_start,
            completed_at=t_end,
        )
        self._round_history.append(round_info)

        result = FitResult(
            parameters=self._model.get_parameters(),
            num_examples=n_samples,
            metrics={
                "loss": round(avg_loss, 6),
                "epochs": epochs,
                "learning_rate": lr,
                "model_version": self._model.version,
                "param_hash": self._model.parameter_hash(),
            },
            training_time_ms=round(training_ms, 2),
            round_id=self._current_round,
        )

        logger.info(
            f"[FL-CLIENT] fit() round {self._current_round} | "
            f"loss={avg_loss:.6f} | {epochs} epochs | "
            f"{n_samples} samples | {training_ms:.1f}ms"
        )

        return result

    def evaluate(
        self,
        global_parameters: List[float],
        config: Optional[Dict[str, Any]] = None,
    ) -> EvaluateResult:
        """
        Evaluate the global model on local validation data.

        Flower-protocol evaluate():
            1. Load global parameters.
            2. Evaluate on local held-out data.
            3. Return loss + accuracy metrics.

        Args:
            global_parameters: Global model parameters to evaluate.
            config: Evaluation configuration.

        Returns:
            EvaluateResult with loss, accuracy, and metrics.
        """
        self._status = ClientStatus.EVALUATING
        t_start = time.perf_counter()

        self._model.set_parameters(global_parameters)

        # Use portion of training data as validation
        if self._training_data is not None:
            features, labels = self._training_data
            # Use last 20% as validation
            split = int(len(labels) * 0.8)
            val_features = features[split:]
            val_labels = labels[split:]
        else:
            val_features = np.random.randn(50, MODEL_DIM).astype(np.float64) * 0.1
            val_labels = np.random.rand(50).astype(np.float64)

        loss, accuracy = self._model.evaluate(val_features, val_labels)

        t_end = time.perf_counter()
        eval_ms = (t_end - t_start) * 1000

        self._status = ClientStatus.CONNECTED

        result = EvaluateResult(
            loss=round(loss, 6),
            accuracy=round(accuracy, 4),
            num_examples=len(val_labels),
            metrics={
                "model_version": float(self._model.version),
                "param_hash": self._model.parameter_hash(),
            },
            evaluation_time_ms=round(eval_ms, 2),
        )

        logger.info(
            f"[FL-CLIENT] evaluate() | loss={loss:.6f} | "
            f"accuracy={accuracy:.4f} | {len(val_labels)} samples | {eval_ms:.1f}ms"
        )

        return result

    def get_round_history(self) -> List[FLRoundInfo]:
        """Return the history of all FL rounds this client participated in."""
        return self._round_history

    def get_client_info(self) -> Dict[str, Any]:
        """Return client metadata for the admin UI."""
        return {
            "node_id": self._node_id,
            "status": self._status.value,
            "current_round": self._current_round,
            "total_rounds": len(self._round_history),
            "orchestrator_url": self._orchestrator_url,
            "model_version": self._model.version,
            "model_param_hash": self._model.parameter_hash(),
            "training_samples": len(self._training_data[1]) if self._training_data else 0,
        }
