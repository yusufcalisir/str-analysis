"""
Federated Learning Strategy — VantageStrategy for Cross-Border STR Matching.

Implements a Federated Averaging (FedAvg) variant designed for privacy-
preserving forensic DNA queries across the VANTAGE-STR network. Instead
of sharing raw DNA profiles, nodes receive a query embedding and return
only similarity confidence scores and optional local model gradient
updates.

Architecture (inspired by Flower/PySyft):
    1. Orchestrator broadcasts a 'Query Embedding' (48-dim vector) to all
       online nodes.
    2. Each node performs a local cosine similarity search against its
       private Milvus collection.
    3. Nodes return 'NodeSearchResult' objects containing only confidence
       scores, match counts, and optional model gradients — never raw DNA.
    4. The Aggregator combines scores using configurable weighting to
       produce a global ranking of the most likely matches.

Privacy Guarantee:
    No raw STR data ever leaves a node. The orchestrator sees only float
    scores and gradient tensors, making it impossible to reconstruct the
    original genomic profiles from the aggregated results.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

VECTOR_DIM: int = 48
MIN_CONFIDENCE_THRESHOLD: float = 0.65
MAX_GLOBAL_RESULTS: int = 50
AGGREGATION_TIMEOUT_SECONDS: float = 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedQueryStatus(str, Enum):
    """Lifecycle state of a federated query round."""
    INITIATED = "initiated"
    BROADCASTING = "broadcasting"
    COLLECTING = "collecting"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class NodeSearchResult(BaseModel):
    """
    Result payload returned by a single node after local search.

    Contains only similarity scores and metadata — never raw DNA data.
    """
    node_id: str
    query_id: str
    matches: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "List of local matches. Each entry: "
            "{'profile_id': str, 'confidence': float, 'distance': float}"
        ),
    )
    local_search_time_ms: float = 0.0
    profiles_searched: int = 0
    model_gradient: Optional[List[float]] = None  # Optional FL gradient update


class AggregatedMatch(BaseModel):
    """A globally-ranked match after cross-node aggregation."""
    profile_id: str
    source_node_id: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    distance: float = 0.0
    contributing_nodes: int = 1
    rank: int = 0


class FederatedQueryRound(BaseModel):
    """Complete state of a single federated query round."""
    query_id: str
    query_embedding: List[float]
    status: FederatedQueryStatus = FederatedQueryStatus.INITIATED
    target_nodes: List[str] = Field(default_factory=list)
    responded_nodes: List[str] = Field(default_factory=list)
    results: List[AggregatedMatch] = Field(default_factory=list)
    total_profiles_searched: int = 0
    initiated_at: float = Field(default_factory=time.time)
    completed_at: Optional[float] = None
    round_time_ms: float = 0.0


class ModelUpdate(BaseModel):
    """Federated model gradient update from a participating node."""
    node_id: str
    round_id: str
    gradient: List[float]
    sample_count: int
    loss: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# AGGREGATION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════════

class AggregationMethod(str, Enum):
    """Available score aggregation methods."""
    WEIGHTED_AVERAGE = "weighted_average"
    MAX_CONFIDENCE = "max_confidence"
    BAYESIAN = "bayesian"


class ScoreAggregator:
    """
    Combines confidence scores from multiple nodes into a global ranking.

    Supports multiple aggregation strategies:
    - weighted_average: Weights by node profile count (larger DBs = higher weight).
    - max_confidence: Takes the single highest confidence per profile_id.
    - bayesian: Bayesian update treating each node result as independent evidence.
    """

    def __init__(self, method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE) -> None:
        self._method = method

    def aggregate(
        self,
        node_results: List[NodeSearchResult],
        node_weights: Optional[Dict[str, float]] = None,
    ) -> List[AggregatedMatch]:
        """
        Aggregate search results from multiple nodes.

        Args:
            node_results: List of per-node search results.
            node_weights: Optional weights per node_id (e.g., by profile count).

        Returns:
            Globally-ranked list of AggregatedMatch objects.
        """
        if self._method == AggregationMethod.WEIGHTED_AVERAGE:
            return self._weighted_average(node_results, node_weights)
        elif self._method == AggregationMethod.MAX_CONFIDENCE:
            return self._max_confidence(node_results)
        elif self._method == AggregationMethod.BAYESIAN:
            return self._bayesian_aggregate(node_results)
        else:
            return self._weighted_average(node_results, node_weights)

    def _weighted_average(
        self,
        results: List[NodeSearchResult],
        weights: Optional[Dict[str, float]],
    ) -> List[AggregatedMatch]:
        """
        Weight each node's confidence by its database size.

        Nodes with more profiles are given higher weight, reflecting the
        statistical significance of a match in a larger database.
        """
        profile_scores: Dict[str, Dict[str, Any]] = {}

        for nr in results:
            w = 1.0
            if weights and nr.node_id in weights:
                w = weights[nr.node_id]

            for match in nr.matches:
                pid = match["profile_id"]
                conf = match.get("confidence", 0.0) * w

                if pid not in profile_scores:
                    profile_scores[pid] = {
                        "weighted_sum": 0.0,
                        "weight_total": 0.0,
                        "source_node": nr.node_id,
                        "distance": match.get("distance", 0.0),
                        "nodes": 0,
                    }

                profile_scores[pid]["weighted_sum"] += conf
                profile_scores[pid]["weight_total"] += w
                profile_scores[pid]["nodes"] += 1

        # Compute final scores
        aggregated: List[AggregatedMatch] = []
        for pid, data in profile_scores.items():
            final_conf = data["weighted_sum"] / max(data["weight_total"], 1e-8)
            aggregated.append(AggregatedMatch(
                profile_id=pid,
                source_node_id=data["source_node"],
                confidence=round(min(final_conf, 1.0), 6),
                distance=data["distance"],
                contributing_nodes=data["nodes"],
            ))

        # Sort by confidence descending, assign ranks
        aggregated.sort(key=lambda m: m.confidence, reverse=True)
        for i, m in enumerate(aggregated):
            m.rank = i + 1

        return aggregated[:MAX_GLOBAL_RESULTS]

    def _max_confidence(self, results: List[NodeSearchResult]) -> List[AggregatedMatch]:
        """Take the single highest confidence score per profile_id."""
        best: Dict[str, AggregatedMatch] = {}

        for nr in results:
            for match in nr.matches:
                pid = match["profile_id"]
                conf = match.get("confidence", 0.0)

                if pid not in best or conf > best[pid].confidence:
                    best[pid] = AggregatedMatch(
                        profile_id=pid,
                        source_node_id=nr.node_id,
                        confidence=round(conf, 6),
                        distance=match.get("distance", 0.0),
                        contributing_nodes=1,
                    )

        aggregated = sorted(best.values(), key=lambda m: m.confidence, reverse=True)
        for i, m in enumerate(aggregated):
            m.rank = i + 1

        return aggregated[:MAX_GLOBAL_RESULTS]

    def _bayesian_aggregate(self, results: List[NodeSearchResult]) -> List[AggregatedMatch]:
        """
        Bayesian evidence aggregation.

        Treats each node's confidence as independent evidence and updates
        a prior belief using Bayes' theorem. Prior is set to the population
        base rate of a random match (1 in 10^15 for 20-locus STR profiles).
        """
        prior = 1e-15  # Random match probability for full STR panel
        profile_evidence: Dict[str, Dict[str, Any]] = {}

        for nr in results:
            for match in nr.matches:
                pid = match["profile_id"]
                conf = match.get("confidence", 0.0)

                if pid not in profile_evidence:
                    profile_evidence[pid] = {
                        "posterior": prior,
                        "source_node": nr.node_id,
                        "distance": match.get("distance", 0.0),
                        "nodes": 0,
                    }

                # Bayesian update: P(match|evidence) ∝ P(evidence|match) × P(match)
                likelihood = conf
                posterior = profile_evidence[pid]["posterior"]
                numerator = likelihood * posterior
                denominator = numerator + (1 - likelihood) * (1 - posterior)
                profile_evidence[pid]["posterior"] = numerator / max(denominator, 1e-20)
                profile_evidence[pid]["nodes"] += 1

        aggregated: List[AggregatedMatch] = []
        for pid, data in profile_evidence.items():
            aggregated.append(AggregatedMatch(
                profile_id=pid,
                source_node_id=data["source_node"],
                confidence=round(min(data["posterior"], 1.0), 6),
                distance=data["distance"],
                contributing_nodes=data["nodes"],
            ))

        aggregated.sort(key=lambda m: m.confidence, reverse=True)
        for i, m in enumerate(aggregated):
            m.rank = i + 1

        return aggregated[:MAX_GLOBAL_RESULTS]


# ═══════════════════════════════════════════════════════════════════════════════
# FEDERATED MODEL AVERAGING
# ═══════════════════════════════════════════════════════════════════════════════

class FedAvgAggregator:
    """
    Federated Averaging (FedAvg) for model gradient aggregation.

    When nodes return local model updates alongside search results,
    this aggregator computes a weighted average of the gradient vectors
    to produce a global model update that can be broadcast back.

    Weights are proportional to the number of samples each node trained on,
    following the McMahan et al. (2017) FedAvg formulation.
    """

    @staticmethod
    def aggregate_gradients(updates: List[ModelUpdate]) -> Optional[NDArray[np.float64]]:
        """
        Compute the weighted average of gradient vectors.

        Args:
            updates: List of ModelUpdate from participating nodes.

        Returns:
            Averaged gradient vector, or None if no valid updates.
        """
        if not updates:
            return None

        total_samples = sum(u.sample_count for u in updates)
        if total_samples == 0:
            return None

        # Initialize with zeros
        dim = len(updates[0].gradient)
        global_gradient = np.zeros(dim, dtype=np.float64)

        for update in updates:
            weight = update.sample_count / total_samples
            local_grad = np.array(update.gradient, dtype=np.float64)
            global_gradient += weight * local_grad

        logger.info(
            f"[FEDAVG] Aggregated {len(updates)} gradients | "
            f"total_samples={total_samples} | dim={dim} | "
            f"norm={np.linalg.norm(global_gradient):.6f}"
        )

        return global_gradient


# ═══════════════════════════════════════════════════════════════════════════════
# VANTAGE STRATEGY — ORCHESTRATION LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

class VantageStrategy:
    """
    Central federated strategy for VANTAGE-STR query orchestration.

    Orchestrates the full lifecycle of a cross-border forensic query:
    1. Initiate a query round with a 48-dim embedding.
    2. Broadcast to all eligible nodes via the NodeManager.
    3. Collect NodeSearchResult responses.
    4. Aggregate scores into a global ranking.
    5. Optionally aggregate model gradients (FedAvg).
    6. Record metrics and finalize the round.

    Usage:
        strategy = VantageStrategy()
        round = strategy.initiate_query(query_id, embedding, target_nodes)
        strategy.receive_result(round.query_id, node_result)
        final = strategy.finalize_round(round.query_id)
    """

    def __init__(
        self,
        aggregation_method: AggregationMethod = AggregationMethod.WEIGHTED_AVERAGE,
        min_confidence: float = MIN_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._score_aggregator = ScoreAggregator(aggregation_method)
        self._fedavg = FedAvgAggregator()
        self._min_confidence = min_confidence
        self._active_rounds: Dict[str, FederatedQueryRound] = {}
        self._pending_results: Dict[str, List[NodeSearchResult]] = {}
        self._pending_gradients: Dict[str, List[ModelUpdate]] = {}

    def initiate_query(
        self,
        query_id: str,
        query_embedding: List[float],
        target_node_ids: List[str],
    ) -> FederatedQueryRound:
        """
        Start a new federated query round.

        Args:
            query_id: Unique identifier for this query.
            query_embedding: 48-dim normalized STR vector.
            target_node_ids: List of node IDs to broadcast to.

        Returns:
            FederatedQueryRound tracking the lifecycle.
        """
        if len(query_embedding) != VECTOR_DIM:
            raise ValueError(f"Embedding must be {VECTOR_DIM}-dim, got {len(query_embedding)}")

        query_round = FederatedQueryRound(
            query_id=query_id,
            query_embedding=query_embedding,
            status=FederatedQueryStatus.BROADCASTING,
            target_nodes=target_node_ids,
        )

        self._active_rounds[query_id] = query_round
        self._pending_results[query_id] = []
        self._pending_gradients[query_id] = []

        logger.info(
            f"[STRATEGY] Query {query_id} INITIATED → "
            f"broadcasting to {len(target_node_ids)} nodes"
        )

        return query_round

    def receive_result(
        self,
        query_id: str,
        result: NodeSearchResult,
        model_update: Optional[ModelUpdate] = None,
    ) -> bool:
        """
        Receive a search result from a responding node.

        Args:
            query_id: ID of the active query round.
            result: NodeSearchResult with local matches.
            model_update: Optional gradient update from the node.

        Returns:
            True if the result was accepted, False if round not found.
        """
        if query_id not in self._active_rounds:
            logger.warning(f"[STRATEGY] Result for unknown query: {query_id}")
            return False

        query_round = self._active_rounds[query_id]
        query_round.status = FederatedQueryStatus.COLLECTING
        query_round.responded_nodes.append(result.node_id)
        query_round.total_profiles_searched += result.profiles_searched

        self._pending_results[query_id].append(result)

        if model_update:
            self._pending_gradients[query_id].append(model_update)

        logger.info(
            f"[STRATEGY] Query {query_id} ← {result.node_id}: "
            f"{len(result.matches)} matches ({result.local_search_time_ms:.1f}ms)"
        )

        return True

    def finalize_round(
        self,
        query_id: str,
        node_weights: Optional[Dict[str, float]] = None,
    ) -> FederatedQueryRound:
        """
        Finalize a query round: aggregate scores and gradients.

        Should be called after all expected nodes have responded or
        the aggregation timeout has elapsed.

        Args:
            query_id: ID of the round to finalize.
            node_weights: Optional per-node weights for aggregation.

        Returns:
            Completed FederatedQueryRound with global results.
        """
        if query_id not in self._active_rounds:
            raise ValueError(f"Query round {query_id} not found")

        query_round = self._active_rounds[query_id]
        query_round.status = FederatedQueryStatus.AGGREGATING

        # Aggregate scores
        results = self._pending_results.get(query_id, [])
        aggregated = self._score_aggregator.aggregate(results, node_weights)

        # Filter by minimum confidence
        query_round.results = [m for m in aggregated if m.confidence >= self._min_confidence]
        # Re-rank after filtering
        for i, m in enumerate(query_round.results):
            m.rank = i + 1

        # Aggregate gradients (if any)
        gradients = self._pending_gradients.get(query_id, [])
        if gradients:
            global_grad = self._fedavg.aggregate_gradients(gradients)
            # Global gradient available for broadcast back to nodes
            logger.info(
                f"[STRATEGY] Global gradient computed for round {query_id} "
                f"(norm={np.linalg.norm(global_grad):.6f})" if global_grad is not None else ""
            )

        # Finalize timing
        query_round.completed_at = time.time()
        query_round.round_time_ms = (query_round.completed_at - query_round.initiated_at) * 1000
        query_round.status = FederatedQueryStatus.COMPLETED

        # Cleanup
        self._pending_results.pop(query_id, None)
        self._pending_gradients.pop(query_id, None)

        logger.info(
            f"[STRATEGY] Query {query_id} COMPLETED | "
            f"{len(query_round.results)} global matches | "
            f"{query_round.total_profiles_searched:,} profiles searched | "
            f"{query_round.round_time_ms:.1f}ms"
        )

        return query_round

    def check_timeouts(self) -> List[str]:
        """
        Check for query rounds that have exceeded the aggregation timeout.

        Returns:
            List of query_ids that were force-completed due to timeout.
        """
        now = time.time()
        timed_out: List[str] = []

        for qid, qr in list(self._active_rounds.items()):
            if qr.status in (FederatedQueryStatus.COMPLETED, FederatedQueryStatus.FAILED):
                continue

            elapsed = now - qr.initiated_at
            if elapsed > AGGREGATION_TIMEOUT_SECONDS:
                qr.status = FederatedQueryStatus.TIMED_OUT
                qr.completed_at = now
                qr.round_time_ms = elapsed * 1000
                timed_out.append(qid)
                logger.warning(f"[STRATEGY] Query {qid} TIMED OUT after {elapsed:.1f}s")

        return timed_out

    def get_round(self, query_id: str) -> Optional[FederatedQueryRound]:
        """Retrieve a query round by ID."""
        return self._active_rounds.get(query_id)

    def get_active_rounds(self) -> List[FederatedQueryRound]:
        """Return all non-completed query rounds."""
        return [
            r for r in self._active_rounds.values()
            if r.status not in (FederatedQueryStatus.COMPLETED, FederatedQueryStatus.FAILED, FederatedQueryStatus.TIMED_OUT)
        ]
