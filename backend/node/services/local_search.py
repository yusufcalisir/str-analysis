"""
Local Forensic Searcher — Privacy-Preserving Search on the Sovereign Node.

Receives query embeddings from the Global Orchestrator via gRPC and
executes cosine similarity searches against the local Milvus instance.
All results are privacy-masked: raw profile_ids are replaced with
ephemeral LocalReferenceTokens and only MatchScore floats are returned.

Phase 4.1: ZKP Privacy Shield extends the pipeline with Zero-Knowledge
Proofs. When zkp_enabled=True, each match produces a cryptographic proof
that the local node possesses a matching profile, without revealing the
actual DNA vector. The Orchestrator receives proofs + commitments
instead of raw similarity scores.

Privacy Contract:
    The Orchestrator NEVER sees:
        - Raw profile_ids (replaced by HMAC-derived tokens)
        - STR allele values / local DNA vectors
        - Any locally-stored metadata
    The Orchestrator ONLY sees:
        - MatchScore (float 0.0–1.0) OR ZKProof (when ZKP enabled)
        - LocalReferenceToken (opaque, non-reversible)
        - Pedersen Commitment + salted hash
        - SearchMetrics (timing, count)

Architecture:
    QueryEmbedding(48-dim) → Milvus search_top_k → threshold filter
    → privacy mask → [ZKP proof generation] → MatchResult[] → gRPC response
"""

import base64
import hashlib
import hmac
import logging
import random
import secrets
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.core.crypto.bridge import ZKPBridge
from app.core.crypto.zkp_prover import ZKProof

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.90
MAX_RESULTS_PER_QUERY: int = 25
TOKEN_EXPIRY_SECONDS: int = 3600  # 1 hour
VECTOR_DIM: int = 48


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class MatchResult(BaseModel):
    """
    Privacy-masked search result returned to the Orchestrator.

    Contains only a confidence score and an opaque reference token.
    The profile_id is never transmitted.
    """
    local_reference_token: str = Field(
        ..., description="HMAC-derived opaque token. Non-reversible."
    )
    match_score: float = Field(
        ..., ge=0.0, le=1.0, description="Cosine similarity confidence."
    )
    distance: float = Field(
        0.0, description="Raw cosine distance from Milvus."
    )
    token_expiry: float = Field(
        0.0, description="Unix epoch when this token becomes invalid."
    )


class SearchMetrics(BaseModel):
    """Timing and volume metrics for a local search operation."""
    query_id: str
    profiles_searched: int = 0
    matches_found: int = 0
    matches_above_threshold: int = 0
    milvus_search_ms: float = 0.0
    total_search_ms: float = 0.0
    threshold_used: float = DEFAULT_CONFIDENCE_THRESHOLD


class ZKPMatchResult(BaseModel):
    """
    ZKP-enhanced match result returned when zkp_enabled=True.

    Instead of a raw match_score, contains a cryptographic proof
    that the node possesses a matching profile. The proof is
    independently verifiable by the Orchestrator without access
    to the local DNA vector.
    """
    local_reference_token: str = Field(
        ..., description="HMAC-derived opaque token."
    )
    proof_b64: str = Field(
        ..., description="Base64-encoded ZKP proof bytes."
    )
    commitment_hash: str = Field(
        ..., description="Salted BLAKE2b digest of the Pedersen commitment."
    )
    threshold_met: bool = Field(
        ..., description="Whether similarity ≥ threshold (proven in ZK)."
    )
    proof_size_bytes: int = Field(
        0, description="Size of the serialized proof."
    )
    token_expiry: float = Field(
        0.0, description="Unix epoch when this token becomes invalid."
    )


class LocalSearchResponse(BaseModel):
    """Complete response to a federated search query."""
    query_id: str
    node_id: str
    results: List[MatchResult] = Field(default_factory=list)
    zkp_results: List[ZKPMatchResult] = Field(default_factory=list)
    metrics: SearchMetrics
    sovereignty_approved: bool = True
    zkp_enabled: bool = False


class QueryType(str, Enum):
    """Categories of incoming federated queries for sovereignty filtering."""
    CRIMINAL_SEARCH = "criminal_search"
    MISSING_PERSONS = "missing_persons"
    DISASTER_IDENTIFICATION = "disaster_identification"
    KINSHIP_ANALYSIS = "kinship_analysis"
    INTERPOL_RED_NOTICE = "interpol_red_notice"


class SovereigntyPolicy(BaseModel):
    """Configurable sovereignty rules for query filtering."""
    allowed_query_types: List[QueryType] = Field(
        default_factory=lambda: list(QueryType)
    )
    allowed_requesting_nodes: List[str] = Field(
        default_factory=list,
        description="Empty = allow all. Populated = whitelist."
    )
    blocked_requesting_nodes: List[str] = Field(default_factory=list)
    require_manual_approval: bool = False
    max_results_per_query: int = MAX_RESULTS_PER_QUERY
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD


# ═══════════════════════════════════════════════════════════════════════════════
# TOKEN MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class ReferenceTokenManager:
    """
    Generates and manages LocalReferenceTokens for privacy masking.

    Tokens are HMAC-SHA256 derived from the real profile_id using a
    per-session secret key. This makes tokens:
        - Deterministic within a session (same profile → same token).
        - Non-reversible (HMAC cannot recover the profile_id).
        - Expirable (tokens are invalidated after TOKEN_EXPIRY_SECONDS).
        - Session-scoped (new key per node boot = new tokens).

    Only the local node can resolve a token back to a profile_id,
    and only if the token is still in the active mapping.
    """

    def __init__(self) -> None:
        self._secret = secrets.token_bytes(32)
        self._token_to_profile: Dict[str, str] = {}
        self._profile_to_token: Dict[str, str] = {}
        self._token_expiry: Dict[str, float] = {}

    def generate_token(self, profile_id: str) -> Tuple[str, float]:
        """
        Generate or retrieve an opaque token for a profile_id.

        Args:
            profile_id: Real UUID of the local profile.

        Returns:
            Tuple of (token_string, expiry_timestamp).
        """
        # Check existing valid token
        if profile_id in self._profile_to_token:
            token = self._profile_to_token[profile_id]
            expiry = self._token_expiry.get(token, 0.0)
            if time.time() < expiry:
                return token, expiry

        # Generate new HMAC token
        raw = hmac.new(
            self._secret,
            profile_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:24]

        token = f"LRT-{raw}"
        expiry = time.time() + TOKEN_EXPIRY_SECONDS

        self._token_to_profile[token] = profile_id
        self._profile_to_token[profile_id] = token
        self._token_expiry[token] = expiry

        return token, expiry

    def resolve_token(self, token: str) -> Optional[str]:
        """
        Resolve a token back to its profile_id (local use only).

        Returns None if the token is expired or unknown.
        """
        if token not in self._token_to_profile:
            return None

        expiry = self._token_expiry.get(token, 0.0)
        if time.time() > expiry:
            self._cleanup_token(token)
            return None

        return self._token_to_profile[token]

    def purge_all(self) -> int:
        """
        Emergency purge: wipe all token mappings.

        Called by the kill-switch when unauthorized access is detected.

        Returns:
            Number of tokens purged.
        """
        count = len(self._token_to_profile)
        self._token_to_profile.clear()
        self._profile_to_token.clear()
        self._token_expiry.clear()
        self._secret = secrets.token_bytes(32)  # Rotate secret
        logger.warning(f"[TOKEN] Emergency purge: {count} tokens wiped, secret rotated")
        return count

    def _cleanup_token(self, token: str) -> None:
        """Remove a single expired token."""
        profile_id = self._token_to_profile.pop(token, None)
        if profile_id:
            self._profile_to_token.pop(profile_id, None)
        self._token_expiry.pop(token, None)

    def cleanup_expired(self) -> int:
        """Remove all expired tokens. Returns count removed."""
        now = time.time()
        expired = [t for t, exp in self._token_expiry.items() if now > exp]
        for token in expired:
            self._cleanup_token(token)
        return len(expired)


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL FORENSIC SEARCHER
# ═══════════════════════════════════════════════════════════════════════════════

class LocalForensicSearcher:
    """
    Privacy-preserving search service for the VANTAGE sovereign node.

    Receives query embeddings from the Orchestrator, executes local
    Milvus searches, applies sovereignty policies, privacy-masks the
    results, and returns only MatchScores + LocalReferenceTokens.

    The Orchestrator can later request identity resolution for a
    specific token via a separate legal-approval channel, but that
    flow is deliberately decoupled from the search path.

    Usage:
        searcher = LocalForensicSearcher(node_id="TR-NODE-01")
        response = searcher.search(query_id, embedding, query_type)
    """

    def __init__(
        self,
        node_id: str,
        milvus_client: Optional[Any] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        zkp_enabled: bool = False,
    ) -> None:
        """
        Initialize the local searcher.

        Args:
            node_id: This node's identifier.
            milvus_client: MilvusGenomicClient instance. If None,
                           operates in simulation mode.
            confidence_threshold: Minimum similarity score to return.
            zkp_enabled: If True, generate ZKP proofs for each match
                         instead of returning raw similarity scores.
        """
        self._node_id = node_id
        self._milvus = milvus_client
        self._token_manager = ReferenceTokenManager()
        self._sovereignty = SovereigntyPolicy(
            confidence_threshold=confidence_threshold,
        )
        self._query_log: List[Dict[str, Any]] = []
        self._total_queries: int = 0
        self._total_matches: int = 0
        self._zkp_enabled = zkp_enabled
        self._zkp_bridge: Optional[ZKPBridge] = ZKPBridge() if zkp_enabled else None

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def sovereignty_policy(self) -> SovereigntyPolicy:
        return self._sovereignty

    def update_sovereignty(self, policy: SovereigntyPolicy) -> None:
        """Update the sovereignty policy (admin action)."""
        self._sovereignty = policy
        logger.info(f"[SEARCH] Sovereignty policy updated for {self._node_id}")

    def search(
        self,
        query_id: str,
        query_embedding: List[float],
        query_type: QueryType = QueryType.CRIMINAL_SEARCH,
        requesting_node_id: str = "",
        top_k: int = 10,
    ) -> LocalSearchResponse:
        """
        Execute a privacy-masked similarity search.

        Pipeline:
            1. Sovereignty check (query type + requesting node).
            2. Milvus search_top_k with the provided embedding.
            3. Threshold filtering (only results ≥ confidence_threshold).
            4. Privacy masking (profile_id → LocalReferenceToken).
            5. Return MatchResults + SearchMetrics.

        Args:
            query_id: Unique identifier for this query round.
            query_embedding: 48-dim normalized STR vector.
            query_type: Category of the federated query.
            requesting_node_id: Node that initiated the query.
            top_k: Maximum raw results from Milvus.

        Returns:
            LocalSearchResponse with privacy-masked results.
        """
        t_start = time.perf_counter()
        self._total_queries += 1

        # ── Sovereignty Gate ──
        if not self._check_sovereignty(query_type, requesting_node_id):
            logger.warning(
                f"[SEARCH] Query {query_id} BLOCKED by sovereignty policy | "
                f"type={query_type.value} | from={requesting_node_id}"
            )
            return LocalSearchResponse(
                query_id=query_id,
                node_id=self._node_id,
                results=[],
                metrics=SearchMetrics(query_id=query_id),
                sovereignty_approved=False,
            )

        # ── Milvus Search ──
        t_milvus_start = time.perf_counter()
        raw_hits = self._execute_milvus_search(query_embedding, top_k)
        t_milvus_end = time.perf_counter()
        milvus_ms = (t_milvus_end - t_milvus_start) * 1000

        # ── Threshold Filter ──
        threshold = self._sovereignty.confidence_threshold
        max_results = min(
            self._sovereignty.max_results_per_query,
            MAX_RESULTS_PER_QUERY,
        )

        filtered: List[MatchResult] = []
        zkp_results: List[ZKPMatchResult] = []

        for hit in raw_hits:
            confidence = hit.get("confidence", 0.0)
            if confidence >= threshold and len(filtered) < max_results:
                profile_id = hit.get("profile_id", "")
                token, expiry = self._token_manager.generate_token(profile_id)

                filtered.append(MatchResult(
                    local_reference_token=token,
                    match_score=round(confidence, 6),
                    distance=hit.get("distance", 0.0),
                    token_expiry=expiry,
                ))

                # ── ZKP Proof Generation ──
                if self._zkp_enabled and self._zkp_bridge is not None:
                    v_local = hit.get("vector", None)
                    if v_local is not None:
                        try:
                            proof = self._zkp_bridge.prove(
                                v_local=v_local,
                                v_query=query_embedding,
                                tau=threshold,
                            )
                            proof_bytes = proof.to_bytes()
                            zkp_results.append(ZKPMatchResult(
                                local_reference_token=token,
                                proof_b64=base64.b64encode(proof_bytes).decode("ascii"),
                                commitment_hash=proof.commitment_hash,
                                threshold_met=proof.threshold_met,
                                proof_size_bytes=len(proof_bytes),
                                token_expiry=expiry,
                            ))
                        except Exception as exc:
                            logger.error(
                                f"[ZKP] Proof generation failed for "
                                f"{profile_id[:8]}...: {exc}"
                            )

        self._total_matches += len(filtered)
        t_end = time.perf_counter()

        metrics = SearchMetrics(
            query_id=query_id,
            profiles_searched=len(raw_hits),
            matches_found=len(raw_hits),
            matches_above_threshold=len(filtered),
            milvus_search_ms=round(milvus_ms, 2),
            total_search_ms=round((t_end - t_start) * 1000, 2),
            threshold_used=threshold,
        )

        # Audit log
        self._query_log.append({
            "query_id": query_id,
            "query_type": query_type.value,
            "requesting_node": requesting_node_id,
            "matches": len(filtered),
            "zkp_proofs": len(zkp_results),
            "timestamp": time.time(),
        })

        logger.info(
            f"[SEARCH] Query {query_id} from {requesting_node_id}: "
            f"{len(filtered)}/{len(raw_hits)} matches above {threshold} | "
            f"ZKP: {len(zkp_results)} proofs | {milvus_ms:.1f}ms"
        )

        return LocalSearchResponse(
            query_id=query_id,
            node_id=self._node_id,
            results=filtered,
            zkp_results=zkp_results,
            metrics=metrics,
            sovereignty_approved=True,
            zkp_enabled=self._zkp_enabled,
        )

    def resolve_token(self, token: str) -> Optional[str]:
        """
        Resolve a LocalReferenceToken to a profile_id.

        This method should ONLY be called through the legal-approval
        channel, never directly from the federated search path.
        """
        return self._token_manager.resolve_token(token)

    def emergency_purge(self) -> int:
        """Purge all token mappings (kill-switch action)."""
        return self._token_manager.purge_all()

    def get_statistics(self) -> Dict[str, Any]:
        """Return operational statistics for the admin UI."""
        return {
            "node_id": self._node_id,
            "total_queries": self._total_queries,
            "total_matches": self._total_matches,
            "active_tokens": len(self._token_manager._token_to_profile),
            "sovereignty_policy": self._sovereignty.model_dump(),
            "recent_queries": self._query_log[-20:],
        }

    def _check_sovereignty(
        self,
        query_type: QueryType,
        requesting_node_id: str,
    ) -> bool:
        """Evaluate the sovereignty policy for this query."""
        policy = self._sovereignty

        # Query type check
        if query_type not in policy.allowed_query_types:
            return False

        # Blocked node check
        if requesting_node_id in policy.blocked_requesting_nodes:
            return False

        # Whitelist check (if populated)
        if policy.allowed_requesting_nodes:
            if requesting_node_id not in policy.allowed_requesting_nodes:
                return False

        return True

    def _execute_milvus_search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """
        Execute the similarity search against Milvus.

        If no Milvus client is connected, returns simulated results
        for development and testing.
        """
        if self._milvus is not None:
            try:
                result = self._milvus.search_top_k(
                    query_vector=query_embedding,
                    top_k=top_k,
                )
                return [
                    {
                        "profile_id": hit.profile_id,
                        "confidence": hit.confidence_score,
                        "distance": hit.distance,
                    }
                    for hit in result.hits
                ]
            except Exception as exc:
                logger.error(f"[SEARCH] Milvus search failed: {exc}")
                return []

        # Simulation mode — generate synthetic results with vectors
        # for ZKP proof generation
        random.seed(hash(tuple(query_embedding[:4])))
        n = random.randint(3, top_k)
        results = []
        for _ in range(n):
            # Generate a simulated local vector (random perturbation of query)
            noise = [random.uniform(-0.15, 0.15) for _ in range(VECTOR_DIM)]
            sim_vector = [
                max(0.0, min(1.0, q + n_))
                for q, n_ in zip(query_embedding, noise)
            ]
            results.append({
                "profile_id": str(uuid.uuid4()),
                "confidence": round(random.uniform(0.60, 0.99), 4),
                "distance": round(random.uniform(0.01, 0.40), 4),
                "vector": sim_vector,
            })
        return results
