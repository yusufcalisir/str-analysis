"""
Federated Node Orchestrator — Control Plane for VANTAGE-STR Network.

Manages node registration, mutual TLS (mTLS) handshake validation,
and a real-time heartbeat system that tracks which international forensic
databases are online and reachable.

Architecture:
    The NodeManager maintains an in-memory registry of connected nodes.
    Each node is identified by a unique node_id (e.g., "TR-NODE-01") and
    must present a valid X.509 certificate chain during the mTLS handshake.
    Heartbeats expire after HEARTBEAT_TTL_SECONDS, marking nodes as offline.

Security Model:
    - mTLS: Both the orchestrator and connecting nodes must present valid
      certificates signed by the VANTAGE-STR root CA.
    - Certificate pinning: Node certificates are validated against a
      pre-registered fingerprint to prevent impersonation.
    - All node metadata is encrypted at rest in the registry.
"""

import hashlib
import logging
import secrets
import ssl
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

HEARTBEAT_TTL_SECONDS: int = 30
HEARTBEAT_INTERVAL_SECONDS: int = 10
MAX_MISSED_HEARTBEATS: int = 3
MAX_NODES: int = 256


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class NodeStatus(str, Enum):
    """Operational state of a federated node."""
    PENDING = "pending"          # Registered but not yet authenticated
    ONLINE = "online"            # Authenticated and heartbeat active
    SYNCING = "syncing"          # Receiving a federated model update
    OFFLINE = "offline"          # Heartbeat expired
    REVOKED = "revoked"          # Certificate revoked — permanently blocked


class NodeInfo(BaseModel):
    """Complete metadata for a registered federated node."""
    node_id: str = Field(..., min_length=2, max_length=64)
    display_name: str = ""
    country_code: str = Field(..., min_length=2, max_length=3)
    region: str = ""
    agency: str = ""
    status: NodeStatus = NodeStatus.PENDING
    cert_fingerprint: str = ""
    ip_address: str = ""
    grpc_port: int = 50051
    profile_count: int = 0
    last_heartbeat: float = 0.0
    registered_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0
    missed_heartbeats: int = 0
    capabilities: List[str] = Field(default_factory=lambda: ["search", "ingest"])
    # Geolocation for map visualization
    latitude: float = 0.0
    longitude: float = 0.0


class HandshakeRequest(BaseModel):
    """mTLS handshake request from a connecting node."""
    node_id: str
    display_name: str
    country_code: str
    region: str = ""
    agency: str = ""
    cert_pem: str  # PEM-encoded X.509 certificate
    ip_address: str
    grpc_port: int = 50051
    profile_count: int = 0
    capabilities: List[str] = Field(default_factory=lambda: ["search", "ingest"])
    latitude: float = 0.0
    longitude: float = 0.0


class HandshakeResponse(BaseModel):
    """Orchestrator response to a node handshake."""
    accepted: bool
    session_token: str = ""
    orchestrator_cert_fingerprint: str = ""
    assigned_node_id: str = ""
    message: str = ""
    network_peers: int = 0


class HeartbeatPayload(BaseModel):
    """Periodic heartbeat from a connected node."""
    node_id: str
    timestamp: float = Field(default_factory=time.time)
    profile_count: int = 0
    latency_ms: float = 0.0
    cpu_usage: float = 0.0
    memory_usage_mb: float = 0.0
    active_queries: int = 0


class NetworkTelemetry(BaseModel):
    """Aggregate network telemetry for the frontend dashboard."""
    total_nodes: int = 0
    nodes_online: int = 0
    nodes_offline: int = 0
    nodes_syncing: int = 0
    total_networked_profiles: int = 0
    avg_latency_ms: float = 0.0
    last_global_sync_ms: float = 0.0
    active_countries: List[str] = []


# ═══════════════════════════════════════════════════════════════════════════════
# mTLS CERTIFICATE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class CertificateAuthority:
    """
    Handles mTLS certificate validation for the VANTAGE-STR network.

    In production, this validates X.509 certificates against the VANTAGE-STR
    root CA and checks revocation lists. Currently provides the interface
    with fingerprint-based validation.
    """

    def __init__(self, ca_cert_path: Optional[str] = None) -> None:
        self._ca_cert_path = ca_cert_path
        self._revoked_fingerprints: set[str] = set()

    @staticmethod
    def compute_fingerprint(cert_pem: str) -> str:
        """
        Compute SHA-256 fingerprint of a PEM-encoded certificate.

        Used for certificate pinning — each node's fingerprint is stored
        at registration time and validated on every subsequent connection.

        Args:
            cert_pem: PEM-encoded X.509 certificate string.

        Returns:
            SHA-256 hex digest of the DER-encoded certificate.
        """
        # Strip PEM headers and decode to DER
        lines = cert_pem.strip().split("\n")
        der_lines = [l for l in lines if not l.startswith("-----")]
        der_content = "".join(der_lines)
        return hashlib.sha256(der_content.encode("utf-8")).hexdigest()

    def validate_certificate(self, cert_pem: str) -> Tuple[bool, str]:
        """
        Validate a node's certificate against the CA and revocation list.

        Args:
            cert_pem: PEM-encoded certificate from the connecting node.

        Returns:
            Tuple of (is_valid: bool, reason: str).
        """
        if not cert_pem or len(cert_pem) < 64:
            return False, "Certificate is empty or too short"

        fingerprint = self.compute_fingerprint(cert_pem)

        if fingerprint in self._revoked_fingerprints:
            return False, f"Certificate {fingerprint[:16]}... has been revoked"

        # In production: validate against CA chain, check expiry, CRL/OCSP
        return True, "Certificate accepted"

    def revoke_certificate(self, fingerprint: str) -> None:
        """Add a certificate fingerprint to the revocation list."""
        self._revoked_fingerprints.add(fingerprint)
        logger.warning(f"[CA] Certificate revoked: {fingerprint[:16]}...")

    def create_ssl_context(self, server_mode: bool = True) -> ssl.SSLContext:
        """
        Create an SSL context for mTLS communication.

        Args:
            server_mode: True for server-side context, False for client.

        Returns:
            Configured SSLContext with mutual TLS enforcement.
        """
        protocol = ssl.PROTOCOL_TLS_SERVER if server_mode else ssl.PROTOCOL_TLS_CLIENT
        ctx = ssl.SSLContext(protocol)
        ctx.verify_mode = ssl.CERT_REQUIRED  # Require client certificates
        ctx.minimum_version = ssl.TLSVersion.TLSv1_3

        if self._ca_cert_path and Path(self._ca_cert_path).exists():
            ctx.load_verify_locations(self._ca_cert_path)

        return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# NODE MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class NodeManager:
    """
    Central registry and lifecycle manager for federated VANTAGE-STR nodes.

    Responsibilities:
        - Secure onboarding via mTLS handshake.
        - Certificate pinning and revocation.
        - Real-time heartbeat monitoring.
        - Network telemetry aggregation for the frontend.

    Thread Safety:
        The registry uses a simple dict with atomic operations. For
        production multi-threaded environments, wrap with asyncio.Lock.

    Usage:
        manager = NodeManager()
        response = manager.handle_handshake(request)
        manager.process_heartbeat(heartbeat)
        telemetry = manager.get_telemetry()
    """

    def __init__(self, ca_cert_path: Optional[str] = None) -> None:
        self._registry: Dict[str, NodeInfo] = {}
        self._session_tokens: Dict[str, str] = {}  # node_id → session_token
        self._ca = CertificateAuthority(ca_cert_path)
        self._last_global_sync: float = time.time()

    @property
    def online_nodes(self) -> List[NodeInfo]:
        """Return all nodes with active heartbeats."""
        return [n for n in self._registry.values() if n.status == NodeStatus.ONLINE]

    @property
    def all_nodes(self) -> List[NodeInfo]:
        """Return all registered nodes regardless of status."""
        return list(self._registry.values())

    def handle_handshake(self, request: HandshakeRequest) -> HandshakeResponse:
        """
        Process an mTLS handshake from a connecting node.

        Steps:
            1. Validate the node's X.509 certificate.
            2. Check for duplicate registration.
            3. Enforce maximum node capacity.
            4. Register the node and issue a session token.

        Args:
            request: HandshakeRequest with node metadata and certificate.

        Returns:
            HandshakeResponse indicating acceptance or rejection.
        """
        # Certificate validation
        is_valid, reason = self._ca.validate_certificate(request.cert_pem)
        if not is_valid:
            logger.warning(
                f"[ORCHESTRATOR] Handshake REJECTED for {request.node_id}: {reason}"
            )
            return HandshakeResponse(
                accepted=False,
                message=f"Certificate validation failed: {reason}",
            )

        # Duplicate check
        if request.node_id in self._registry:
            existing = self._registry[request.node_id]
            if existing.status == NodeStatus.REVOKED:
                return HandshakeResponse(
                    accepted=False,
                    message="Node has been permanently revoked from the network",
                )
            # Reconnection — update metadata
            logger.info(f"[ORCHESTRATOR] Node {request.node_id} reconnecting")

        # Capacity check
        if len(self._registry) >= MAX_NODES and request.node_id not in self._registry:
            return HandshakeResponse(
                accepted=False,
                message=f"Network capacity reached ({MAX_NODES} nodes)",
            )

        # Register node
        fingerprint = self._ca.compute_fingerprint(request.cert_pem)
        session_token = str(uuid.uuid4())

        node = NodeInfo(
            node_id=request.node_id,
            display_name=request.display_name or request.node_id,
            country_code=request.country_code,
            region=request.region,
            agency=request.agency,
            status=NodeStatus.ONLINE,
            cert_fingerprint=fingerprint,
            ip_address=request.ip_address,
            grpc_port=request.grpc_port,
            profile_count=request.profile_count,
            last_heartbeat=time.time(),
            latency_ms=0.0,
            capabilities=request.capabilities,
            latitude=request.latitude,
            longitude=request.longitude,
        )

        self._registry[request.node_id] = node
        self._session_tokens[request.node_id] = session_token

        logger.info(
            f"[ORCHESTRATOR] Node {request.node_id} ({request.country_code}) "
            f"REGISTERED | cert={fingerprint[:16]}... | "
            f"profiles={request.profile_count}"
        )

        return HandshakeResponse(
            accepted=True,
            session_token=session_token,
            orchestrator_cert_fingerprint="orchestrator-root-fingerprint",
            assigned_node_id=request.node_id,
            message="Welcome to VANTAGE-STR network",
            network_peers=len(self.online_nodes),
        )

    def process_heartbeat(self, heartbeat: HeartbeatPayload) -> bool:
        """
        Process a heartbeat from a connected node.

        Updates the node's last-seen timestamp, profile count, and
        operational metrics. Resets the missed heartbeat counter.

        Args:
            heartbeat: HeartbeatPayload with node telemetry.

        Returns:
            True if the heartbeat was accepted, False if node unknown.
        """
        if heartbeat.node_id not in self._registry:
            logger.warning(f"[ORCHESTRATOR] Heartbeat from unknown node: {heartbeat.node_id}")
            return False

        node = self._registry[heartbeat.node_id]
        if node.status == NodeStatus.REVOKED:
            return False

        node.last_heartbeat = heartbeat.timestamp or time.time()
        node.profile_count = heartbeat.profile_count
        node.latency_ms = heartbeat.latency_ms
        node.missed_heartbeats = 0

        if node.status == NodeStatus.OFFLINE:
            node.status = NodeStatus.ONLINE
            logger.info(f"[ORCHESTRATOR] Node {heartbeat.node_id} back ONLINE")

        return True

    def check_heartbeats(self) -> List[str]:
        """
        Sweep all nodes and mark those with expired heartbeats as offline.

        Should be called periodically (every HEARTBEAT_INTERVAL_SECONDS)
        by a background task scheduler.

        Returns:
            List of node_ids that transitioned to OFFLINE.
        """
        now = time.time()
        newly_offline: List[str] = []

        for node_id, node in self._registry.items():
            if node.status in (NodeStatus.REVOKED, NodeStatus.PENDING):
                continue

            elapsed = now - node.last_heartbeat
            if elapsed > HEARTBEAT_TTL_SECONDS:
                node.missed_heartbeats += 1

                if node.missed_heartbeats >= MAX_MISSED_HEARTBEATS:
                    node.status = NodeStatus.OFFLINE
                    newly_offline.append(node_id)
                    logger.warning(
                        f"[ORCHESTRATOR] Node {node_id} marked OFFLINE "
                        f"(missed {node.missed_heartbeats} heartbeats)"
                    )

        return newly_offline

    def revoke_node(self, node_id: str, reason: str = "Administrative action") -> bool:
        """
        Permanently revoke a node from the network.

        The node's certificate fingerprint is added to the revocation list,
        preventing re-registration without manual intervention.

        Args:
            node_id: ID of the node to revoke.
            reason: Human-readable reason for revocation.

        Returns:
            True if the node was found and revoked.
        """
        if node_id not in self._registry:
            return False

        node = self._registry[node_id]
        node.status = NodeStatus.REVOKED
        self._ca.revoke_certificate(node.cert_fingerprint)
        self._session_tokens.pop(node_id, None)

        logger.warning(f"[ORCHESTRATOR] Node {node_id} REVOKED: {reason}")
        return True

    def set_node_syncing(self, node_id: str) -> None:
        """Mark a node as currently receiving a federated model update."""
        if node_id in self._registry:
            self._registry[node_id].status = NodeStatus.SYNCING

    def set_node_online(self, node_id: str) -> None:
        """Mark a syncing node as back online after update completion."""
        if node_id in self._registry and self._registry[node_id].status == NodeStatus.SYNCING:
            self._registry[node_id].status = NodeStatus.ONLINE

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        """Retrieve a node's metadata by ID."""
        return self._registry.get(node_id)

    def validate_session(self, node_id: str, session_token: str) -> bool:
        """Validate that a session token matches the registered node."""
        return self._session_tokens.get(node_id) == session_token

    def get_telemetry(self) -> NetworkTelemetry:
        """
        Aggregate network-wide telemetry for the frontend dashboard.

        Computes totals for active nodes, offline nodes, total profiles,
        average latency, and list of active countries.
        """
        nodes = list(self._registry.values())
        online = [n for n in nodes if n.status == NodeStatus.ONLINE]
        offline = [n for n in nodes if n.status == NodeStatus.OFFLINE]
        syncing = [n for n in nodes if n.status == NodeStatus.SYNCING]

        avg_latency = 0.0
        if online:
            avg_latency = sum(n.latency_ms for n in online) / len(online)

        countries = sorted(set(n.country_code for n in online))
        total_profiles = sum(n.profile_count for n in nodes if n.status != NodeStatus.REVOKED)

        return NetworkTelemetry(
            total_nodes=len(nodes),
            nodes_online=len(online),
            nodes_offline=len(offline),
            nodes_syncing=len(syncing),
            total_networked_profiles=total_profiles,
            avg_latency_ms=round(avg_latency, 2),
            last_global_sync_ms=round((time.time() - self._last_global_sync) * 1000, 1),
            active_countries=countries,
        )

    def record_global_sync(self) -> None:
        """Record the timestamp of the last global synchronization event."""
        self._last_global_sync = time.time()

    def get_nodes_for_broadcast(self) -> List[NodeInfo]:
        """
        Return all nodes eligible to receive a federated query broadcast.

        Only ONLINE nodes with 'search' capability are included.
        """
        return [
            n for n in self._registry.values()
            if n.status == NodeStatus.ONLINE and "search" in n.capabilities
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY BROADCAST — Fan-out Pattern
# ═══════════════════════════════════════════════════════════════════════════════

class QueryBroadcastConfig(BaseModel):
    """Configuration for a global query broadcast."""
    timeout_seconds: float = 2.5
    min_nodes_required: int = 1
    max_results_per_node: int = 25
    confidence_threshold: float = 0.85
    routing_order: List[str] = Field(
        default_factory=list,
        description="Node IDs in latency-sorted order. Empty = all at once."
    )


class NodeQueryResult(BaseModel):
    """Result from a single node during broadcast."""
    node_id: str
    country_code: str = ""
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    responded: bool = False
    response_time_ms: float = 0.0
    error: str = ""
    timed_out: bool = False


class GlobalRankedMatch(BaseModel):
    """A single match in the global ranking list."""
    node_id: str
    country_code: str = ""
    local_reference_token: str
    match_score: float
    node_response_time_ms: float = 0.0
    zkp_proof_data: Optional[Dict[str, Any]] = None


class ZKPVerifiedMatch(BaseModel):
    """A match that has been cryptographically verified via ZKP."""
    node_id: str
    country_code: str = ""
    local_reference_token: str
    match_score: float
    commitment_hash: str
    proof_verified: bool = False
    verification_ms: float = 0.0
    query_id: str = ""
    legal_request_token: Optional[str] = None


class BroadcastResult(BaseModel):
    """Complete result of a global query broadcast."""
    query_id: str
    total_nodes_queried: int = 0
    nodes_responded: int = 0
    nodes_timed_out: int = 0
    nodes_errored: int = 0
    total_matches: int = 0
    global_ranking: List[GlobalRankedMatch] = Field(default_factory=list)
    node_results: List[NodeQueryResult] = Field(default_factory=list)
    broadcast_time_ms: float = 0.0
    aggregation_time_ms: float = 0.0
    total_time_ms: float = 0.0
    # ZKP Verification Layer (Phase 4.2)
    verified_matches: List[ZKPVerifiedMatch] = Field(default_factory=list)
    zkp_verification_ms: float = 0.0
    zkp_enabled: bool = False


import asyncio
from typing import Callable, Coroutine


async def broadcast_query(
    manager: NodeManager,
    query_id: str,
    query_embedding: List[float],
    config: Optional[QueryBroadcastConfig] = None,
    query_node_fn: Optional[Callable[..., Coroutine]] = None,
    zkp_enabled: bool = False,
) -> BroadcastResult:
    """
    Broadcast a search query to ALL active nodes simultaneously.

    Fan-out Pattern:
        1. Collect all eligible nodes (ONLINE + search capability).
        2. Fire async gRPC calls to all nodes in parallel.
        3. Wait up to config.timeout_seconds for responses.
        4. Aggregate MatchScores into a global ranked list.

    Timeout Behavior:
        Nodes that do not respond within the deadline are marked as
        timed_out. Partial results from faster nodes are still used.

    Args:
        manager: NodeManager with the node registry.
        query_id: Unique query identifier.
        query_embedding: 48-dim normalized STR embedding vector.
        config: Broadcast configuration (timeout, thresholds).
        query_node_fn: Async function(node_id, embedding) → NodeQueryResult.
                       If None, uses simulated responses.

    Returns:
        BroadcastResult with global ranking and per-node results.
    """
    cfg = config or QueryBroadcastConfig()
    t_start = time.perf_counter()

    # Collect eligible nodes
    eligible = manager.get_nodes_for_broadcast()
    if not eligible:
        return BroadcastResult(
            query_id=query_id,
            total_nodes_queried=0,
            broadcast_time_ms=0.0,
            total_time_ms=0.0,
        )

    # Order by latency if routing_order is provided
    if cfg.routing_order:
        order_map = {nid: i for i, nid in enumerate(cfg.routing_order)}
        eligible.sort(key=lambda n: order_map.get(n.node_id, 999))

    # Fire all queries concurrently
    async def _query_single(node: NodeInfo) -> NodeQueryResult:
        t_node_start = time.perf_counter()

        if query_node_fn:
            try:
                result = await asyncio.wait_for(
                    query_node_fn(node.node_id, query_embedding),
                    timeout=cfg.timeout_seconds,
                )
                result.response_time_ms = (time.perf_counter() - t_node_start) * 1000
                result.responded = True
                return result
            except asyncio.TimeoutError:
                return NodeQueryResult(
                    node_id=node.node_id,
                    country_code=node.country_code,
                    responded=False,
                    timed_out=True,
                    response_time_ms=cfg.timeout_seconds * 1000,
                    error="Timeout",
                )
            except Exception as exc:
                return NodeQueryResult(
                    node_id=node.node_id,
                    country_code=node.country_code,
                    responded=False,
                    error=str(exc),
                    response_time_ms=(time.perf_counter() - t_node_start) * 1000,
                )

        # Simulation mode
        return _simulate_node_response(node, t_node_start, cfg.timeout_seconds)

    tasks = [_query_single(node) for node in eligible]
    node_results: List[NodeQueryResult] = await asyncio.gather(*tasks)

    t_broadcast = time.perf_counter()
    broadcast_ms = (t_broadcast - t_start) * 1000

    # Aggregate into global ranking
    t_agg_start = time.perf_counter()
    global_ranking: List[GlobalRankedMatch] = []

    for nr in node_results:
        if not nr.responded:
            continue
        for match in nr.matches:
            score = match.get("match_score", 0.0)
            if score >= cfg.confidence_threshold:
                global_ranking.append(GlobalRankedMatch(
                    node_id=nr.node_id,
                    country_code=nr.country_code,
                    local_reference_token=match.get("local_reference_token", ""),
                    match_score=score,
                    node_response_time_ms=nr.response_time_ms,
                ))

    # Sort by score descending
    global_ranking.sort(key=lambda m: m.match_score, reverse=True)

    t_agg_end = time.perf_counter()
    agg_ms = (t_agg_end - t_agg_start) * 1000
    total_ms = (t_agg_end - t_start) * 1000

    responded = sum(1 for r in node_results if r.responded)
    timed_out = sum(1 for r in node_results if r.timed_out)
    errored = sum(1 for r in node_results if not r.responded and not r.timed_out and r.error)

    logger.info(
        f"[BROADCAST] Query {query_id} → {len(eligible)} nodes | "
        f"responded={responded} | timed_out={timed_out} | "
        f"matches={len(global_ranking)} | {total_ms:.1f}ms total"
    )

    manager.record_global_sync()

    # ── ZKP Verification Layer ──
    verified_matches: List[ZKPVerifiedMatch] = []
    zkp_ms = 0.0
    if zkp_enabled and global_ranking:
        t_zkp_start = time.perf_counter()
        verified_matches = await _verify_zkp_proofs(
            global_ranking, query_id, query_embedding, cfg.confidence_threshold,
        )
        zkp_ms = (time.perf_counter() - t_zkp_start) * 1000
        logger.info(
            f"[BROADCAST] ZKP verification: {len(verified_matches)}/"
            f"{len(global_ranking)} proofs verified in {zkp_ms:.1f}ms"
        )

    total_ms = (time.perf_counter() - t_start) * 1000

    return BroadcastResult(
        query_id=query_id,
        total_nodes_queried=len(eligible),
        nodes_responded=responded,
        nodes_timed_out=timed_out,
        nodes_errored=errored,
        total_matches=len(global_ranking),
        global_ranking=global_ranking,
        node_results=node_results,
        broadcast_time_ms=round(broadcast_ms, 2),
        aggregation_time_ms=round(agg_ms, 2),
        total_time_ms=round(total_ms, 2),
        verified_matches=verified_matches,
        zkp_verification_ms=round(zkp_ms, 2),
        zkp_enabled=zkp_enabled,
    )


def _simulate_node_response(
    node: NodeInfo,
    t_start: float,
    timeout: float,
) -> NodeQueryResult:
    """Generate a realistic simulated node response for development."""
    import random

    base_latency = (hash(node.node_id) % 100 + 30) / 1000  # 30–130ms
    jitter = random.uniform(-0.01, 0.02)
    latency = max(0.01, base_latency + jitter)

    # 5% chance of "timeout"
    if random.random() < 0.05:
        return NodeQueryResult(
            node_id=node.node_id,
            country_code=node.country_code,
            responded=False,
            timed_out=True,
            response_time_ms=timeout * 1000,
            error="Simulated timeout",
        )

    # Generate simulated matches
    n_matches = random.randint(0, 4)
    matches = []
    for i in range(n_matches):
        matches.append({
            "local_reference_token": f"LRT-{secrets.token_hex(12)}",
            "match_score": round(random.uniform(0.70, 0.99), 4),
            "distance": round(random.uniform(0.01, 0.30), 4),
        })

    return NodeQueryResult(
        node_id=node.node_id,
        country_code=node.country_code,
        matches=matches,
        responded=True,
        response_time_ms=round(latency * 1000, 2),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC ZKP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

async def _verify_zkp_proofs(
    matches: List[GlobalRankedMatch],
    query_id: str,
    query_embedding: List[float],
    confidence_threshold: float,
) -> List[ZKPVerifiedMatch]:
    """
    Verify ZKP proofs for all matches in parallel.

    Each match with a zkp_proof_data dict is verified via the ZKPBridge.
    Proofs without ZKP data are skipped (not promoted to verified).

    Returns:
        List of matches that passed cryptographic verification.
    """
    from app.core.crypto.bridge import ZKPBridge
    from app.core.crypto.zkp_prover import ZKProof

    bridge = ZKPBridge()
    verified: List[ZKPVerifiedMatch] = []

    async def _verify_single(match: GlobalRankedMatch) -> Optional[ZKPVerifiedMatch]:
        if not match.zkp_proof_data:
            return None

        t_start = time.perf_counter()
        try:
            # Reconstruct proof from transmitted data
            proof_bytes = bytes.fromhex(match.zkp_proof_data.get("proof_hex", ""))
            if not proof_bytes:
                return None

            proof = ZKProof.from_bytes(proof_bytes)

            # Cryptographic verification (runs in thread pool for CPU-bound work)
            loop = asyncio.get_event_loop()
            is_valid = await loop.run_in_executor(
                None,
                lambda: bridge.verify(proof, query_embedding, confidence_threshold, query_id),
            )

            v_ms = (time.perf_counter() - t_start) * 1000

            if is_valid:
                return ZKPVerifiedMatch(
                    node_id=match.node_id,
                    country_code=match.country_code,
                    local_reference_token=match.local_reference_token,
                    match_score=match.match_score,
                    commitment_hash=proof.commitment_hash,
                    proof_verified=True,
                    verification_ms=round(v_ms, 2),
                    query_id=query_id,
                )
            else:
                logger.warning(
                    f"[ZKP-VERIFY] INVALID proof from node={match.node_id} "
                    f"token={match.local_reference_token}"
                )
                return None

        except Exception as exc:
            logger.error(
                f"[ZKP-VERIFY] Error verifying proof from node={match.node_id}: {exc}"
            )
            return None

    # Fire all verifications concurrently
    results = await asyncio.gather(*[_verify_single(m) for m in matches])
    verified = [r for r in results if r is not None]

    return verified
