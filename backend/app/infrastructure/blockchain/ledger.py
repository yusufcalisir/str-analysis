"""
ForensicLedger — Tamper-Evident Distributed Audit Chain.

Implements a lightweight append-only blockchain with Merkle tree integrity
verification for the VANTAGE-STR forensic network. Every query attempt,
ZKP verification result, and authorization decision is recorded as an
immutable ledger entry.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  ForensicLedger (Singleton)                          │
    │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
    │  │ AsyncQ   │→ │ Consumer │→ │ Chain + Merkle   │   │
    │  │ (writes) │  │ (bg task)│  │ (append-only)    │   │
    │  └──────────┘  └──────────┘  └──────────────────┘   │
    └──────────────────────────────────────────────────────┘

Security Invariants:
    - DNA data is NEVER stored. Only SHA-256 hashes of query parameters.
    - Each entry's hash includes the previous entry's hash (chain).
    - Merkle root provides O(log n) tamper detection.
    - Async queue ensures zero search-pipeline latency impact.

Usage:
    ledger = ForensicLedger.get_instance()
    await ledger.start()
    await ledger.record_event(
        query_hash="sha256:abc...",
        node_id="INTERPOL-DE",
        zkp_status="verified",
        authorization_token="LRT-...",
    )
    integrity = ledger.verify_integrity()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GENESIS_HASH = "0" * 64  # SHA-256 zero hash for the genesis block
HASH_ALGORITHM = "sha256"
LEDGER_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class ZKPVerificationStatus(str, Enum):
    """Possible ZKP verification outcomes logged to the chain."""
    PENDING = "pending"
    VERIFIED = "verified"
    INVALID = "invalid"
    SKIPPED = "skipped"


class ComplianceDecision(str, Enum):
    """Jurisdictional compliance gate outcomes."""
    AUTHORIZED = "authorized"
    REVERTED = "reverted"
    BYPASSED = "bypassed"


@dataclass(frozen=True)
class LedgerEntry:
    """
    A single immutable record in the forensic audit chain.

    Fields are hashed together to form the entry's digest. The chain
    guarantee comes from including the previous entry's hash.

    CRITICAL: No DNA data. Only cryptographic hashes of actions.
    """
    index: int
    timestamp: str  # ISO-8601 UTC
    query_hash: str  # SHA-256 of query parameters (NOT the DNA data)
    node_id: str
    zkp_status: str  # ZKPVerificationStatus value
    authorization_token: str  # Legal Request Token ID or empty
    compliance_decision: str  # ComplianceDecision value
    metadata: Dict[str, Any]  # Extra context (agency, crime category, etc.)
    previous_hash: str
    entry_hash: str  # SHA-256 of this entry (computed at creation)
    merkle_root: str = ""  # Updated after Merkle tree rebuild

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-safe dictionary."""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "query_hash": self.query_hash,
            "node_id": self.node_id,
            "zkp_status": self.zkp_status,
            "authorization_token": self.authorization_token,
            "compliance_decision": self.compliance_decision,
            "metadata": self.metadata,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "merkle_root": self.merkle_root,
        }


class IntegrityReport(BaseModel):
    """Result of a full chain integrity verification."""
    is_valid: bool = True
    chain_length: int = 0
    merkle_root: str = ""
    first_invalid_index: int = -1
    error_message: str = ""
    verified_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class LedgerStats(BaseModel):
    """Summary statistics for the auditor view."""
    total_entries: int = 0
    verified_proofs: int = 0
    invalid_proofs: int = 0
    reverted_queries: int = 0
    authorized_queries: int = 0
    chain_age_seconds: float = 0.0
    merkle_root: str = ""
    is_chain_valid: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# MERKLE TREE
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _hash_pair(left: str, right: str) -> str:
    """Hash two nodes together for Merkle tree construction."""
    return _sha256(left + right)


class MerkleTree:
    """
    Incremental binary Merkle tree over ledger entry hashes.

    Provides O(log n) tamper detection: if any leaf is modified,
    the root hash changes. The tree is rebuilt on every append
    (O(n) but n is bounded by chain length, which grows slowly).

    Structure:
        Level 0 (leaves):  [H(e0), H(e1), H(e2), H(e3), ...]
        Level 1:           [H(H0+H1), H(H2+H3), ...]
        Level 2 (root):    [H(L1_0 + L1_1)]
    """

    def __init__(self) -> None:
        self._leaves: List[str] = []
        self._root: str = GENESIS_HASH

    @property
    def root(self) -> str:
        """Current Merkle root hash."""
        return self._root

    @property
    def leaf_count(self) -> int:
        return len(self._leaves)

    def add_leaf(self, entry_hash: str) -> str:
        """
        Add a new leaf and recompute the root.

        Args:
            entry_hash: SHA-256 hash of the ledger entry.

        Returns:
            Updated Merkle root.
        """
        self._leaves.append(entry_hash)
        self._root = self._compute_root(self._leaves)
        return self._root

    def verify(self, entry_hashes: List[str]) -> bool:
        """
        Verify that the given entry hashes produce the current root.

        Args:
            entry_hashes: Ordered list of all entry hashes.

        Returns:
            True if recomputed root matches stored root.
        """
        if not entry_hashes:
            return self._root == GENESIS_HASH
        computed = self._compute_root(entry_hashes)
        return computed == self._root

    @staticmethod
    def _compute_root(leaves: List[str]) -> str:
        """Build the Merkle tree bottom-up and return root."""
        if not leaves:
            return GENESIS_HASH

        # Start with a copy of leaves
        level = list(leaves)

        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                left = level[i]
                # If odd number of nodes, duplicate the last one
                right = level[i + 1] if i + 1 < len(level) else left
                next_level.append(_hash_pair(left, right))
            level = next_level

        return level[0]


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY HASH COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_entry_hash(
    index: int,
    timestamp: str,
    query_hash: str,
    node_id: str,
    zkp_status: str,
    authorization_token: str,
    compliance_decision: str,
    metadata: Dict[str, Any],
    previous_hash: str,
) -> str:
    """
    Compute the SHA-256 digest of a ledger entry.

    The hash binds ALL fields — any modification to any field
    produces a different hash, breaking the chain.
    """
    canonical = json.dumps(
        {
            "index": index,
            "timestamp": timestamp,
            "query_hash": query_hash,
            "node_id": node_id,
            "zkp_status": zkp_status,
            "authorization_token": authorization_token,
            "compliance_decision": compliance_decision,
            "metadata": metadata,
            "previous_hash": previous_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256(canonical)


# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC LEDGER (SINGLETON)
# ═══════════════════════════════════════════════════════════════════════════════

class ForensicLedger:
    """
    Append-only audit chain with asynchronous non-blocking writes.

    The ledger uses an asyncio.Queue to decouple write requests from
    the search pipeline. A background consumer task drains the queue
    and appends entries to the chain + Merkle tree.

    Singleton Pattern:
        ledger = ForensicLedger.get_instance()
        await ledger.start()

    Thread Safety:
        All mutations happen in the single consumer coroutine.
        Read operations (get_chain, verify_integrity) are safe
        to call from any coroutine as they operate on immutable
        list snapshots.
    """

    _instance: Optional[ForensicLedger] = None

    def __init__(self) -> None:
        self._chain: List[LedgerEntry] = []
        self._merkle_tree = MerkleTree()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self._created_at = time.time()

    @classmethod
    def get_instance(cls) -> ForensicLedger:
        """Get or create the singleton ledger instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing only)."""
        if cls._instance and cls._instance._running:
            cls._instance._running = False
        cls._instance = None

    # ── Lifecycle ──

    async def start(self) -> None:
        """Start the background consumer task."""
        if self._running:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())
        logger.info("[LEDGER] Forensic audit chain started — consumer active")

    async def stop(self) -> None:
        """Gracefully stop the consumer (flush pending writes)."""
        self._running = False
        if self._consumer_task:
            # Drain remaining items
            await self._queue.join()
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None
        logger.info(
            f"[LEDGER] Shut down — {len(self._chain)} entries committed"
        )

    # ── Write Interface (Non-Blocking) ──

    async def record_event(
        self,
        query_hash: str,
        node_id: str,
        zkp_status: str = ZKPVerificationStatus.SKIPPED.value,
        authorization_token: str = "",
        compliance_decision: str = ComplianceDecision.AUTHORIZED.value,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Enqueue a new audit event for recording.

        This method returns immediately — the actual chain append
        happens asynchronously in the consumer task. Zero latency
        impact on the search pipeline.

        Args:
            query_hash: SHA-256 of the query parameters.
            node_id: Originating or target node identifier.
            zkp_status: ZKP verification outcome.
            authorization_token: Legal request token ID (if any).
            compliance_decision: Jurisdictional compliance gate result.
            metadata: Additional context (agency, crime category, etc.).
        """
        await self._queue.put({
            "query_hash": query_hash,
            "node_id": node_id,
            "zkp_status": zkp_status,
            "authorization_token": authorization_token,
            "compliance_decision": compliance_decision,
            "metadata": metadata or {},
        })

    # ── Read Interface ──

    @property
    def chain_length(self) -> int:
        """Number of entries in the chain."""
        return len(self._chain)

    @property
    def merkle_root(self) -> str:
        """Current Merkle root of the chain."""
        return self._merkle_tree.root

    def get_chain(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Get a paginated snapshot of the chain.

        Args:
            limit: Maximum entries to return.
            offset: Starting index.

        Returns:
            List of serialized ledger entries (newest first).
        """
        snapshot = list(reversed(self._chain))
        page = snapshot[offset: offset + limit]
        return [e.to_dict() for e in page]

    def get_entry(self, index: int) -> Optional[Dict]:
        """Get a specific entry by index."""
        if 0 <= index < len(self._chain):
            return self._chain[index].to_dict()
        return None

    def get_stats(self) -> LedgerStats:
        """Compute summary statistics for the auditor view."""
        verified = sum(1 for e in self._chain if e.zkp_status == ZKPVerificationStatus.VERIFIED.value)
        invalid = sum(1 for e in self._chain if e.zkp_status == ZKPVerificationStatus.INVALID.value)
        reverted = sum(1 for e in self._chain if e.compliance_decision == ComplianceDecision.REVERTED.value)
        authorized = sum(1 for e in self._chain if e.compliance_decision == ComplianceDecision.AUTHORIZED.value)

        age = time.time() - self._created_at if self._chain else 0.0

        return LedgerStats(
            total_entries=len(self._chain),
            verified_proofs=verified,
            invalid_proofs=invalid,
            reverted_queries=reverted,
            authorized_queries=authorized,
            chain_age_seconds=round(age, 2),
            merkle_root=self._merkle_tree.root,
            is_chain_valid=self.verify_integrity().is_valid,
        )

    def get_filtered(
        self,
        filter_type: str = "all",
        limit: int = 50,
    ) -> List[Dict]:
        """
        Get filtered chain entries for the audit portal.

        Args:
            filter_type: One of "all", "failed", "cross_border", "anomalies".
            limit: Maximum entries to return.

        Returns:
            Filtered list of serialized entries (newest first).
        """
        chain = list(reversed(self._chain))

        if filter_type == "failed":
            chain = [
                e for e in chain
                if e.compliance_decision == ComplianceDecision.REVERTED.value
                or e.zkp_status == ZKPVerificationStatus.INVALID.value
            ]
        elif filter_type == "cross_border":
            chain = [
                e for e in chain
                if e.metadata.get("cross_border", False)
            ]
        elif filter_type == "anomalies":
            chain = [
                e for e in chain
                if e.metadata.get("anomaly_flagged", False)
            ]

        return [e.to_dict() for e in chain[:limit]]

    # ── Integrity Verification ──

    def verify_integrity(self) -> IntegrityReport:
        """
        Perform full chain integrity verification.

        Recomputes every entry's hash and verifies the chain linkage.
        Also verifies the Merkle tree root.

        Returns:
            IntegrityReport with validation results.
        """
        if not self._chain:
            return IntegrityReport(
                is_valid=True,
                chain_length=0,
                merkle_root=GENESIS_HASH,
            )

        entry_hashes: List[str] = []

        for i, entry in enumerate(self._chain):
            # Verify previous hash linkage
            expected_prev = (
                GENESIS_HASH if i == 0
                else self._chain[i - 1].entry_hash
            )
            if entry.previous_hash != expected_prev:
                return IntegrityReport(
                    is_valid=False,
                    chain_length=len(self._chain),
                    merkle_root=self._merkle_tree.root,
                    first_invalid_index=i,
                    error_message=f"Chain break at index {i}: previous_hash mismatch",
                )

            # Recompute entry hash
            recomputed = _compute_entry_hash(
                entry.index,
                entry.timestamp,
                entry.query_hash,
                entry.node_id,
                entry.zkp_status,
                entry.authorization_token,
                entry.compliance_decision,
                entry.metadata,
                entry.previous_hash,
            )
            if recomputed != entry.entry_hash:
                return IntegrityReport(
                    is_valid=False,
                    chain_length=len(self._chain),
                    merkle_root=self._merkle_tree.root,
                    first_invalid_index=i,
                    error_message=f"Hash mismatch at index {i}: entry tampered",
                )

            entry_hashes.append(entry.entry_hash)

        # Verify Merkle root
        if not self._merkle_tree.verify(entry_hashes):
            return IntegrityReport(
                is_valid=False,
                chain_length=len(self._chain),
                merkle_root=self._merkle_tree.root,
                first_invalid_index=-1,
                error_message="Merkle root mismatch: tree tampered",
            )

        return IntegrityReport(
            is_valid=True,
            chain_length=len(self._chain),
            merkle_root=self._merkle_tree.root,
        )

    # ── Background Consumer ──

    async def _consume_loop(self) -> None:
        """Background task: drain the queue and append entries."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0,
                )
                self._append_entry(event)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error(f"[LEDGER] Consumer error: {exc}")

    def _append_entry(self, event: Dict[str, Any]) -> LedgerEntry:
        """
        Create and append a new entry to the chain.

        Computes the entry hash, links to previous entry, and
        updates the Merkle tree.
        """
        index = len(self._chain)
        previous_hash = (
            self._chain[-1].entry_hash if self._chain
            else GENESIS_HASH
        )
        timestamp = datetime.now(timezone.utc).isoformat()

        entry_hash = _compute_entry_hash(
            index=index,
            timestamp=timestamp,
            query_hash=event["query_hash"],
            node_id=event["node_id"],
            zkp_status=event["zkp_status"],
            authorization_token=event["authorization_token"],
            compliance_decision=event["compliance_decision"],
            metadata=event["metadata"],
            previous_hash=previous_hash,
        )

        # Update Merkle tree
        merkle_root = self._merkle_tree.add_leaf(entry_hash)

        entry = LedgerEntry(
            index=index,
            timestamp=timestamp,
            query_hash=event["query_hash"],
            node_id=event["node_id"],
            zkp_status=event["zkp_status"],
            authorization_token=event["authorization_token"],
            compliance_decision=event["compliance_decision"],
            metadata=event["metadata"],
            previous_hash=previous_hash,
            entry_hash=entry_hash,
            merkle_root=merkle_root,
        )

        self._chain.append(entry)

        logger.debug(
            f"[LEDGER] Entry #{index} committed — "
            f"hash={entry_hash[:16]}... "
            f"merkle_root={merkle_root[:16]}..."
        )

        return entry


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE: HASH QUERY PARAMS
# ═══════════════════════════════════════════════════════════════════════════════

def hash_query_params(
    query_embedding: List[float],
    threshold: float,
    query_id: str = "",
) -> str:
    """
    Create a SHA-256 hash of query parameters for ledger recording.

    NEVER hashes the raw DNA data — only the mathematical representation
    and search parameters. This hash is what gets stored on-chain.

    Args:
        query_embedding: The vectorized query (48-dim floats).
        threshold: Similarity threshold.
        query_id: Unique query identifier.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    canonical = json.dumps(
        {
            "embedding_hash": hashlib.sha256(
                str(query_embedding).encode()
            ).hexdigest(),
            "threshold": threshold,
            "query_id": query_id,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
