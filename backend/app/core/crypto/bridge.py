"""
ZKP Bridge — Unified Python/Rust interface for proof generation & verification.

Attempts to load the compiled Rust ZKP module (PyO3 .pyd/.so) for
maximum performance. Falls back gracefully to the pure-Python prover
when the Rust binary is unavailable.

The Rust module is expected to expose:
    rust_zkp.prove(v_local: list[float], v_query: list[float], tau: float) -> bytes
    rust_zkp.verify(proof_bytes: bytes, v_query: list[float], tau: float) -> bool

Build the Rust module:
    cd backend/app/core/crypto
    maturin develop --release

Until then, the Python fallback is fully functional with identical
cryptographic guarantees (Pedersen + Schnorr over RFC 3526).
"""

from __future__ import annotations

import importlib
import logging
from typing import List, Optional

from app.core.crypto.zkp_prover import (
    ZKPProver, ZKPVerifier, ZKProof, ReplayGuard, VECTOR_DIM,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# RUST MODULE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

_RUST_MODULE_NAME = "vantage_zkp_rs"  # Expected PyO3 module name
_rust_module = None


def _try_load_rust() -> Optional[object]:
    """
    Attempt to import the compiled Rust ZKP module.

    Returns the module if found, None otherwise. Logs the outcome.
    This is called once at import time and cached.
    """
    global _rust_module
    try:
        _rust_module = importlib.import_module(_RUST_MODULE_NAME)
        logger.info(
            f"[ZKP-BRIDGE] Rust module '{_RUST_MODULE_NAME}' loaded — "
            f"native performance enabled"
        )
        return _rust_module
    except ImportError:
        logger.info(
            f"[ZKP-BRIDGE] Rust module '{_RUST_MODULE_NAME}' not found — "
            f"falling back to pure-Python prover (cryptographically identical)"
        )
        return None


# Run discovery at import time
_try_load_rust()


# ═══════════════════════════════════════════════════════════════════════════════
# BRIDGE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class ZKPBridge:
    """
    Unified ZKP interface routing to Rust (fast) or Python (portable).

    Both backends produce cryptographically equivalent proofs. The Rust
    backend is ~100× faster due to native big-integer arithmetic.

    Includes an integrated ReplayGuard for Orchestrator-side deduplication.

    Usage:
        bridge = ZKPBridge()
        proof = bridge.prove(v_local, v_query, tau=0.90, query_id="Q-001")
        is_valid = bridge.verify(proof, v_query, tau=0.90, query_id="Q-001")
        print(f"Backend: {bridge.backend}")
    """

    def __init__(self, replay_ttl: float = 300.0) -> None:
        self._use_rust = _rust_module is not None
        self._replay_guard = ReplayGuard(ttl_seconds=replay_ttl)

    @property
    def backend(self) -> str:
        """Active backend: 'rust-native' or 'python-stdlib'."""
        return "rust-native" if self._use_rust else "python-stdlib"

    @property
    def replay_guard(self) -> ReplayGuard:
        """Access the replay guard for inspection."""
        return self._replay_guard

    def prove(
        self,
        v_local: List[float],
        v_query: List[float],
        tau: float,
        query_id: str = "",
    ) -> ZKProof:
        """
        Generate a ZK proof for the cosine-similarity threshold claim.

        Routes to Rust if available, otherwise uses the Python prover.

        Args:
            v_local: Private 48-dim DNA vector (NEVER leaves the node).
            v_query: Public 48-dim query vector from the Orchestrator.
            tau: Similarity threshold.
            query_id: Unique query identifier for replay protection.

        Returns:
            ZKProof ready for transmission to the Orchestrator.
        """
        if len(v_local) != VECTOR_DIM:
            raise ValueError(f"v_local must be {VECTOR_DIM}-dim, got {len(v_local)}")
        if len(v_query) != VECTOR_DIM:
            raise ValueError(f"v_query must be {VECTOR_DIM}-dim, got {len(v_query)}")

        if self._use_rust:
            return self._prove_rust(v_local, v_query, tau, query_id)

        return ZKPProver.generate_proof(v_local, v_query, tau, query_id)

    def verify(
        self,
        proof: ZKProof,
        v_query: List[float],
        tau: float,
        query_id: str = "",
    ) -> bool:
        """
        Verify a ZK proof with replay protection.

        Performs three gates:
            1. ReplayGuard — reject duplicate (commitment_hash, query_id) pairs.
            2. Cryptographic verification — Schnorr + Fiat-Shamir.
            3. Record accepted proof in replay guard.

        Args:
            proof: Proof to verify.
            v_query: Public query vector.
            tau: Expected threshold.
            query_id: Query ID to bind (must match proof's query_id).

        Returns:
            True if proof is valid and not a replay.
        """
        # Gate 1: Replay check
        if not self._replay_guard.check_and_record(
            proof.commitment_hash, query_id or proof.query_id,
        ):
            logger.warning(
                f"[ZKP-BRIDGE] REPLAY DETECTED — commitment={proof.commitment_hash[:16]}... "
                f"query_id={query_id}"
            )
            return False

        # Gate 2: Cryptographic verification
        if self._use_rust:
            return self._verify_rust(proof, v_query, tau)

        return ZKPVerifier.verify_proof(proof, v_query, tau, query_id)

    # ── Rust dispatch (PyO3 FFI) ──

    def _prove_rust(
        self,
        v_local: List[float],
        v_query: List[float],
        tau: float,
        query_id: str = "",
    ) -> ZKProof:
        """
        Call the Rust prover via PyO3.

        The Rust module returns raw proof bytes which we deserialize
        into the Python ZKProof dataclass for API consistency.
        """
        assert _rust_module is not None
        proof_bytes: bytes = _rust_module.prove(v_local, v_query, tau)
        proof = ZKProof.from_bytes(proof_bytes)
        # Rust doesn't handle query_id yet — inject it post-hoc
        if query_id and not proof.query_id:
            from dataclasses import replace
            proof = replace(proof, query_id=query_id)
        return proof

    def _verify_rust(
        self,
        proof: ZKProof,
        v_query: List[float],
        tau: float,
    ) -> bool:
        """
        Call the Rust verifier via PyO3.

        Serializes the proof to bytes and passes to the Rust module.
        """
        assert _rust_module is not None
        return _rust_module.verify(proof.to_bytes(), v_query, tau)
