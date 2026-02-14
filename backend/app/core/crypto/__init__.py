"""
VANTAGE-STR Cryptographic Privacy Shield — Phase 4.1 / 4.2.

Zero-Knowledge Proof subsystem enabling sovereign nodes to prove DNA profile
matches without revealing the local vector.

Public API:
    - ZKPProver:    Generate ZK proofs for similarity threshold claims.
    - ZKPVerifier:  Verify proofs without access to the private vector.
    - ZKPBridge:    Unified interface with Rust/Python routing + replay guard.
    - ReplayGuard:  Proof deduplication with TTL-based expiry.
"""

from app.core.crypto.zkp_prover import (
    ZKPProver,
    ZKPVerifier,
    ZKProof,
    ReplayGuard,
    VECTOR_DIM,
)
from app.core.crypto.bridge import ZKPBridge

__all__ = [
    "ZKPProver",
    "ZKPVerifier",
    "ZKProof",
    "ZKPBridge",
    "ReplayGuard",
    "VECTOR_DIM",
]
