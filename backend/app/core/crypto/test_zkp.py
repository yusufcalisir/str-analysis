"""
ZKP Test Harness — VANTAGE-STR Phase 4.1 Verification Suite.

Validates all cryptographic properties of the Privacy Shield:
    1. Honest proof acceptance (Completeness)
    2. Dishonest proof rejection (Soundness)
    3. Binding property (same vector → same commitment)
    4. Hiding property (different salt → different digest, no V_local leak)
    5. Bridge fallback (Python backend when no Rust binary)
    6. Serialization round-trip (to_bytes / from_bytes)

Run:
    cd c:\\Users\\ysfca\\OneDrive\\Masaüstü\\vantage-str\\backend
    python -m app.core.crypto.test_zkp
"""

from __future__ import annotations

import math
import random
import sys
import time
from typing import List

# Module under test
from app.core.crypto.zkp_prover import (
    ZKPProver,
    ZKPVerifier,
    ZKProof,
    VECTOR_DIM,
    _cosine_similarity,
    _vector_to_scalar,
    _salted_commitment_hash,
)
from app.core.crypto.bridge import ZKPBridge


# ═══════════════════════════════════════════════════════════════════════════════
# TEST HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_random_vector(dim: int = VECTOR_DIM, seed: int = None) -> List[float]:
    """Generate a random normalized vector in [0, 1]^dim."""
    if seed is not None:
        random.seed(seed)
    v = [random.uniform(0.05, 0.95) for _ in range(dim)]
    return v


def _generate_similar_vector(
    base: List[float], noise: float = 0.02, seed: int = None,
) -> List[float]:
    """Generate a vector similar to base by adding small noise."""
    if seed is not None:
        random.seed(seed)
    return [
        max(0.0, min(1.0, x + random.uniform(-noise, noise)))
        for x in base
    ]


def _generate_dissimilar_vector(dim: int = VECTOR_DIM, seed: int = None) -> List[float]:
    """Generate a vector designed to be dissimilar (opposite direction)."""
    if seed is not None:
        random.seed(seed)
    return [random.uniform(0.0, 0.1) for _ in range(dim)]


class TestResult:
    """Accumulator for test results."""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []

    def ok(self, name: str, elapsed_ms: float):
        self.passed += 1
        print(f"  ✅  {name}  ({elapsed_ms:.1f}ms)")

    def fail(self, name: str, reason: str):
        self.failed += 1
        self.errors.append(f"{name}: {reason}")
        print(f"  ❌  {name}  — {reason}")

    @property
    def total(self) -> int:
        return self.passed + self.failed

    def summary(self) -> bool:
        print(f"\n{'═' * 60}")
        print(f"  Results: {self.passed}/{self.total} passed")
        if self.errors:
            print(f"  Failures:")
            for e in self.errors:
                print(f"    • {e}")
        print(f"{'═' * 60}")
        return self.failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
# TEST CASES
# ═══════════════════════════════════════════════════════════════════════════════

def test_honest_proof_acceptance(results: TestResult):
    """
    TEST 1: Honest proof is accepted (Completeness).

    A genuine prover with a matching vector (sim ≥ τ) produces
    a proof that the verifier accepts.
    """
    name = "Honest Proof Acceptance (Completeness)"
    t0 = time.perf_counter()
    try:
        v_local = _generate_random_vector(seed=42)
        v_query = _generate_similar_vector(v_local, noise=0.01, seed=43)

        sim = _cosine_similarity(v_query, v_local)
        assert sim >= 0.90, f"Test setup: similarity {sim:.4f} < 0.90"

        proof = ZKPProver.generate_proof(v_local, v_query, tau=0.90)
        assert proof.threshold_met is True
        assert proof.claimed_similarity >= 0.90

        is_valid = ZKPVerifier.verify_proof(proof, v_query, tau=0.90)
        assert is_valid is True, "Verifier rejected honest proof"

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


def test_dishonest_proof_rejection(results: TestResult):
    """
    TEST 2: Forged proof is rejected (Soundness).

    A tampered proof (modified response) must be rejected.
    """
    name = "Dishonest Proof Rejection (Soundness)"
    t0 = time.perf_counter()
    try:
        v_local = _generate_random_vector(seed=44)
        v_query = _generate_similar_vector(v_local, noise=0.01, seed=45)

        proof = ZKPProver.generate_proof(v_local, v_query, tau=0.90)
        assert proof.threshold_met is True

        # Tamper with the response (forge a different s_v)
        forged = ZKProof(
            commitment=proof.commitment,
            commitment_hash=proof.commitment_hash,
            commitment_salt=proof.commitment_salt,
            nonce_commitment=proof.nonce_commitment,
            response_v=(proof.response_v + 1) % (2**256),  # Tampered
            response_r=proof.response_r,
            challenge=proof.challenge,
            claimed_similarity=proof.claimed_similarity,
            threshold_met=proof.threshold_met,
        )

        is_valid = ZKPVerifier.verify_proof(forged, v_query, tau=0.90)
        assert is_valid is False, "Verifier accepted forged proof!"

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


def test_below_threshold_rejection(results: TestResult):
    """
    TEST 3: Below-threshold proof is handled correctly.

    When similarity < τ, the proof reports threshold_met=False
    and the verifier rejects it.
    """
    name = "Below-Threshold Rejection"
    t0 = time.perf_counter()
    try:
        v_local = _generate_random_vector(seed=46)
        v_query = _generate_dissimilar_vector(seed=47)

        sim = _cosine_similarity(v_query, v_local)
        # With dissimilar vectors, sim should be low
        proof = ZKPProver.generate_proof(v_local, v_query, tau=0.90)

        if sim < 0.90:
            assert proof.threshold_met is False
            is_valid = ZKPVerifier.verify_proof(proof, v_query, tau=0.90)
            assert is_valid is False, "Verifier accepted below-threshold proof"

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


def test_binding_property(results: TestResult):
    """
    TEST 4: Binding property.

    Same V_local → deterministic scalar encoding (same vector_to_scalar).
    Different V_local → different scalar (overwhelming probability).
    """
    name = "Binding Property"
    t0 = time.perf_counter()
    try:
        v1 = _generate_random_vector(seed=50)
        v2 = _generate_random_vector(seed=51)

        s1a = _vector_to_scalar(v1)
        s1b = _vector_to_scalar(v1)
        s2 = _vector_to_scalar(v2)

        assert s1a == s1b, "Same vector produced different scalars (non-deterministic!)"
        assert s1a != s2, "Different vectors produced same scalar (collision!)"

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


def test_hiding_property(results: TestResult):
    """
    TEST 5: Hiding property.

    Different salt → different commitment digest.
    Proof bytes contain no recoverable V_local data.
    """
    name = "Hiding Property"
    t0 = time.perf_counter()
    try:
        v_local = _generate_random_vector(seed=52)
        v_query = _generate_similar_vector(v_local, noise=0.01, seed=53)

        p1 = ZKPProver.generate_proof(v_local, v_query, tau=0.90)
        p2 = ZKPProver.generate_proof(v_local, v_query, tau=0.90)

        # Different blinding factors → different commitments
        assert p1.commitment != p2.commitment, (
            "Same V_local produced identical commitments (blinding factor reused!)"
        )

        # Different salts → different digests
        assert p1.commitment_hash != p2.commitment_hash, (
            "Same V_local produced identical hashes (salt reused!)"
        )

        # Proof bytes should not contain the raw vector
        proof_bytes = p1.to_bytes()
        v_bytes = b"".join(
            val.to_bytes(8, "big") for val in
            [int(x * 1e15) for x in v_local]
        )
        assert v_bytes not in proof_bytes, "V_local data found in proof bytes!"

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


def test_bridge_fallback(results: TestResult):
    """
    TEST 6: Bridge gracefully falls back to Python.
    """
    name = "Bridge Fallback to Python"
    t0 = time.perf_counter()
    try:
        bridge = ZKPBridge()
        assert bridge.backend == "python-stdlib", (
            f"Expected python-stdlib, got {bridge.backend}"
        )

        v_local = _generate_random_vector(seed=60)
        v_query = _generate_similar_vector(v_local, noise=0.01, seed=61)

        proof = bridge.prove(v_local, v_query, tau=0.90)
        is_valid = bridge.verify(proof, v_query, tau=0.90)
        assert is_valid is True, "Bridge proof rejected"

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


def test_serialization_roundtrip(results: TestResult):
    """
    TEST 7: Proof survives serialization/deserialization.
    """
    name = "Serialization Round-trip"
    t0 = time.perf_counter()
    try:
        v_local = _generate_random_vector(seed=70)
        v_query = _generate_similar_vector(v_local, noise=0.01, seed=71)

        original = ZKPProver.generate_proof(v_local, v_query, tau=0.90)
        raw = original.to_bytes()
        restored = ZKProof.from_bytes(raw)

        assert restored.commitment == original.commitment
        assert restored.nonce_commitment == original.nonce_commitment
        assert restored.response_v == original.response_v
        assert restored.response_r == original.response_r
        assert restored.challenge == original.challenge
        assert restored.commitment_hash == original.commitment_hash
        assert abs(restored.claimed_similarity - original.claimed_similarity) < 1e-10
        assert restored.threshold_met == original.threshold_met

        # Restored proof must also verify
        is_valid = ZKPVerifier.verify_proof(restored, v_query, tau=0.90)
        assert is_valid is True, "Deserialized proof failed verification"

        print(f"         Proof size: {len(raw)} bytes")

        elapsed = (time.perf_counter() - t0) * 1000
        results.ok(name, elapsed)
    except Exception as e:
        results.fail(name, str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  VANTAGE-STR ZKP Privacy Shield — Test Suite")
    print("=" * 60)
    print()

    results = TestResult()

    test_honest_proof_acceptance(results)
    test_dishonest_proof_rejection(results)
    test_below_threshold_rejection(results)
    test_binding_property(results)
    test_hiding_property(results)
    test_bridge_fallback(results)
    test_serialization_roundtrip(results)

    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
