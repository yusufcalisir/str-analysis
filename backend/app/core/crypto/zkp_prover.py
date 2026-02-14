"""
ZKP Prover Engine — VANTAGE-STR Phase 4.1 Privacy Shield.

Non-interactive Zero-Knowledge Proof protocol enabling a sovereign Node to
prove possession of a DNA profile matching a query above a similarity
threshold, without revealing the profile vector.

═══════════════════════════════════════════════════════════════════════════════
CRYPTOGRAPHIC PROTOCOL
═══════════════════════════════════════════════════════════════════════════════

  Statement:  "I know V_local such that CosineSim(V_query, V_local) ≥ τ"
  Witness:    V_local ∈ ℝ^48  (private DNA embedding)
  Public:     V_query ∈ ℝ^48, τ ∈ [0,1]

  Protocol Stack:
    Layer 1 — Pedersen Commitment (RFC 3526 2048-bit Schnorr Group)
              C = g^{H(V_local)} · h^r  (mod p)
              Properties: Computationally Binding, Perfectly Hiding.

    Layer 2 — Schnorr Proof-of-Knowledge (Fiat-Shamir Transform)
              Non-interactive proof that the prover knows (v, r) opening C.
              Properties: Complete, Sound, HVZK → full ZK via Random Oracle.

    Layer 3 — Similarity Attestation (Commitment-Bound)
              Claimed similarity s and threshold flag are cryptographically
              bound to C via the Fiat-Shamir challenge hash.

    Layer 4 — Salted Digest (BLAKE2b-256)
              commitment_hash = BLAKE2b(C_bytes ‖ salt)
              Prevents rainbow-table attacks on common STR allele patterns.

═══════════════════════════════════════════════════════════════════════════════
SECURITY ANALYSIS
═══════════════════════════════════════════════════════════════════════════════

  Binding:   Discrete-Log assumption in the RFC 3526 group (112-bit security).
             Finding (v', r') ≠ (v, r) s.t. g^v'·h^r' = g^v·h^r ⟹ log_g(h).

  Hiding:    Perfect — commitment C is uniformly distributed for any fixed v
             because blinding factor r is sampled uniformly from Z_q.

  Soundness: Special Soundness of Schnorr protocol. Knowledge extractor
             recovers witness from two transcripts with distinct challenges.

  Zero-Knowledge:
             Honest-Verifier ZK, upgraded to full ZK via Fiat-Shamir
             in the Random Oracle Model (ROM).

  Dependencies: ZERO external packages. Pure Python stdlib only.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
import time
from dataclasses import dataclass, field as dc_field
from typing import List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# SCHNORR GROUP PARAMETERS — RFC 3526 §3, 2048-bit MODP Group
# ═══════════════════════════════════════════════════════════════════════════════
# Security level: 112 bits (NIST SP 800-57 Part 1, Table 2)
# Used in IKEv2, SSH, TLS. Extensively audited.

GROUP_P: int = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD1"
    "29024E088A67CC74020BBEA63B139B22514A08798E3404DD"
    "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245"
    "E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED"
    "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F"
    "83655D23DCA3AD961C62F356208552BB9ED529077096966D"
    "670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B"
    "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9"
    "DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)

GROUP_G: int = 2
GROUP_Q: int = (GROUP_P - 1) // 2  # Safe prime ⟹ q is also prime


def _derive_second_generator() -> int:
    """
    Derive a second generator H for Pedersen commitments.

    H = hash_to_subgroup("VANTAGE-STR-ZKP-PEDERSEN-H-v1")

    Nobody knows log_G(H) because H is derived via a cryptographic hash.
    This is the standard "nothing-up-my-sleeve" construction.
    For safe prime p, every quadratic residue ≠ 1 generates the full
    order-q subgroup, so H is guaranteed to be a valid generator.
    """
    seed = hashlib.blake2b(
        b"VANTAGE-STR-ZKP-PEDERSEN-H-v1", digest_size=64
    ).digest()
    t = int.from_bytes(seed, "big") % GROUP_P
    # Project into QR subgroup: h = t^2 mod p
    h = pow(t, 2, GROUP_P)
    assert h > 1, "Degenerate second generator — change seed string"
    return h


GROUP_H: int = _derive_second_generator()

# ── Constants ──
VECTOR_DIM: int = 48
SALT_BYTES: int = 32
FLOAT_EPSILON: float = 1e-12
PROOF_VERSION: int = 1


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _vector_to_scalar(v: List[float]) -> int:
    """
    Deterministically map a 48-dim float vector → scalar in Z_q.

    Encoding: IEEE 754 double-precision, big-endian, then BLAKE2b-256.
    The output is reduced mod q to fit the Schnorr group order.

    Args:
        v: 48-dimensional normalized DNA vector.

    Returns:
        Integer in [1, q-1].
    """
    raw = struct.pack(f">{len(v)}d", *v)
    digest = hashlib.blake2b(raw, digest_size=32).digest()
    scalar = int.from_bytes(digest, "big") % GROUP_Q
    return scalar if scalar > 0 else 1  # Avoid degenerate zero


def _cosine_similarity(u: List[float], v: List[float]) -> float:
    """
    Compute cosine similarity: S(u,v) = (u·v) / (‖u‖ × ‖v‖).

    Returns 0.0 for zero-norm vectors (undefined case).
    Result is clamped to [0.0, 1.0].

    This is a local implementation to avoid importing the vectorizer
    into the cryptographic module — separation of concerns.
    """
    if len(u) != len(v):
        raise ValueError(f"Dimension mismatch: {len(u)} vs {len(v)}")

    dot = sum(a * b for a, b in zip(u, v))
    norm_u = math.sqrt(sum(a * a for a in u))
    norm_v = math.sqrt(sum(b * b for b in v))

    if norm_u < FLOAT_EPSILON or norm_v < FLOAT_EPSILON:
        return 0.0

    sim = dot / (norm_u * norm_v)
    return max(0.0, min(1.0, sim))


def _compute_challenge(
    nonce_commitment: int,
    commitment: int,
    v_query: List[float],
    tau: float,
    claimed_similarity: float,
    threshold_met: bool,
    query_id: str = "",
) -> int:
    """
    Fiat-Shamir challenge: deterministic hash of the transcript.

    e = BLAKE2b(R ‖ C ‖ V_query ‖ τ ‖ s ‖ flag ‖ query_id)

    Binding the similarity claim, threshold flag, AND query_id into
    the challenge prevents post-hoc modification and replay attacks.
    """
    h = hashlib.blake2b(digest_size=32)
    # Nonce commitment R
    h.update(nonce_commitment.to_bytes(256, "big"))
    # Pedersen commitment C
    h.update(commitment.to_bytes(256, "big"))
    # Public query vector
    h.update(struct.pack(f">{len(v_query)}d", *v_query))
    # Threshold τ
    h.update(struct.pack(">d", tau))
    # Claimed similarity
    h.update(struct.pack(">d", claimed_similarity))
    # Threshold met flag
    h.update(struct.pack(">?", threshold_met))
    # Query ID — replay protection
    if query_id:
        h.update(query_id.encode("utf-8"))

    return int.from_bytes(h.digest(), "big") % GROUP_Q


def _salted_commitment_hash(commitment: int, salt: bytes) -> str:
    """
    BLAKE2b-256 salted digest of a Pedersen commitment.

    Prevents rainbow-table attacks on common STR marker combinations.
    The salt is per-proof and transmitted alongside the proof.
    """
    h = hashlib.blake2b(digest_size=32, salt=salt[:16])
    h.update(commitment.to_bytes(256, "big"))
    return h.hexdigest()


def _random_scalar() -> int:
    """Sample a uniform random scalar from Z_q \\ {0}."""
    while True:
        r = secrets.randbelow(GROUP_Q)
        if r > 0:
            return r


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ZKProof:
    """
    Non-interactive Zero-Knowledge Proof for DNA similarity threshold.

    Contains all components needed for independent verification.
    Serializable to bytes for gRPC transmission.

    Fields:
        commitment:         Pedersen commitment C = g^v · h^r (mod p)
        commitment_hash:    Salted BLAKE2b digest of C
        commitment_salt:    256-bit random salt (transmitted in cleartext)
        nonce_commitment:   Schnorr nonce R = g^k_v · h^k_r (mod p)
        response_v:         Schnorr response s_v = k_v + e·v (mod q)
        response_r:         Schnorr response s_r = k_r + e·r (mod q)
        challenge:          Fiat-Shamir challenge e
        claimed_similarity: Cosine similarity computed by the prover
        threshold_met:      Whether claimed_similarity ≥ τ
        query_id:           Query identifier bound into challenge (replay protection)
        proof_version:      Protocol version for forward compatibility
        timestamp:          Unix timestamp of proof generation
    """
    commitment: int
    commitment_hash: str
    commitment_salt: bytes
    nonce_commitment: int
    response_v: int
    response_r: int
    challenge: int
    claimed_similarity: float
    threshold_met: bool
    query_id: str = ""
    proof_version: int = PROOF_VERSION
    timestamp: float = dc_field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        """
        Serialize the proof to a compact binary format.

        Layout (big-endian):
            [1B version] [8B timestamp] [256B commitment] [256B nonce_commitment]
            [32B response_v] [32B response_r] [32B challenge]
            [8B claimed_sim] [1B threshold_met] [32B salt] [32B commitment_hash]
        """
        parts = [
            struct.pack(">B", self.proof_version),
            struct.pack(">d", self.timestamp),
            self.commitment.to_bytes(256, "big"),
            self.nonce_commitment.to_bytes(256, "big"),
            self.response_v.to_bytes(256, "big"),
            self.response_r.to_bytes(256, "big"),
            self.challenge.to_bytes(32, "big"),
            struct.pack(">d", self.claimed_similarity),
            struct.pack(">?", self.threshold_met),
            self.commitment_salt,
            self.commitment_hash.encode("ascii"),
        ]
        # Append query_id with length prefix
        qid_bytes = self.query_id.encode("utf-8")
        parts.append(struct.pack(">H", len(qid_bytes)))
        parts.append(qid_bytes)
        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "ZKProof":
        """Deserialize a proof from its binary format."""
        offset = 0

        version = struct.unpack_from(">B", data, offset)[0]; offset += 1
        timestamp = struct.unpack_from(">d", data, offset)[0]; offset += 8
        commitment = int.from_bytes(data[offset:offset + 256], "big"); offset += 256
        nonce_commitment = int.from_bytes(data[offset:offset + 256], "big"); offset += 256
        response_v = int.from_bytes(data[offset:offset + 256], "big"); offset += 256
        response_r = int.from_bytes(data[offset:offset + 256], "big"); offset += 256
        challenge = int.from_bytes(data[offset:offset + 32], "big"); offset += 32
        claimed_sim = struct.unpack_from(">d", data, offset)[0]; offset += 8
        threshold_met = struct.unpack_from(">?", data, offset)[0]; offset += 1
        salt = data[offset:offset + SALT_BYTES]; offset += SALT_BYTES
        hash_str = data[offset:offset + 64].decode("ascii"); offset += 64

        # Read query_id (length-prefixed)
        query_id = ""
        if offset + 2 <= len(data):
            qid_len = struct.unpack_from(">H", data, offset)[0]; offset += 2
            if qid_len > 0 and offset + qid_len <= len(data):
                query_id = data[offset:offset + qid_len].decode("utf-8")

        return cls(
            commitment=commitment,
            commitment_hash=hash_str,
            commitment_salt=salt,
            nonce_commitment=nonce_commitment,
            response_v=response_v,
            response_r=response_r,
            challenge=challenge,
            claimed_similarity=claimed_sim,
            threshold_met=threshold_met,
            query_id=query_id,
            proof_version=version,
            timestamp=timestamp,
        )

    @property
    def size_bytes(self) -> int:
        """Proof size in bytes."""
        return len(self.to_bytes())


# ═══════════════════════════════════════════════════════════════════════════════
# ZKP PROVER — Proof Generation
# ═══════════════════════════════════════════════════════════════════════════════

class ZKPProver:
    """
    Generates non-interactive ZK proofs for cosine-similarity threshold claims.

    Usage:
        prover = ZKPProver()
        proof = prover.generate_proof(v_local, v_query, tau=0.90)
        assert proof.threshold_met is True
    """

    @staticmethod
    def generate_proof(
        v_local: List[float],
        v_query: List[float],
        tau: float,
        query_id: str = "",
    ) -> ZKProof:
        """
        Generate a ZK proof that CosineSim(V_query, V_local) ≥ τ.

        Protocol:
            1. Encode V_local → scalar v_enc = BLAKE2b(V_local) mod q
            2. Sample blinding factor r ←$ Z_q
            3. Compute Pedersen commitment C = g^v_enc · h^r (mod p)
            4. Compute similarity s = CosineSim(V_query, V_local)
            5. Sample Schnorr nonces k_v, k_r ←$ Z_q
            6. Compute nonce commitment R = g^k_v · h^k_r (mod p)
            7. Fiat-Shamir challenge e = H(R ‖ C ‖ V_query ‖ τ ‖ s ‖ flag ‖ query_id)
            8. Compute responses s_v = k_v + e·v_enc (mod q),
                                 s_r = k_r + e·r    (mod q)
            9. Salt the commitment digest
           10. Package proof

        Args:
            v_local: Private 48-dim normalized DNA vector (WITNESS).
            v_query: Public 48-dim query vector from the Orchestrator.
            tau: Public similarity threshold (e.g. 0.90).
            query_id: Unique query identifier for replay protection.

        Returns:
            ZKProof containing all verification components.

        Raises:
            ValueError: If vectors have wrong dimensions.
        """
        if len(v_local) != VECTOR_DIM:
            raise ValueError(f"v_local must be {VECTOR_DIM}-dim, got {len(v_local)}")
        if len(v_query) != VECTOR_DIM:
            raise ValueError(f"v_query must be {VECTOR_DIM}-dim, got {len(v_query)}")

        # ── Step 1: Encode private vector to scalar ──
        v_enc = _vector_to_scalar(v_local)

        # ── Step 2: Sample blinding factor ──
        r = _random_scalar()

        # ── Step 3: Pedersen commitment ──
        # C = g^v_enc · h^r (mod p)
        g_v = pow(GROUP_G, v_enc, GROUP_P)
        h_r = pow(GROUP_H, r, GROUP_P)
        commitment = (g_v * h_r) % GROUP_P

        # ── Step 4: Compute actual similarity ──
        # CRITICAL: Round before hashing so prover and verifier use
        # identical float representations in the Fiat-Shamir transcript.
        similarity = round(_cosine_similarity(v_query, v_local), 10)
        threshold_met = similarity >= tau - FLOAT_EPSILON

        # ── Step 5: Schnorr nonces ──
        k_v = _random_scalar()
        k_r = _random_scalar()

        # ── Step 6: Nonce commitment ──
        # R = g^k_v · h^k_r (mod p)
        g_kv = pow(GROUP_G, k_v, GROUP_P)
        h_kr = pow(GROUP_H, k_r, GROUP_P)
        nonce_commitment = (g_kv * h_kr) % GROUP_P

        # ── Step 7: Fiat-Shamir challenge ──
        challenge = _compute_challenge(
            nonce_commitment, commitment, v_query, tau,
            similarity, threshold_met, query_id,
        )

        # ── Step 8: Schnorr responses ──
        response_v = (k_v + challenge * v_enc) % GROUP_Q
        response_r = (k_r + challenge * r) % GROUP_Q

        # ── Step 9: Salted commitment digest ──
        salt = secrets.token_bytes(SALT_BYTES)
        commitment_hash = _salted_commitment_hash(commitment, salt)

        # ── Step 10: Package ──
        return ZKProof(
            commitment=commitment,
            commitment_hash=commitment_hash,
            commitment_salt=salt,
            nonce_commitment=nonce_commitment,
            response_v=response_v,
            response_r=response_r,
            challenge=challenge,
            claimed_similarity=similarity,
            threshold_met=threshold_met,
            query_id=query_id,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ZKP VERIFIER — Proof Verification
# ═══════════════════════════════════════════════════════════════════════════════

class ZKPVerifier:
    """
    Verifies ZK proofs without access to the private DNA vector.

    The verifier checks:
        1. Threshold claim: s ≥ τ and threshold_met flag is True.
        2. Fiat-Shamir integrity: recomputed challenge matches proof challenge.
        3. Schnorr equation: g^s_v · h^s_r ≡ R · C^e (mod p).
        4. Commitment digest integrity: BLAKE2b(C ‖ salt) matches hash.

    If all checks pass, the verifier is convinced that:
        - The prover KNOWS a vector V_local that opens commitment C.
        - The prover CLAIMS CosineSim(V_query, V_local) ≥ τ.
        - The claim is cryptographically bound to the commitment.

    NOTE: The algebraic proof that the similarity was COMPUTED CORRECTLY
    from V_local requires R1CS constraints (see zkp_prover.rs). This
    verifier validates proof-of-knowledge + bound attestation.

    Usage:
        verifier = ZKPVerifier()
        is_valid = verifier.verify_proof(proof, v_query, tau=0.90)
    """

    @staticmethod
    def verify_proof(
        proof: ZKProof,
        v_query: List[float],
        tau: float,
        query_id: str = "",
    ) -> bool:
        """
        Verify a ZK proof of similarity threshold.

        Args:
            proof: ZKProof to verify.
            v_query: Public query vector (same as used in proof generation).
            tau: Public similarity threshold.
            query_id: Query ID that must match the one bound in the proof.

        Returns:
            True if proof is valid, False otherwise.
        """
        # ── Check 0: Structural validation ──
        if proof.proof_version != PROOF_VERSION:
            return False
        if len(v_query) != VECTOR_DIM:
            return False

        # ── Check 1: Threshold attestation ──
        if not proof.threshold_met:
            return False
        if proof.claimed_similarity < tau - FLOAT_EPSILON:
            return False

        # ── Check 1.5: Query ID binding (replay protection) ──
        if query_id and proof.query_id != query_id:
            return False

        # ── Check 2: Fiat-Shamir challenge integrity ──
        expected_challenge = _compute_challenge(
            proof.nonce_commitment,
            proof.commitment,
            v_query,
            tau,
            proof.claimed_similarity,
            proof.threshold_met,
            proof.query_id,
        )
        if expected_challenge != proof.challenge:
            return False

        # ── Check 3: Schnorr verification equation ──
        # g^s_v · h^s_r  ≟  R · C^e  (mod p)
        #
        # Proof:
        #   LHS = g^(k_v + e·v) · h^(k_r + e·r)
        #       = g^k_v · g^(e·v) · h^k_r · h^(e·r)
        #       = (g^k_v · h^k_r) · (g^v · h^r)^e
        #       = R · C^e = RHS  ✓
        lhs = (
            pow(GROUP_G, proof.response_v, GROUP_P)
            * pow(GROUP_H, proof.response_r, GROUP_P)
        ) % GROUP_P

        rhs = (
            proof.nonce_commitment
            * pow(proof.commitment, proof.challenge, GROUP_P)
        ) % GROUP_P

        if lhs != rhs:
            return False

        # ── Check 4: Commitment digest integrity ──
        expected_hash = _salted_commitment_hash(
            proof.commitment, proof.commitment_salt,
        )
        if expected_hash != proof.commitment_hash:
            return False

        return True


# ═══════════════════════════════════════════════════════════════════════════════
# REPLAY GUARD — Proof Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

class ReplayGuard:
    """
    Prevents replay attacks by tracking consumed proof identifiers.

    Each accepted proof is identified by (commitment_hash, query_id).
    Duplicate submissions within the TTL window are rejected.

    Thread Safety:
        Uses a simple dict. For production multi-threaded environments,
        wrap check_and_record with asyncio.Lock or threading.Lock.
    """

    def __init__(self, ttl_seconds: float = 300.0) -> None:
        """
        Args:
            ttl_seconds: Time-to-live for recorded proofs. After this
                         period, proofs are evicted and could theoretically
                         be replayed — but the query_id binding in the
                         Fiat-Shamir challenge makes this harmless as long
                         as query IDs are not reused.
        """
        self._seen: dict[str, float] = {}  # key → timestamp
        self._ttl = ttl_seconds

    def _make_key(self, commitment_hash: str, query_id: str) -> str:
        """Compound key for deduplication."""
        return f"{commitment_hash}:{query_id}"

    def check_and_record(self, commitment_hash: str, query_id: str) -> bool:
        """
        Check if a proof has already been consumed.

        Returns:
            True if this is a NEW proof (accepted).
            False if this proof was already seen (replay detected).
        """
        self._evict_expired()
        key = self._make_key(commitment_hash, query_id)
        if key in self._seen:
            return False  # Replay detected
        self._seen[key] = time.time()
        return True

    def _evict_expired(self) -> None:
        """Remove entries older than TTL."""
        cutoff = time.time() - self._ttl
        expired = [k for k, ts in self._seen.items() if ts < cutoff]
        for k in expired:
            del self._seen[k]

    @property
    def size(self) -> int:
        """Number of tracked proofs."""
        return len(self._seen)
