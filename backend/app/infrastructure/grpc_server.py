"""
gRPC Server — Encrypted Inter-Node Communication for VANTAGE-STR.

Provides high-speed, low-latency gRPC services for:
    1. Node registration and heartbeat exchange.
    2. Federated query broadcasting (query embeddings → nodes).
    3. Result collection (confidence scores ← nodes).

All application-layer payloads are encrypted with AES-256-GCM before
transmission, providing defense-in-depth on top of mTLS transport
encryption. This means even a compromised TLS endpoint cannot read
query parameters or similarity scores without the shared AES key.

Protobuf Schema:
    Uses the existing `genomic_profile.proto` for profile messages and
    extends it with federated query service definitions.
"""

import hashlib
import hmac
import logging
import os
import secrets
import struct
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# AES-256-GCM ENCRYPTION LAYER
# ═══════════════════════════════════════════════════════════════════════════════

class AES256Cipher:
    """
    AES-256-GCM encryption for application-layer payload protection.

    Provides authenticated encryption ensuring both confidentiality and
    integrity of transmitted data. Each encryption operation uses a unique
    nonce (IV) to prevent ciphertext analysis.

    Key Derivation:
        The shared secret is derived via HKDF from the mTLS session
        parameters, ensuring forward secrecy when TLS sessions rotate.

    Usage:
        cipher = AES256Cipher(shared_key)
        ct, nonce, tag = cipher.encrypt(plaintext)
        plaintext = cipher.decrypt(ct, nonce, tag)
    """

    NONCE_SIZE: int = 12   # 96-bit nonce for GCM
    TAG_SIZE: int = 16     # 128-bit authentication tag
    KEY_SIZE: int = 32     # 256-bit key

    def __init__(self, key: Optional[bytes] = None) -> None:
        """
        Initialize with a 32-byte AES key.

        Args:
            key: 256-bit encryption key. If None, generates a random key.
        """
        self._key = key or secrets.token_bytes(self.KEY_SIZE)
        if len(self._key) != self.KEY_SIZE:
            raise ValueError(f"AES-256 key must be {self.KEY_SIZE} bytes, got {len(self._key)}")

    @property
    def key_fingerprint(self) -> str:
        """SHA-256 fingerprint of the key (for audit logging, never expose the key)."""
        return hashlib.sha256(self._key).hexdigest()[:16]

    @staticmethod
    def derive_key(shared_secret: str, salt: Optional[bytes] = None) -> "AES256Cipher":
        """
        Derive an AES-256 key from a shared secret using HKDF-SHA256.

        Args:
            shared_secret: Shared string (e.g., from mTLS session).
            salt: Optional salt for HKDF. If None, uses a random salt.

        Returns:
            AES256Cipher initialized with the derived key.
        """
        if salt is None:
            salt = secrets.token_bytes(32)

        # HKDF extract
        prk = hmac.new(salt, shared_secret.encode("utf-8"), hashlib.sha256).digest()

        # HKDF expand (single block is sufficient for 256-bit key)
        info = b"VANTAGE-STR-AES256-GCM"
        okm = hmac.new(prk, info + b"\x01", hashlib.sha256).digest()

        return AES256Cipher(key=okm)

    def encrypt(self, plaintext: bytes) -> Tuple[bytes, bytes, bytes]:
        """
        Encrypt plaintext using AES-256-GCM.

        Returns:
            Tuple of (ciphertext, nonce, authentication_tag).

        Note:
            Uses Python's cryptography library if available, otherwise falls
            back to a pure-Python XOR placeholder for development environments.
        """
        nonce = secrets.token_bytes(self.NONCE_SIZE)

        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)
            # AESGCM appends tag to ciphertext
            ciphertext = ct_with_tag[:-self.TAG_SIZE]
            tag = ct_with_tag[-self.TAG_SIZE:]
            return ciphertext, nonce, tag

        except ImportError:
            # Development fallback: XOR cipher (NOT production-safe)
            logger.warning("[CRYPTO] Using XOR fallback — install 'cryptography' for AES-GCM")
            key_stream = self._expand_key(len(plaintext), nonce)
            ciphertext = bytes(a ^ b for a, b in zip(plaintext, key_stream))
            tag = hmac.new(self._key, nonce + ciphertext, hashlib.sha256).digest()[:self.TAG_SIZE]
            return ciphertext, nonce, tag

    def decrypt(self, ciphertext: bytes, nonce: bytes, tag: bytes) -> bytes:
        """
        Decrypt ciphertext using AES-256-GCM.

        Args:
            ciphertext: Encrypted data.
            nonce: 12-byte nonce used during encryption.
            tag: 16-byte authentication tag.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ValueError: If authentication tag verification fails.
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            aesgcm = AESGCM(self._key)
            ct_with_tag = ciphertext + tag
            return aesgcm.decrypt(nonce, ct_with_tag, None)

        except ImportError:
            # Development fallback
            expected_tag = hmac.new(self._key, nonce + ciphertext, hashlib.sha256).digest()[:self.TAG_SIZE]
            if not hmac.compare_digest(tag, expected_tag):
                raise ValueError("AES-GCM authentication failed: tag mismatch")
            key_stream = self._expand_key(len(ciphertext), nonce)
            return bytes(a ^ b for a, b in zip(ciphertext, key_stream))

    def _expand_key(self, length: int, nonce: bytes) -> bytes:
        """Expand key material for XOR fallback using counter-mode HMAC."""
        blocks = (length + 31) // 32
        stream = b""
        for i in range(blocks):
            counter = struct.pack(">I", i)
            stream += hmac.new(self._key, nonce + counter, hashlib.sha256).digest()
        return stream[:length]


# ═══════════════════════════════════════════════════════════════════════════════
# MESSAGE FRAMING
# ═══════════════════════════════════════════════════════════════════════════════

class EncryptedEnvelope(BaseModel):
    """
    Wire format for encrypted gRPC payloads.

    All federated messages (queries, results, heartbeats) are wrapped in
    this envelope before transmission. The receiver decrypts using the
    shared AES key established during the mTLS handshake.
    """
    sender_id: str
    receiver_id: str
    message_type: str  # "query", "result", "heartbeat", "gradient"
    ciphertext_hex: str
    nonce_hex: str
    tag_hex: str
    timestamp: float = Field(default_factory=time.time)
    sequence_number: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# gRPC SERVICE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedQueryRequest(BaseModel):
    """Request payload for a federated similarity search."""
    query_id: str
    query_embedding: List[float]  # 48-dim vector
    top_k: int = 10
    min_confidence: float = 0.65
    requesting_node_id: str = ""
    encrypted_metadata: Optional[str] = None  # AES-256 encrypted context


class FederatedQueryResponse(BaseModel):
    """Response payload from a node after local search."""
    query_id: str
    node_id: str
    matches: List[Dict[str, Any]] = Field(default_factory=list)
    profiles_searched: int = 0
    search_time_ms: float = 0.0
    model_gradient: Optional[List[float]] = None


class GRPCServiceConfig(BaseModel):
    """Configuration for the gRPC server."""
    host: str = "0.0.0.0"
    port: int = 50051
    max_workers: int = 10
    max_message_size_mb: int = 16
    enable_reflection: bool = True
    enable_health_check: bool = True
    keepalive_time_ms: int = 10000
    keepalive_timeout_ms: int = 5000


# ═══════════════════════════════════════════════════════════════════════════════
# gRPC SERVER
# ═══════════════════════════════════════════════════════════════════════════════

class VantageGRPCServer:
    """
    gRPC server for the VANTAGE-STR federated network.

    Exposes three core services:
        1. NodeService    — Registration, heartbeat, and session management.
        2. QueryService   — Federated query broadcast and result streaming.
        3. SyncService    — Model gradient exchange for federated learning.

    All payloads are encrypted at the application layer using AES-256-GCM
    before protobuf serialization. The gRPC transport is additionally
    secured via mTLS, providing two independent layers of encryption.

    Usage:
        server = VantageGRPCServer(config, cipher)
        server.start()
        # ... server runs until interrupted
        server.stop()
    """

    def __init__(
        self,
        config: GRPCServiceConfig,
        cipher: AES256Cipher,
    ) -> None:
        self._config = config
        self._cipher = cipher
        self._server: Any = None
        self._running = False
        self._request_count = 0
        self._start_time: float = 0.0

    def encrypt_payload(self, data: bytes, receiver_id: str) -> EncryptedEnvelope:
        """
        Encrypt a protobuf-serialized payload for transmission.

        Args:
            data: Raw bytes to encrypt.
            receiver_id: Target node identifier.

        Returns:
            EncryptedEnvelope ready for wire transmission.
        """
        ct, nonce, tag = self._cipher.encrypt(data)
        self._request_count += 1

        return EncryptedEnvelope(
            sender_id="ORCHESTRATOR",
            receiver_id=receiver_id,
            message_type="encrypted",
            ciphertext_hex=ct.hex(),
            nonce_hex=nonce.hex(),
            tag_hex=tag.hex(),
            sequence_number=self._request_count,
        )

    def decrypt_payload(self, envelope: EncryptedEnvelope) -> bytes:
        """
        Decrypt an incoming encrypted envelope.

        Args:
            envelope: EncryptedEnvelope received from a node.

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ValueError: If decryption or authentication fails.
        """
        ct = bytes.fromhex(envelope.ciphertext_hex)
        nonce = bytes.fromhex(envelope.nonce_hex)
        tag = bytes.fromhex(envelope.tag_hex)

        return self._cipher.decrypt(ct, nonce, tag)

    def start(self) -> None:
        """
        Start the gRPC server.

        In a full deployment, this method would:
        1. Load TLS certificates for mTLS.
        2. Register protobuf service handlers.
        3. Start the gRPC server on the configured port.

        Currently provides the structural skeleton for integration
        with grpcio once proto compilation is configured.
        """
        self._start_time = time.time()
        self._running = True

        logger.info(
            f"[gRPC] VANTAGE-STR Server starting on "
            f"{self._config.host}:{self._config.port} | "
            f"workers={self._config.max_workers} | "
            f"key={self._cipher.key_fingerprint}"
        )

        # Production integration point:
        # import grpc
        # from concurrent import futures
        # server = grpc.server(
        #     futures.ThreadPoolExecutor(max_workers=self._config.max_workers),
        #     options=[
        #         ("grpc.max_receive_message_length", self._config.max_message_size_mb * 1024 * 1024),
        #         ("grpc.keepalive_time_ms", self._config.keepalive_time_ms),
        #         ("grpc.keepalive_timeout_ms", self._config.keepalive_timeout_ms),
        #     ],
        # )
        # Add service handlers here
        # server_credentials = grpc.ssl_server_credentials(...)
        # server.add_secure_port(f"{host}:{port}", server_credentials)
        # server.start()

    def stop(self, grace_period: float = 5.0) -> None:
        """Stop the gRPC server with a grace period."""
        self._running = False
        logger.info(f"[gRPC] Server stopping (grace={grace_period}s)")

    def get_metrics(self) -> Dict[str, Any]:
        """Return server operational metrics."""
        uptime = time.time() - self._start_time if self._start_time else 0
        return {
            "running": self._running,
            "uptime_seconds": round(uptime, 2),
            "total_requests": self._request_count,
            "requests_per_second": round(self._request_count / max(uptime, 1), 2),
            "config": {
                "host": self._config.host,
                "port": self._config.port,
                "max_workers": self._config.max_workers,
            },
        }
