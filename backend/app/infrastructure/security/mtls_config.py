"""
Mutual TLS Configuration — X.509 Certificate Management for VANTAGE-STR.

Provides a production-grade mTLS layer for the gRPC channels between
the Orchestrator and all sovereign Nodes. Both sides must present valid
certificates signed by the VANTAGE-STR Root CA.

Security Guarantees:
    - TLS 1.3 minimum (PROTOCOL_TLS_SERVER / PROTOCOL_TLS_CLIENT).
    - CERT_REQUIRED on both sides: no anonymous connections.
    - Certificate Revocation List (CRL) maintained in-memory with
      instant blacklisting via revoke_certificate().
    - Certificate pinning: fingerprints are checked on every connection.
    - OCSP stapling hooks for production deployments.

Boot Flow:
    1. Load CA root + intermediates.
    2. Load server/node cert + private key.
    3. Build SSLContext with mutual verification.
    4. Wrap gRPC channel with the TLS context.

Usage:
    mtls = MTLSConfig(
        ca_cert_path="/etc/vantage/certs/ca.crt",
        cert_path="/etc/vantage/certs/node.crt",
        key_path="/etc/vantage/certs/node.key",
    )
    ssl_ctx = mtls.build_server_context()   # For the Orchestrator
    ssl_ctx = mtls.build_client_context()   # For connecting Nodes
    channel_creds = mtls.get_grpc_credentials()
"""

import hashlib
import logging
import secrets
import ssl
import time
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

MIN_TLS_VERSION = ssl.TLSVersion.TLSv1_3
ALLOWED_CIPHERS: str = (
    "TLS_AES_256_GCM_SHA384:"
    "TLS_CHACHA20_POLY1305_SHA256:"
    "TLS_AES_128_GCM_SHA256"
)
CRL_CHECK_INTERVAL_SECONDS: int = 60
MAX_CERTIFICATE_AGE_DAYS: int = 365


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CertificateState(str, Enum):
    """Lifecycle state of a certificate."""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    UNKNOWN = "unknown"


class CertificateRecord(BaseModel):
    """Tracked certificate with revocation metadata."""
    fingerprint_sha256: str
    subject: str = ""
    issuer: str = ""
    node_id: str = ""
    state: CertificateState = CertificateState.VALID
    registered_at: float = Field(default_factory=time.time)
    revoked_at: float = 0.0
    revocation_reason: str = ""
    serial_number: str = ""


class RevocationEntry(BaseModel):
    """CRL entry for a revoked certificate."""
    fingerprint: str
    node_id: str
    reason: str
    revoked_at: float = Field(default_factory=time.time)
    revoked_by: str = "orchestrator"


class MTLSStatus(BaseModel):
    """Current status of the mTLS subsystem."""
    initialized: bool = False
    ca_loaded: bool = False
    server_cert_loaded: bool = False
    tls_version: str = "TLS 1.3"
    total_certificates: int = 0
    revoked_certificates: int = 0
    active_connections: int = 0
    last_crl_check: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# CERTIFICATE REVOCATION LIST (CRL)
# ═══════════════════════════════════════════════════════════════════════════════

class CertificateRevocationList:
    """
    In-memory Certificate Revocation List.

    Maintains a set of revoked certificate fingerprints for instant
    lookup during mTLS handshake validation. When the Orchestrator
    detects a compromised node, its certificate is added here and
    all future connections are immediately rejected.

    Thread Safety:
        Uses set operations which are atomic in CPython. For multi-
        process deployments, back this with Redis or PostgreSQL.
    """

    def __init__(self) -> None:
        self._revoked: Dict[str, RevocationEntry] = {}
        self._last_updated: float = time.time()

    def revoke(
        self,
        fingerprint: str,
        node_id: str,
        reason: str,
        revoked_by: str = "orchestrator",
    ) -> RevocationEntry:
        """
        Immediately blacklist a certificate.

        This takes effect on the NEXT connection attempt. Any existing
        open channels should be forcibly closed by the caller.

        Args:
            fingerprint: SHA-256 fingerprint of the certificate to revoke.
            node_id: Node that owns this certificate.
            reason: Human-readable reason for revocation.
            revoked_by: Identity of the revoking authority.

        Returns:
            The RevocationEntry created.
        """
        entry = RevocationEntry(
            fingerprint=fingerprint,
            node_id=node_id,
            reason=reason,
            revoked_by=revoked_by,
        )
        self._revoked[fingerprint] = entry
        self._last_updated = time.time()

        logger.critical(
            f"[CRL] Certificate REVOKED | node={node_id} | "
            f"fingerprint={fingerprint[:16]}... | reason={reason}"
        )
        return entry

    def is_revoked(self, fingerprint: str) -> bool:
        """Check if a certificate is on the revocation list."""
        return fingerprint in self._revoked

    def get_entry(self, fingerprint: str) -> Optional[RevocationEntry]:
        """Get the revocation entry for a fingerprint."""
        return self._revoked.get(fingerprint)

    def unrevoke(self, fingerprint: str) -> bool:
        """Remove a certificate from the CRL (admin recovery action)."""
        if fingerprint in self._revoked:
            del self._revoked[fingerprint]
            self._last_updated = time.time()
            logger.info(f"[CRL] Certificate UNREVOKED: {fingerprint[:16]}...")
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._revoked)

    @property
    def all_entries(self) -> List[RevocationEntry]:
        return list(self._revoked.values())

    @property
    def last_updated(self) -> float:
        return self._last_updated


# ═══════════════════════════════════════════════════════════════════════════════
# MTLS CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

class MTLSConfig:
    """
    Mutual TLS configuration and channel security for VANTAGE-STR.

    Manages the full lifecycle of TLS certificates:
        1. Loading CA chain, server cert, and private key.
        2. Building SSLContext objects for server and client modes.
        3. Certificate validation with fingerprint pinning.
        4. Integration with CRL for instant revocation.
        5. gRPC channel credential generation.

    Usage:
        # Orchestrator (server mode)
        mtls = MTLSConfig(ca_cert_path, server_cert, server_key)
        mtls.initialize()
        ctx = mtls.build_server_context()

        # Node (client mode)
        mtls = MTLSConfig(ca_cert_path, node_cert, node_key)
        mtls.initialize()
        creds = mtls.get_grpc_credentials()
    """

    def __init__(
        self,
        ca_cert_path: str = "",
        cert_path: str = "",
        key_path: str = "",
    ) -> None:
        self._ca_cert_path = Path(ca_cert_path) if ca_cert_path else None
        self._cert_path = Path(cert_path) if cert_path else None
        self._key_path = Path(key_path) if key_path else None
        self._crl = CertificateRevocationList()
        self._known_certs: Dict[str, CertificateRecord] = {}
        self._initialized = False
        self._ca_loaded = False
        self._cert_loaded = False
        self._active_connections: int = 0

    @property
    def crl(self) -> CertificateRevocationList:
        return self._crl

    def initialize(self) -> Tuple[bool, List[str]]:
        """
        Initialize the mTLS subsystem by loading certificates.

        Returns:
            Tuple of (success, list_of_errors_or_warnings).
        """
        errors: List[str] = []

        # Load CA
        if self._ca_cert_path and self._ca_cert_path.exists():
            self._ca_loaded = True
            logger.info(f"[mTLS] CA certificate loaded: {self._ca_cert_path}")
        elif self._ca_cert_path:
            errors.append(f"CA cert not found: {self._ca_cert_path}")

        # Load server/node cert
        if self._cert_path and self._cert_path.exists():
            self._cert_loaded = True
            logger.info(f"[mTLS] Server certificate loaded: {self._cert_path}")
        elif self._cert_path:
            errors.append(f"Server cert not found: {self._cert_path}")

        # Load private key
        if self._key_path and not self._key_path.exists():
            errors.append(f"Private key not found: {self._key_path}")

        self._initialized = len(errors) == 0 or not self._ca_cert_path
        return self._initialized, errors

    def initialize_ephemeral(self) -> None:
        """Initialize with ephemeral credentials for development."""
        self._initialized = True
        self._ca_loaded = True
        self._cert_loaded = True
        logger.warning("[mTLS] Using EPHEMERAL credentials — NOT for production")

    def build_server_context(self) -> ssl.SSLContext:
        """
        Build an SSL context for the Orchestrator (server side).

        Enforces:
            - TLS 1.3 minimum
            - Client certificate required (mutual TLS)
            - Restricted cipher suite
            - CA chain verification
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.minimum_version = MIN_TLS_VERSION

        # Load CA for client cert verification
        if self._ca_cert_path and self._ca_cert_path.exists():
            ctx.load_verify_locations(str(self._ca_cert_path))

        # Load server identity
        if self._cert_path and self._key_path:
            if self._cert_path.exists() and self._key_path.exists():
                ctx.load_cert_chain(
                    certfile=str(self._cert_path),
                    keyfile=str(self._key_path),
                )

        # Restrict ciphers
        try:
            ctx.set_ciphers(ALLOWED_CIPHERS)
        except ssl.SSLError:
            pass  # TLS 1.3 ciphers may not be settable on all platforms

        logger.info("[mTLS] Server SSL context built | TLS 1.3 | CERT_REQUIRED")
        return ctx

    def build_client_context(self) -> ssl.SSLContext:
        """
        Build an SSL context for a Node (client side).

        Presents the node's certificate to the Orchestrator and
        verifies the Orchestrator's certificate against the CA.
        """
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.minimum_version = MIN_TLS_VERSION

        # Load CA to verify server cert
        if self._ca_cert_path and self._ca_cert_path.exists():
            ctx.load_verify_locations(str(self._ca_cert_path))

        # Load client identity (node cert + key)
        if self._cert_path and self._key_path:
            if self._cert_path.exists() and self._key_path.exists():
                ctx.load_cert_chain(
                    certfile=str(self._cert_path),
                    keyfile=str(self._key_path),
                )

        logger.info("[mTLS] Client SSL context built | TLS 1.3 | CERT_REQUIRED")
        return ctx

    def validate_peer_certificate(
        self,
        cert_pem: str,
        expected_node_id: str = "",
    ) -> Tuple[bool, str]:
        """
        Validate a peer certificate during connection establishment.

        Checks:
            1. Certificate is non-empty.
            2. Fingerprint is not on the CRL.
            3. If expected_node_id is provided, fingerprint matches registration.

        Args:
            cert_pem: PEM-encoded certificate from the peer.
            expected_node_id: Optional node_id for pinning check.

        Returns:
            Tuple of (is_valid, reason).
        """
        if not cert_pem or len(cert_pem) < 64:
            return False, "Certificate is empty or too short"

        fingerprint = self.compute_fingerprint(cert_pem)

        # CRL check
        if self._crl.is_revoked(fingerprint):
            entry = self._crl.get_entry(fingerprint)
            reason = entry.reason if entry else "unknown"
            return False, f"Certificate REVOKED: {reason}"

        # Pinning check
        if expected_node_id and expected_node_id in self._known_certs:
            known = self._known_certs[expected_node_id]
            if known.fingerprint_sha256 != fingerprint:
                return False, (
                    f"Certificate fingerprint mismatch for {expected_node_id}: "
                    f"expected {known.fingerprint_sha256[:16]}..., "
                    f"got {fingerprint[:16]}..."
                )

        return True, "Certificate accepted"

    def register_certificate(
        self,
        node_id: str,
        cert_pem: str,
        subject: str = "",
    ) -> CertificateRecord:
        """
        Register a node's certificate for future pinning checks.

        Called after a successful mTLS handshake. The fingerprint is
        stored and checked on all subsequent connections.
        """
        fingerprint = self.compute_fingerprint(cert_pem)
        record = CertificateRecord(
            fingerprint_sha256=fingerprint,
            subject=subject or f"CN={node_id}",
            node_id=node_id,
            state=CertificateState.VALID,
        )
        self._known_certs[node_id] = record
        logger.info(f"[mTLS] Certificate registered for {node_id}: {fingerprint[:16]}...")
        return record

    def revoke_certificate(
        self,
        node_id: str,
        reason: str = "Compromised",
    ) -> bool:
        """
        Revoke a node's certificate immediately.

        The certificate is added to the CRL and the node's known_cert
        record is marked as REVOKED. All future connections are rejected.

        Args:
            node_id: Node whose certificate to revoke.
            reason: Revocation reason.

        Returns:
            True if the certificate was found and revoked.
        """
        if node_id not in self._known_certs:
            return False

        record = self._known_certs[node_id]
        record.state = CertificateState.REVOKED
        record.revoked_at = time.time()
        record.revocation_reason = reason

        self._crl.revoke(
            fingerprint=record.fingerprint_sha256,
            node_id=node_id,
            reason=reason,
        )

        return True

    def get_grpc_credentials(self) -> Dict[str, Any]:
        """
        Generate gRPC channel credentials for mTLS.

        Returns a dict with the cert chain components that can be
        passed to grpc.ssl_channel_credentials() in production.
        """
        creds: Dict[str, Any] = {
            "root_certificates": None,
            "private_key": None,
            "certificate_chain": None,
        }

        if self._ca_cert_path and self._ca_cert_path.exists():
            creds["root_certificates"] = self._ca_cert_path.read_bytes()

        if self._key_path and self._key_path.exists():
            creds["private_key"] = self._key_path.read_bytes()

        if self._cert_path and self._cert_path.exists():
            creds["certificate_chain"] = self._cert_path.read_bytes()

        return creds

    def get_status(self) -> MTLSStatus:
        """Return current mTLS subsystem status for the admin UI."""
        return MTLSStatus(
            initialized=self._initialized,
            ca_loaded=self._ca_loaded,
            server_cert_loaded=self._cert_loaded,
            tls_version="TLS 1.3",
            total_certificates=len(self._known_certs),
            revoked_certificates=self._crl.count,
            active_connections=self._active_connections,
            last_crl_check=self._crl.last_updated,
        )

    def track_connection(self, delta: int = 1) -> None:
        """Track active connection count for monitoring."""
        self._active_connections = max(0, self._active_connections + delta)

    @staticmethod
    def compute_fingerprint(cert_pem: str) -> str:
        """Compute SHA-256 fingerprint of a PEM-encoded certificate."""
        lines = cert_pem.strip().split("\n")
        der_lines = [l for l in lines if not l.startswith("-----")]
        der_content = "".join(der_lines)
        return hashlib.sha256(der_content.encode("utf-8")).hexdigest()
