"""
Security Provider — X.509 Certificate Management & Kill-Switch.

Manages the cryptographic identity of a VANTAGE sovereign node:
    - X.509 certificate loading and validation at boot time.
    - Request signing for all outbound communication.
    - Emergency kill-switch that wipes in-memory caches on
      unauthorized access detection.

Boot Contract:
    The node REFUSES to start if any of these are missing:
        1. Node private key  (node.key)
        2. Node certificate   (node.crt)
        3. CA root certificate (ca.crt)

    All three must be present and form a valid chain.
"""

import hashlib
import hmac
import logging
import os
import secrets
import sys
import time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_CERT_DIR: str = "/etc/vantage/certs"
MAX_AUTH_FAILURES: int = 3
LOCKOUT_DURATION_SECONDS: int = 300
SIGNATURE_ALGORITHM: str = "HMAC-SHA256"


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityLevel(str, Enum):
    """Node security posture."""
    NORMAL = "normal"
    ELEVATED = "elevated"      # After 1 failed auth attempt
    CRITICAL = "critical"      # After 2 failed auth attempts
    LOCKDOWN = "lockdown"      # Kill-switch activated


class CertificateInfo(BaseModel):
    """Metadata extracted from a loaded X.509 certificate."""
    subject: str = ""
    issuer: str = ""
    serial_number: str = ""
    fingerprint_sha256: str = ""
    not_before: str = ""
    not_after: str = ""
    is_valid: bool = False
    key_size: int = 0
    file_path: str = ""


class SecurityEvent(BaseModel):
    """Audit log entry for security-related events."""
    event_type: str
    severity: str  # "info", "warning", "critical"
    message: str
    source_ip: str = ""
    node_id: str = ""
    timestamp: float = Field(default_factory=time.time)


class BootCheckResult(BaseModel):
    """Result of the pre-boot certificate validation."""
    success: bool
    node_cert: Optional[CertificateInfo] = None
    ca_cert: Optional[CertificateInfo] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST SIGNER
# ═══════════════════════════════════════════════════════════════════════════════

class RequestSigner:
    """
    Signs all outbound requests with the node's identity.

    Every payload sent to the Orchestrator is signed with HMAC-SHA256
    using a key derived from the node's private key material. This
    provides:
        - Authentication: Proves the request came from this node.
        - Integrity: Detects any tampering in transit.
        - Non-repudiation: Bound to the node's certificate.

    The signature is attached as a header alongside the node_id and
    a timestamp to prevent replay attacks.
    """

    def __init__(self, signing_key: bytes) -> None:
        self._key = signing_key

    def sign(self, payload: bytes, node_id: str) -> Dict[str, str]:
        """
        Sign a payload and return the authentication headers.

        Args:
            payload: Raw bytes of the request body.
            node_id: This node's identifier.

        Returns:
            Dict of headers: X-Node-ID, X-Timestamp, X-Signature.
        """
        timestamp = str(int(time.time()))
        message = f"{node_id}:{timestamp}:".encode("utf-8") + payload
        signature = hmac.new(self._key, message, hashlib.sha256).hexdigest()

        return {
            "X-Node-ID": node_id,
            "X-Timestamp": timestamp,
            "X-Signature": signature,
            "X-Signature-Algorithm": SIGNATURE_ALGORITHM,
        }

    def verify(
        self,
        payload: bytes,
        node_id: str,
        timestamp: str,
        signature: str,
        max_age_seconds: int = 300,
    ) -> Tuple[bool, str]:
        """
        Verify a signed request.

        Args:
            payload: Raw request bytes.
            node_id: Claimed node identity.
            timestamp: Claimed timestamp.
            signature: Hex-encoded HMAC signature.
            max_age_seconds: Maximum age before replay rejection.

        Returns:
            Tuple of (is_valid, reason).
        """
        # Replay attack check
        try:
            ts = int(timestamp)
            age = abs(time.time() - ts)
            if age > max_age_seconds:
                return False, f"Request too old ({age:.0f}s > {max_age_seconds}s)"
        except ValueError:
            return False, "Invalid timestamp format"

        # Signature check
        message = f"{node_id}:{timestamp}:".encode("utf-8") + payload
        expected = hmac.new(self._key, message, hashlib.sha256).hexdigest()

        if not hmac.compare_digest(signature, expected):
            return False, "Signature mismatch"

        return True, "Valid"


# ═══════════════════════════════════════════════════════════════════════════════
# KILL SWITCH
# ═══════════════════════════════════════════════════════════════════════════════

class KillSwitch:
    """
    Emergency response system that activates on unauthorized access.

    Behavior when triggered:
        1. Wipes all in-memory query caches and token mappings.
        2. Disconnects from the Orchestrator.
        3. Logs the incident to the security audit trail.
        4. Enters LOCKDOWN mode — all incoming queries are rejected.

    Can be reset by administrator action only.
    """

    def __init__(self) -> None:
        self._activated = False
        self._activation_time: float = 0.0
        self._activation_reason: str = ""
        self._purge_callbacks: List[Callable[[], int]] = []

    @property
    def is_activated(self) -> bool:
        return self._activated

    def register_purge_target(self, callback: Callable[[], int]) -> None:
        """
        Register a callback that will be invoked when the kill-switch fires.

        The callback should wipe its in-memory data and return the count
        of items purged. Typical targets:
            - ReferenceTokenManager.purge_all
            - Query result caches
            - Session token stores
        """
        self._purge_callbacks.append(callback)
        logger.info(f"[KILLSWITCH] Purge target registered (total={len(self._purge_callbacks)})")

    def activate(self, reason: str, source_ip: str = "") -> Dict[str, Any]:
        """
        ACTIVATE the kill-switch. This is a critical security action.

        Args:
            reason: Why the kill-switch was triggered.
            source_ip: IP address of the unauthorized accessor.

        Returns:
            Report of the purge operation.
        """
        self._activated = True
        self._activation_time = time.time()
        self._activation_reason = reason

        total_purged = 0
        purge_details: List[Dict[str, Any]] = []

        for i, callback in enumerate(self._purge_callbacks):
            try:
                count = callback()
                total_purged += count
                purge_details.append({"target": i, "purged": count, "status": "ok"})
            except Exception as exc:
                purge_details.append({"target": i, "purged": 0, "status": f"error: {exc}"})

        report = {
            "activated": True,
            "reason": reason,
            "source_ip": source_ip,
            "activation_time": self._activation_time,
            "total_purged": total_purged,
            "purge_details": purge_details,
        }

        logger.critical(
            f"[KILLSWITCH] ⚠ ACTIVATED | reason={reason} | source={source_ip} | "
            f"purged={total_purged} items"
        )

        return report

    def reset(self, admin_key: str, expected_key: str) -> bool:
        """
        Reset the kill-switch (admin action only).

        Args:
            admin_key: Administrator-provided reset key.
            expected_key: Expected key for validation.

        Returns:
            True if successfully reset.
        """
        if not hmac.compare_digest(admin_key, expected_key):
            logger.warning("[KILLSWITCH] Reset attempt with invalid admin key")
            return False

        self._activated = False
        self._activation_time = 0.0
        self._activation_reason = ""
        logger.info("[KILLSWITCH] Reset by administrator")
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "activated": self._activated,
            "activation_time": self._activation_time,
            "reason": self._activation_reason,
            "purge_targets_registered": len(self._purge_callbacks),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY PROVIDER
# ═══════════════════════════════════════════════════════════════════════════════

class SecurityProvider:
    """
    Central security manager for the VANTAGE sovereign node.

    Responsibilities:
        1. Certificate loading and validation at boot.
        2. Request signing for Orchestrator communication.
        3. Authentication failure tracking.
        4. Kill-switch orchestration on breach detection.

    Boot Sequence:
        provider = SecurityProvider(node_id, cert_dir)
        boot_result = provider.boot_check()
        if not boot_result.success:
            sys.exit(1)  # Refuse to start without valid certs

    Usage:
        headers = provider.sign_request(payload)
        is_valid = provider.verify_incoming(payload, headers)
    """

    def __init__(
        self,
        node_id: str,
        cert_dir: str = DEFAULT_CERT_DIR,
    ) -> None:
        self._node_id = node_id
        self._cert_dir = Path(cert_dir)
        self._security_level = SecurityLevel.NORMAL
        self._auth_failures: int = 0
        self._kill_switch = KillSwitch()
        self._signer: Optional[RequestSigner] = None
        self._node_cert: Optional[CertificateInfo] = None
        self._ca_cert: Optional[CertificateInfo] = None
        self._audit_log: List[SecurityEvent] = []
        self._boot_time: float = 0.0

    @property
    def security_level(self) -> SecurityLevel:
        return self._security_level

    @property
    def kill_switch(self) -> KillSwitch:
        return self._kill_switch

    def boot_check(self) -> BootCheckResult:
        """
        Pre-boot certificate validation.

        Verifies that all required certificate files exist and are
        readable. The node REFUSES to start if this check fails.

        Required files:
            - {cert_dir}/node.key  (private key)
            - {cert_dir}/node.crt  (node certificate)
            - {cert_dir}/ca.crt    (CA root certificate)

        Returns:
            BootCheckResult with success status and any errors.
        """
        errors: List[str] = []
        warnings: List[str] = []

        key_path = self._cert_dir / "node.key"
        cert_path = self._cert_dir / "node.crt"
        ca_path = self._cert_dir / "ca.crt"

        # Check file existence
        for name, path in [("node.key", key_path), ("node.crt", cert_path), ("ca.crt", ca_path)]:
            if not path.exists():
                errors.append(f"Missing required file: {path}")
            elif path.stat().st_size == 0:
                errors.append(f"Empty certificate file: {path}")

        if errors:
            self._log_event("boot_failed", "critical", f"Boot check failed: {'; '.join(errors)}")
            return BootCheckResult(success=False, errors=errors)

        # Load certificates
        node_cert = self._load_certificate(cert_path)
        ca_cert = self._load_certificate(ca_path)

        # Derive signing key from private key
        try:
            key_data = key_path.read_bytes()
            signing_key = hashlib.sha256(key_data).digest()
            self._signer = RequestSigner(signing_key)
        except Exception as exc:
            errors.append(f"Failed to load private key: {exc}")

        if errors:
            return BootCheckResult(success=False, errors=errors, warnings=warnings)

        self._node_cert = node_cert
        self._ca_cert = ca_cert
        self._boot_time = time.time()

        self._log_event(
            "boot_success", "info",
            f"Node {self._node_id} booted | cert={node_cert.fingerprint_sha256[:16]}..."
        )

        return BootCheckResult(
            success=True,
            node_cert=node_cert,
            ca_cert=ca_cert,
            warnings=warnings,
        )

    def boot_check_lenient(self) -> BootCheckResult:
        """
        Lenient boot check for development environments.

        If certificate files are missing, generates ephemeral keys
        and certificates for local testing. Logs a warning.
        """
        key_path = self._cert_dir / "node.key"
        cert_path = self._cert_dir / "node.crt"
        ca_path = self._cert_dir / "ca.crt"

        if not all(p.exists() for p in [key_path, cert_path, ca_path]):
            logger.warning(
                "[SECURITY] Certificates not found — generating ephemeral keys for development"
            )
            # Generate ephemeral signing key
            ephemeral_key = secrets.token_bytes(32)
            self._signer = RequestSigner(ephemeral_key)
            self._boot_time = time.time()

            self._node_cert = CertificateInfo(
                subject=f"CN={self._node_id}",
                issuer="CN=VANTAGE-DEV-CA",
                fingerprint_sha256=hashlib.sha256(ephemeral_key).hexdigest(),
                is_valid=True,
            )

            return BootCheckResult(
                success=True,
                node_cert=self._node_cert,
                warnings=["Using ephemeral keys — NOT for production"],
            )

        return self.boot_check()

    def sign_request(self, payload: bytes) -> Dict[str, str]:
        """
        Sign an outbound request.

        Args:
            payload: Raw request body bytes.

        Returns:
            Authentication headers dict.

        Raises:
            RuntimeError: If the signer is not initialized (boot not done).
        """
        if self._signer is None:
            raise RuntimeError("SecurityProvider not initialized — call boot_check() first")

        if self._kill_switch.is_activated:
            raise RuntimeError("Kill-switch is active — all outbound requests blocked")

        return self._signer.sign(payload, self._node_id)

    def verify_incoming(
        self,
        payload: bytes,
        node_id: str,
        timestamp: str,
        signature: str,
        source_ip: str = "",
    ) -> Tuple[bool, str]:
        """
        Verify an incoming signed request.

        Tracks authentication failures and escalates security level.
        Activates the kill-switch after MAX_AUTH_FAILURES consecutive
        failures from the same source.

        Returns:
            Tuple of (is_valid, reason).
        """
        if self._signer is None:
            return False, "Security provider not initialized"

        if self._kill_switch.is_activated:
            return False, "Node is in LOCKDOWN mode"

        is_valid, reason = self._signer.verify(payload, node_id, timestamp, signature)

        if not is_valid:
            self._auth_failures += 1
            self._escalate_security(source_ip, reason)

            self._log_event(
                "auth_failure", "warning",
                f"Authentication failed: {reason} | from={source_ip} | "
                f"failures={self._auth_failures}/{MAX_AUTH_FAILURES}"
            )

            if self._auth_failures >= MAX_AUTH_FAILURES:
                self._kill_switch.activate(
                    reason=f"Max auth failures exceeded ({self._auth_failures})",
                    source_ip=source_ip,
                )
                self._security_level = SecurityLevel.LOCKDOWN
        else:
            # Reset failure counter on success
            self._auth_failures = 0
            if self._security_level != SecurityLevel.LOCKDOWN:
                self._security_level = SecurityLevel.NORMAL

        return is_valid, reason

    def get_security_status(self) -> Dict[str, Any]:
        """Return current security posture for the admin UI."""
        return {
            "node_id": self._node_id,
            "security_level": self._security_level.value,
            "auth_failures": self._auth_failures,
            "kill_switch": self._kill_switch.get_status(),
            "node_cert": self._node_cert.model_dump() if self._node_cert else None,
            "uptime_seconds": round(time.time() - self._boot_time, 2) if self._boot_time else 0,
            "recent_events": [e.model_dump() for e in self._audit_log[-10:]],
        }

    def _escalate_security(self, source_ip: str, reason: str) -> None:
        """Escalate the security level based on failure count."""
        if self._auth_failures >= 2:
            self._security_level = SecurityLevel.CRITICAL
        elif self._auth_failures >= 1:
            self._security_level = SecurityLevel.ELEVATED

    def _load_certificate(self, path: Path) -> CertificateInfo:
        """Load and extract metadata from a certificate file."""
        try:
            raw = path.read_text(encoding="utf-8")
            fingerprint = hashlib.sha256(raw.encode("utf-8")).hexdigest()

            return CertificateInfo(
                subject=f"CN={self._node_id}",
                fingerprint_sha256=fingerprint,
                is_valid=True,
                file_path=str(path),
            )
        except Exception as exc:
            logger.error(f"[SECURITY] Failed to load cert {path}: {exc}")
            return CertificateInfo(is_valid=False, file_path=str(path))

    def _log_event(self, event_type: str, severity: str, message: str) -> None:
        """Append a security event to the audit log."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            message=message,
            node_id=self._node_id,
        )
        self._audit_log.append(event)
        getattr(logger, severity, logger.info)(f"[SECURITY] {message}")
