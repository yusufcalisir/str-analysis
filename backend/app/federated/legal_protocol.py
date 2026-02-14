"""
Legal Handshake Protocol — Post-ZKP Verification Bridge to Law Enforcement.

After a ZKP proof is cryptographically verified, this module generates a
'Legal Request Token' that initiates the formal legal process for requesting
the identity associated with a matched DNA profile.

The token is HMAC-SHA256 signed and contains:
    - query_id:         Unique query identifier
    - node_id:          Node that produced the verified match
    - commitment_hash:  ZKP commitment hash (proof anchor)
    - jurisdiction:     Legal jurisdiction of the source node
    - issued_at:        UTC timestamp of issuance
    - expires_at:       Expiry timestamp (default: 72 hours)

State Machine:
    PENDING_VERIFICATION → VERIFIED → LEGAL_REQUEST_ISSUED
    → IDENTITY_REQUESTED → FULFILLED

Security:
    - Tokens are HMAC-SHA256 signed — tamper-evident
    - Token expiry prevents stale requests
    - State transitions are strictly monotonic (no rollback)
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_TOKEN_TTL_HOURS: int = 72
SIGNING_KEY_ENV: str = "VANTAGE_LEGAL_SIGNING_KEY"


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════════════════════

class LegalHandshakeState(str, Enum):
    """State machine for the post-verification legal process."""
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    LEGAL_REQUEST_ISSUED = "legal_request_issued"
    IDENTITY_REQUESTED = "identity_requested"
    FULFILLED = "fulfilled"


# Valid state transitions (strictly monotonic)
_VALID_TRANSITIONS: Dict[LegalHandshakeState, List[LegalHandshakeState]] = {
    LegalHandshakeState.PENDING_VERIFICATION: [LegalHandshakeState.VERIFIED],
    LegalHandshakeState.VERIFIED: [LegalHandshakeState.LEGAL_REQUEST_ISSUED],
    LegalHandshakeState.LEGAL_REQUEST_ISSUED: [LegalHandshakeState.IDENTITY_REQUESTED],
    LegalHandshakeState.IDENTITY_REQUESTED: [LegalHandshakeState.FULFILLED],
    LegalHandshakeState.FULFILLED: [],  # Terminal state
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class LegalRequestToken(BaseModel):
    """
    Cryptographically signed token authorizing a formal identity request.

    Generated only after a ZKP proof is verified — serves as the bridge
    between the technical cryptographic verification and the legal process.
    """
    token_id: str = Field(default_factory=lambda: secrets.token_hex(16))
    query_id: str
    node_id: str
    commitment_hash: str
    jurisdiction: str = ""
    issued_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    expires_at: str = ""
    signature: str = ""  # HMAC-SHA256 hex digest
    state: LegalHandshakeState = LegalHandshakeState.VERIFIED

    @property
    def is_expired(self) -> bool:
        """Check if the token has expired."""
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) > exp
        except (ValueError, TypeError):
            return True

    @property
    def payload_for_signing(self) -> str:
        """Canonical payload string for HMAC computation."""
        return (
            f"{self.token_id}:{self.query_id}:{self.node_id}:"
            f"{self.commitment_hash}:{self.jurisdiction}:"
            f"{self.issued_at}:{self.expires_at}"
        )


class LegalCaseRecord(BaseModel):
    """Tracks the full lifecycle of a legal request."""
    token: LegalRequestToken
    state_history: List[Dict] = Field(default_factory=list)
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOL MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class LegalProtocolManager:
    """
    Manages the legal handshake protocol lifecycle.

    Responsibilities:
        - Generate HMAC-signed Legal Request Tokens after ZKP verification.
        - Validate token signatures and expiry.
        - Track and enforce state transitions.

    Thread Safety:
        Uses in-memory dict. For production, replace with a persistent
        audit-logged database.
    """

    def __init__(
        self,
        signing_key: Optional[str] = None,
        token_ttl_hours: int = DEFAULT_TOKEN_TTL_HOURS,
    ) -> None:
        """
        Args:
            signing_key: HMAC-SHA256 signing key. If None, generates a
                         random ephemeral key (suitable for dev only).
            token_ttl_hours: Token time-to-live in hours.
        """
        self._signing_key = (
            signing_key or secrets.token_hex(32)
        ).encode("utf-8")
        self._token_ttl_hours = token_ttl_hours
        self._cases: Dict[str, LegalCaseRecord] = {}  # token_id → case

    @property
    def active_cases(self) -> int:
        """Number of tracked legal cases."""
        return len(self._cases)

    def generate_token(
        self,
        query_id: str,
        node_id: str,
        commitment_hash: str,
        jurisdiction: str = "",
    ) -> LegalRequestToken:
        """
        Generate a signed Legal Request Token.

        This should ONLY be called after a ZKP proof has been cryptographically
        verified. The token serves as formal authorization to request the
        identity behind the DNA match via legal channels.

        Args:
            query_id: Unique query identifier (ties to the broadcast).
            node_id: Node that produced the verified match.
            commitment_hash: ZKP commitment hash (proof anchor).
            jurisdiction: Legal jurisdiction (e.g. "TR", "DE", "US").

        Returns:
            Signed LegalRequestToken.
        """
        from datetime import timedelta

        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=self._token_ttl_hours)

        token = LegalRequestToken(
            query_id=query_id,
            node_id=node_id,
            commitment_hash=commitment_hash,
            jurisdiction=jurisdiction,
            issued_at=now.isoformat(),
            expires_at=expires.isoformat(),
            state=LegalHandshakeState.VERIFIED,
        )

        # Sign the token
        signature = self._compute_signature(token.payload_for_signing)
        token = token.model_copy(update={"signature": signature})

        # Create case record
        case = LegalCaseRecord(
            token=token,
            state_history=[{
                "state": LegalHandshakeState.VERIFIED.value,
                "timestamp": now.isoformat(),
                "actor": "orchestrator",
            }],
        )
        self._cases[token.token_id] = case

        logger.info(
            f"[LEGAL] Token {token.token_id[:12]}... issued — "
            f"query={query_id} node={node_id} "
            f"jurisdiction={jurisdiction or 'unspecified'} "
            f"expires={expires.isoformat()}"
        )

        return token

    def validate_token(self, token: LegalRequestToken) -> bool:
        """
        Validate a token's signature and expiry.

        Args:
            token: Token to validate.

        Returns:
            True if signature matches and token is not expired.
        """
        if token.is_expired:
            logger.warning(f"[LEGAL] Token {token.token_id[:12]}... EXPIRED")
            return False

        expected_sig = self._compute_signature(token.payload_for_signing)
        if not hmac.compare_digest(token.signature, expected_sig):
            logger.warning(f"[LEGAL] Token {token.token_id[:12]}... INVALID SIGNATURE")
            return False

        return True

    def transition_state(
        self,
        token_id: str,
        new_state: LegalHandshakeState,
        actor: str = "system",
    ) -> bool:
        """
        Advance a legal case to the next state.

        Enforces strictly monotonic transitions per the state machine.

        Args:
            token_id: Token identifier.
            new_state: Target state.
            actor: Who/what initiated the transition.

        Returns:
            True if transition succeeded, False if invalid.
        """
        if token_id not in self._cases:
            logger.warning(f"[LEGAL] Unknown token: {token_id[:12]}...")
            return False

        case = self._cases[token_id]
        current_state = case.token.state

        # Check valid transitions
        valid_next = _VALID_TRANSITIONS.get(current_state, [])
        if new_state not in valid_next:
            logger.warning(
                f"[LEGAL] Invalid transition {current_state.value} → "
                f"{new_state.value} for token {token_id[:12]}..."
            )
            return False

        # Apply transition
        now = datetime.now(timezone.utc).isoformat()
        case.token = case.token.model_copy(update={"state": new_state})
        case.state_history.append({
            "state": new_state.value,
            "timestamp": now,
            "actor": actor,
        })
        case.updated_at = now

        logger.info(
            f"[LEGAL] Token {token_id[:12]}... → {new_state.value} "
            f"(by {actor})"
        )
        return True

    def get_case(self, token_id: str) -> Optional[LegalCaseRecord]:
        """Retrieve a legal case record by token ID."""
        return self._cases.get(token_id)

    def get_cases_for_query(self, query_id: str) -> List[LegalCaseRecord]:
        """Retrieve all legal cases associated with a query."""
        return [
            c for c in self._cases.values()
            if c.token.query_id == query_id
        ]

    def _compute_signature(self, payload: str) -> str:
        """Compute HMAC-SHA256 signature for a payload string."""
        return hmac.new(
            self._signing_key,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
