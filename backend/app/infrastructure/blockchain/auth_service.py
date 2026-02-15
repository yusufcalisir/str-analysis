"""
Auth Service — Forensic Session Management.

Handles the generation and validation of short-lived, HMAC-signed session tokens
that bind an investigator's Ethereum address to a temporary access window.
This ensures that the blockchain handshake happens once per session, not per request,
reducing latency while maintaining security.
"""

import base64
import hashlib
import hmac
import json
import logging
import time
from typing import Dict, Optional

from app.core.config import settings

logger = logging.getLogger(__name__)


class AuthError(Exception):
    """Base class for authentication failures."""
    pass


class TokenExpired(AuthError):
    """Raised when the session token TTL has elapsed."""
    pass


class TokenInvalid(AuthError):
    """Raised when the token signature or format is bad."""
    pass


def generate_session_token(investigator_address: str) -> Dict[str, str]:
    """
    Create a signed session token for an authorized investigator.

    Payload:
        - sub: investigator_address
        - iat: issued_at timestamp (int seconds)
        - exp: expires_at timestamp (int seconds)

    Returns:
        Dict containing 'token' and 'expires_at' (ISO string).
    """
    now = int(time.time())
    ttl_seconds = settings.SESSION_TOKEN_TTL_MINUTES * 60
    exp = now + ttl_seconds

    payload = {
        "sub": investigator_address,
        "iat": now,
        "exp": exp,
    }

    # 1. Serialize & Encode Payload
    json_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = base64.urlsafe_b64encode(json_bytes).decode("utf-8").rstrip("=")

    # 2. Sign (HMAC-SHA256)
    signature = _sign(payload_b64)

    # 3. Construct Token
    token = f"{payload_b64}.{signature}"

    return {
        "token": token,
        "expires_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp)),
        "investigator": investigator_address
    }


def validate_session_token(token: str) -> str:
    """
    Validate a session token and return the investigator address (sub).

    Checks:
        - Format (payload.sig)
        - Signature integrity
        - Expiration

    Returns:
        investigator_address (str)

    Raises:
        TokenInvalid: if format or signature is bad.
        TokenExpired: if exp < now.
    """
    if not token or "." not in token:
        raise TokenInvalid("Invalid token format")

    try:
        payload_b64, provided_sig = token.rsplit(".", 1)
    except ValueError:
        raise TokenInvalid("Malformed token")

    # 1. Verify Signature
    expected_sig = _sign(payload_b64)
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise TokenInvalid("Invalid signature")

    # 2. Decode Payload
    try:
        # Pad base64 if needed (urlsafe_b64decode requires padding)
        padding = "=" * (-len(payload_b64) % 4)
        json_bytes = base64.urlsafe_b64decode(payload_b64 + padding)
        payload = json.loads(json_bytes)
    except Exception as e:
        raise TokenInvalid(f"Corrupt payload: {e}")

    # 3. Check Expiry
    exp = payload.get("exp", 0)
    if time.time() > exp:
        raise TokenExpired("Session token has expired")

    return payload.get("sub", "")


def _sign(data: str) -> str:
    """Compute HMAC-SHA256 signature for a string using settings.SECRET_KEY."""
    key = settings.SECRET_KEY.encode("utf-8")
    msg = data.encode("utf-8")
    sig_bytes = hmac.new(key, msg, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig_bytes).decode("utf-8").rstrip("=")
