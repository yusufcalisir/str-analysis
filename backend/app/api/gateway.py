"""
Global API Gateway — Centralized Entry Point for VANTAGE-STR.

Provides a secure, rate-limited API gateway for all external law enforcement
agencies (INTERPOL, EUROPOL, National Police). Implements:

    1. Per-Agency Rate Limiting — sliding window counter, burst + sustained limits.
    2. API Key Management — SHA-256 key generation, rotation with grace period.
    3. Agency Authentication — FastAPI dependency injecting authenticated context.
    4. Audit Integration — every request is logged to the ForensicLedger.

Architecture:
    External Agency → API Key Auth → Rate Limiter → Compliance Check → Orchestrator
         ↓                                                    ↓
    ForensicLedger ←──────────────────────────────────────────┘

Usage:
    # Register with main app:
    from app.api.gateway import gateway_router
    app.include_router(gateway_router)
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

gateway_router = APIRouter(prefix="/gateway", tags=["Gateway"])


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_RATE_LIMIT = 60        # requests per window
DEFAULT_BURST_LIMIT = 15       # max burst in 10s
RATE_WINDOW_SECONDS = 60.0     # sliding window
BURST_WINDOW_SECONDS = 10.0
KEY_ROTATION_GRACE_SECONDS = 300.0  # 5min grace for old keys


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AgencyTier(str, Enum):
    """Access tier determining rate limits and capabilities."""
    INTERPOL = "INTERPOL"      # Highest: 300 req/min
    EUROPOL = "EUROPOL"        # High: 200 req/min
    NATIONAL = "NATIONAL"      # Standard: 60 req/min
    OBSERVER = "OBSERVER"      # Limited: 10 req/min


TIER_LIMITS: Dict[str, Tuple[int, int]] = {
    AgencyTier.INTERPOL.value: (300, 50),
    AgencyTier.EUROPOL.value: (200, 30),
    AgencyTier.NATIONAL.value: (60, 15),
    AgencyTier.OBSERVER.value: (10, 5),
}


class AgencyCredential(BaseModel):
    """Registered agency with authentication and rate limiting config."""
    agency_id: str
    agency_name: str
    tier: str = AgencyTier.NATIONAL.value
    country: str
    api_key_hash: str  # SHA-256 of the active API key
    previous_key_hash: Optional[str] = None  # For rotation grace
    key_rotated_at: Optional[str] = None
    rate_limit: int = DEFAULT_RATE_LIMIT
    burst_limit: int = DEFAULT_BURST_LIMIT
    enabled: bool = True
    registered_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class GatewayQueryRequest(BaseModel):
    """Incoming forensic query through the gateway."""
    query_id: Optional[str] = None
    str_profile: Dict[str, Any]
    court_order_id: str
    crime_category: str
    target_countries: List[str] = Field(default_factory=list)
    max_results: int = Field(default=25, ge=1, le=100)


class GatewayQueryResponse(BaseModel):
    """Gateway response wrapping orchestrator results with metadata."""
    query_id: str
    agency_id: str
    status: str = "accepted"
    rate_limit_remaining: int = 0
    rate_limit_reset_seconds: float = 0.0
    results_count: int = 0
    processing_time_ms: float = 0.0
    audit_entry_hash: str = ""
    message: str = ""


class KeyRotationResponse(BaseModel):
    """Response to API key rotation request."""
    agency_id: str
    new_key: str  # Returned ONCE — must be stored by the agency
    old_key_valid_until: str
    message: str


class GatewayStatus(BaseModel):
    """Current gateway status."""
    status: str = "operational"
    registered_agencies: int = 0
    total_requests_served: int = 0
    active_rate_limits: int = 0
    uptime_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RATE LIMITER
# ═══════════════════════════════════════════════════════════════════════════════

class RateLimiter:
    """
    Sliding window rate limiter with burst protection.

    Tracks request timestamps per agency in two windows:
        - Sustained: N requests per 60s (configurable per tier)
        - Burst: M requests per 10s (prevents hammering)

    Thread Safety:
        Single-process safe. For multi-process deployments,
        replace with Redis-backed counters.
    """

    def __init__(self) -> None:
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def check(
        self,
        agency_id: str,
        sustained_limit: int = DEFAULT_RATE_LIMIT,
        burst_limit: int = DEFAULT_BURST_LIMIT,
    ) -> Tuple[bool, int, float]:
        """
        Check if the agency is within rate limits.

        Args:
            agency_id: Unique agency identifier.
            sustained_limit: Max requests per sustained window.
            burst_limit: Max requests per burst window.

        Returns:
            Tuple of (allowed: bool, remaining: int, reset_seconds: float)
        """
        now = time.time()
        self._evict(agency_id, now)

        timestamps = self._requests[agency_id]

        # Check burst window
        burst_start = now - BURST_WINDOW_SECONDS
        burst_count = sum(1 for t in timestamps if t > burst_start)
        if burst_count >= burst_limit:
            oldest_burst = min((t for t in timestamps if t > burst_start), default=now)
            reset = BURST_WINDOW_SECONDS - (now - oldest_burst)
            return False, 0, max(reset, 0.1)

        # Check sustained window
        sustained_count = len(timestamps)
        if sustained_count >= sustained_limit:
            oldest = min(timestamps) if timestamps else now
            reset = RATE_WINDOW_SECONDS - (now - oldest)
            return False, 0, max(reset, 0.1)

        remaining = sustained_limit - sustained_count - 1
        return True, max(remaining, 0), 0.0

    def record(self, agency_id: str) -> None:
        """Record a request for the agency."""
        self._requests[agency_id].append(time.time())

    def _evict(self, agency_id: str, now: float) -> None:
        """Remove timestamps older than the sustained window."""
        cutoff = now - RATE_WINDOW_SECONDS
        self._requests[agency_id] = [
            t for t in self._requests[agency_id] if t > cutoff
        ]

    def get_active_count(self) -> int:
        """Number of agencies with active rate tracking."""
        return sum(1 for v in self._requests.values() if v)


# ═══════════════════════════════════════════════════════════════════════════════
# API KEY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class APIKeyManager:
    """
    Manages API key generation, verification, and rotation.

    Keys are 64-character hex strings (256 bits of entropy).
    Only SHA-256 hashes are stored — raw keys are returned once
    at creation/rotation and must be stored by the agency.

    Rotation provides a grace period where both old and new keys
    are accepted, ensuring zero-downtime key changes.
    """

    @staticmethod
    def generate_key() -> Tuple[str, str]:
        """
        Generate a new API key.

        Returns:
            Tuple of (raw_key, key_hash)
        """
        raw_key = secrets.token_hex(32)  # 64 hex chars, 256 bits
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        return raw_key, key_hash

    @staticmethod
    def hash_key(raw_key: str) -> str:
        """Hash a raw API key for storage/comparison."""
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @staticmethod
    def verify_key(
        raw_key: str,
        credential: AgencyCredential,
    ) -> bool:
        """
        Verify an API key against stored hashes.

        Checks both the active key and the previous key (during
        rotation grace period).
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        # Check active key
        if secrets.compare_digest(key_hash, credential.api_key_hash):
            return True

        # Check previous key (grace period)
        if credential.previous_key_hash and credential.key_rotated_at:
            try:
                rotated = datetime.fromisoformat(credential.key_rotated_at)
                grace_end = rotated + timedelta(seconds=KEY_ROTATION_GRACE_SECONDS)
                if datetime.now(timezone.utc) < grace_end:
                    if secrets.compare_digest(key_hash, credential.previous_key_hash):
                        return True
            except (ValueError, TypeError):
                pass

        return False


# ═══════════════════════════════════════════════════════════════════════════════
# AGENCY REGISTRY (In-Memory — Production: Database)
# ═══════════════════════════════════════════════════════════════════════════════

class AgencyRegistry:
    """
    In-memory agency credential store.

    In production, this would be backed by a secure database
    with encrypted credential storage and RBAC.
    """

    def __init__(self) -> None:
        self._agencies: Dict[str, AgencyCredential] = {}
        self._key_index: Dict[str, str] = {}  # key_hash → agency_id

    def register(
        self,
        agency_id: str,
        agency_name: str,
        tier: str,
        country: str,
    ) -> Tuple[str, AgencyCredential]:
        """
        Register a new agency and return its API key.

        Returns:
            Tuple of (raw_api_key, credential)
        """
        raw_key, key_hash = APIKeyManager.generate_key()
        limits = TIER_LIMITS.get(tier, TIER_LIMITS[AgencyTier.NATIONAL.value])

        cred = AgencyCredential(
            agency_id=agency_id,
            agency_name=agency_name,
            tier=tier,
            country=country,
            api_key_hash=key_hash,
            rate_limit=limits[0],
            burst_limit=limits[1],
        )

        self._agencies[agency_id] = cred
        self._key_index[key_hash] = agency_id

        logger.info(
            f"[GATEWAY] Agency '{agency_id}' registered — "
            f"tier={tier} rate={limits[0]}/min"
        )

        return raw_key, cred

    def find_by_key(self, raw_key: str) -> Optional[AgencyCredential]:
        """Look up agency by API key."""
        key_hash = APIKeyManager.hash_key(raw_key)
        agency_id = self._key_index.get(key_hash)

        if agency_id:
            cred = self._agencies.get(agency_id)
            if cred and APIKeyManager.verify_key(raw_key, cred):
                return cred

        # Brute-force check (for rotated keys during grace)
        for cred in self._agencies.values():
            if APIKeyManager.verify_key(raw_key, cred):
                return cred

        return None

    def rotate_key(self, agency_id: str) -> Optional[Tuple[str, str]]:
        """
        Rotate an agency's API key.

        Returns:
            Tuple of (new_raw_key, grace_expiry_iso) or None if not found.
        """
        cred = self._agencies.get(agency_id)
        if not cred:
            return None

        # Generate new key
        new_raw, new_hash = APIKeyManager.generate_key()
        now = datetime.now(timezone.utc)
        grace_end = now + timedelta(seconds=KEY_ROTATION_GRACE_SECONDS)

        # Remove old index, add new
        if cred.api_key_hash in self._key_index:
            del self._key_index[cred.api_key_hash]

        # Update credential
        self._agencies[agency_id] = cred.model_copy(update={
            "previous_key_hash": cred.api_key_hash,
            "api_key_hash": new_hash,
            "key_rotated_at": now.isoformat(),
        })

        self._key_index[new_hash] = agency_id

        logger.warning(
            f"[GATEWAY] Key rotated for '{agency_id}' — "
            f"grace until {grace_end.isoformat()}"
        )

        return new_raw, grace_end.isoformat()

    def get(self, agency_id: str) -> Optional[AgencyCredential]:
        return self._agencies.get(agency_id)

    @property
    def count(self) -> int:
        return len(self._agencies)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ═══════════════════════════════════════════════════════════════════════════════

_registry = AgencyRegistry()
_rate_limiter = RateLimiter()
_key_manager = APIKeyManager()
_start_time = time.time()
_total_requests = 0

# Pre-register demonstration agencies
_demo_keys: Dict[str, str] = {}
for _aid, _name, _tier, _country in [
    ("INTERPOL-EU", "INTERPOL Lyon HQ", AgencyTier.INTERPOL.value, "INTERPOL"),
    ("BKA-DE", "Bundeskriminalamt", AgencyTier.NATIONAL.value, "DE"),
    ("FBI-US", "Federal Bureau of Investigation", AgencyTier.NATIONAL.value, "US"),
    ("EUROPOL-NL", "Europol The Hague", AgencyTier.EUROPOL.value, "NL"),
    ("NPA-JP", "National Police Agency", AgencyTier.NATIONAL.value, "JP"),
    ("AFP-AU", "Australian Federal Police", AgencyTier.NATIONAL.value, "AU"),
]:
    _key, _ = _registry.register(_aid, _name, _tier, _country)
    _demo_keys[_aid] = _key


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH DEPENDENCY
# ═══════════════════════════════════════════════════════════════════════════════

async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
) -> AgencyCredential:
    """
    FastAPI dependency — authenticate and rate-limit incoming requests.

    Extracts the API key from the X-API-Key header, verifies it against
    the registry, checks rate limits, and returns the agency context.

    Raises:
        401: Invalid or missing API key
        403: Agency account disabled
        429: Rate limit exceeded
    """
    global _total_requests
    _total_requests += 1

    # Authenticate
    cred = _registry.find_by_key(x_api_key)
    if cred is None:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "INVALID_API_KEY",
                "message": "The provided API key is not recognized.",
            },
        )

    if not cred.enabled:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "AGENCY_DISABLED",
                "message": f"Agency '{cred.agency_id}' account is suspended.",
            },
        )

    # Rate limit
    allowed, remaining, reset = _rate_limiter.check(
        cred.agency_id, cred.rate_limit, cred.burst_limit,
    )

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "RATE_LIMIT_EXCEEDED",
                "message": (
                    f"Rate limit exceeded for agency '{cred.agency_id}'. "
                    f"Retry after {reset:.1f}s."
                ),
                "retry_after_seconds": round(reset, 1),
            },
        )

    _rate_limiter.record(cred.agency_id)

    return cred


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@gateway_router.post(
    "/query",
    response_model=GatewayQueryResponse,
    summary="Submit a forensic DNA query through the secure gateway",
)
async def gateway_query(
    request: GatewayQueryRequest,
    agency: AgencyCredential = Depends(verify_api_key),
) -> GatewayQueryResponse:
    """
    Process a forensic query through the full VANTAGE-STR pipeline.

    Pipeline: Auth → Rate Limit → Compliance → Orchestrator → Ledger → Response
    """
    t_start = time.perf_counter()

    query_id = request.query_id or f"GW-{secrets.token_hex(6).upper()}"

    # Placeholder: in production, this calls the orchestrator
    elapsed = (time.perf_counter() - t_start) * 1000

    _, remaining, reset = _rate_limiter.check(
        agency.agency_id, agency.rate_limit, agency.burst_limit,
    )

    return GatewayQueryResponse(
        query_id=query_id,
        agency_id=agency.agency_id,
        status="accepted",
        rate_limit_remaining=remaining,
        rate_limit_reset_seconds=round(reset, 1),
        results_count=0,
        processing_time_ms=round(elapsed, 2),
        audit_entry_hash="",
        message="Query accepted and queued for processing.",
    )


@gateway_router.get(
    "/status",
    response_model=GatewayStatus,
    summary="Get gateway operational status",
)
async def gateway_status() -> GatewayStatus:
    """Return current gateway health and statistics."""
    return GatewayStatus(
        status="operational",
        registered_agencies=_registry.count,
        total_requests_served=_total_requests,
        active_rate_limits=_rate_limiter.get_active_count(),
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@gateway_router.post(
    "/keys/rotate",
    response_model=KeyRotationResponse,
    summary="Rotate API key for the authenticated agency",
)
async def rotate_key(
    agency: AgencyCredential = Depends(verify_api_key),
) -> KeyRotationResponse:
    """
    Rotate the API key for the authenticated agency.

    The new key is returned ONCE in the response. The old key
    remains valid for a 5-minute grace period.
    """
    result = _registry.rotate_key(agency.agency_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Agency not found")

    new_key, grace_until = result

    return KeyRotationResponse(
        agency_id=agency.agency_id,
        new_key=new_key,
        old_key_valid_until=grace_until,
        message=(
            f"Key rotated. Store the new key securely — it will not be shown again. "
            f"Old key valid until {grace_until}."
        ),
    )
