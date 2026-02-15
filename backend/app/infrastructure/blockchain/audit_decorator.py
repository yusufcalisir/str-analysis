"""
Blockchain Audit Decorator — ForensicAudit Access Gate.

Provides a reusable function that:
    1. Reads the investigator's Ethereum address from the request header.
    2. Checks on-chain authorization (view call, zero gas).
    3. Logs the query on-chain (signed tx).
    4. Raises 403 if any step fails.

If the blockchain service is not configured, the gate passes silently
to allow local development without a running node.
"""

import hashlib
import logging
from functools import wraps
from typing import Callable

from fastapi import HTTPException, Request, status

from app.infrastructure.blockchain.web3_service import (
    BlockchainError,
    get_service,
)

logger = logging.getLogger(__name__)

INVESTIGATOR_HEADER = "X-Investigator-Id"


async def require_blockchain_audit(
    request: Request,
    query_type: str,
    profile_id: str,
) -> None:
    """
    Enforce on-chain access control before returning analysis results.

    Steps:
        1. Extract investigator address from X-Investigator-Id header.
        2. Call check_authorization — reject if NO_TOKEN or SUSPENDED.
        3. Call log_query — reject if tx reverts (rate limit breach, etc).

    Args:
        request: The incoming FastAPI request.
        query_type: Label for the audit log (e.g. "STR_ANALYSIS").
        profile_id: The profile being queried (hashed on-chain).

    Raises:
        HTTPException 403: On any authorization or logging failure.
    """
    service = get_service()
    if service is None:
        # Blockchain not configured — pass through (dev mode)
        return

    # -- Step 1: Read header --
    investigator_id = request.headers.get(INVESTIGATOR_HEADER)
    if not investigator_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Missing investigator identity",
                "reason": "MISSING_HEADER",
                "header": INVESTIGATOR_HEADER,
                "message": (
                    f"The '{INVESTIGATOR_HEADER}' header is required. "
                    "Provide your Ethereum address to authenticate."
                ),
            },
        )

    # -- Step 2: Check on-chain authorization --
    try:
        is_authorized, auth_status = service.check_authorization(investigator_id)
    except Exception as e:
        logger.error(f"[AUDIT] Authorization check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Blockchain authorization check failed",
                "reason": "AUTH_CHECK_ERROR",
                "message": str(e),
            },
        )

    if not is_authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Access denied by ForensicAudit contract",
                "reason": auth_status,
                "investigator": investigator_id,
                "message": (
                    "Your address does not have an active access token."
                    if auth_status == "NO_TOKEN"
                    else "Your account has been SUSPENDED due to rate-limit violation."
                ),
            },
        )

    # -- Step 3: Log query on-chain (atomic) --
    try:
        tx_hash = service.log_query(
            investigator_address=investigator_id,
            query_type=query_type,
            profile_id=profile_id,
        )
        logger.info(
            f"[AUDIT] Query logged on-chain — "
            f"tx={tx_hash[:16]}... type={query_type} profile={profile_id[:12]}..."
        )
    except BlockchainError as e:
        logger.warning(f"[AUDIT] logQuery reverted: {e} (reason={e.reason})")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "On-chain audit logging failed",
                "reason": e.reason,
                "investigator": investigator_id,
                "message": str(e),
            },
        )
