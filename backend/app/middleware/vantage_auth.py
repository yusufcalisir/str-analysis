import logging
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
try:
    from app.infrastructure.blockchain.web3_service import get_service
except ImportError:
    def get_service(): return None

logger = logging.getLogger(__name__)

class VantageAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 1. Filter for protected routes
        if request.url.path.startswith("/profile"):
            
            # 2. Check if blockchain service is even available
            service = get_service()
            blockchain_available = service is not None and service.is_connected()

            # 3. Extract Headers
            investigator = request.headers.get("X-Investigator-Address")
            session_token = request.headers.get("X-Session-Token")

            if blockchain_available:
                # Strict enforcement when blockchain is online
                if not investigator or not session_token:
                    return await self._deny_access("Missing authentication headers.")

                is_authorized = service.is_investigator_authorized(investigator)
                if not is_authorized:
                    return await self._deny_access("Blockchain Audit: Access Denied or Suspended.")
            else:
                # Safety Mode: blockchain unavailable — allow with warning
                logger.warning("[AUTH] Blockchain unavailable — Safety Mode active, allowing request.")
                investigator = investigator or "SAFETY-MODE-DEV"
                session_token = session_token or "SAFETY-MODE-TOKEN"

            # 4. Attach to request state for endpoint use
            request.state.investigator = investigator
            request.state.session_token = session_token

        response = await call_next(request)
        return response

    async def _deny_access(self, detail: str):
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"detail": detail}
        )
