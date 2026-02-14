"""
Jurisdictional Compliance Engine — Smart Contract Logic for VANTAGE-STR.

Python-based equivalent of a Solidity smart contract, enforcing jurisdictional
compliance before any query is broadcast to the federated network.

Three-Gate Authorization:
    ┌─────────────────────┐
    │ Gate 1: Court Order  │──→ Valid ID? Not expired?
    │ Gate 2: Agency Auth  │──→ Agency authorized for this crime category?
    │ Gate 3: Cross-Border │──→ Bilateral treaty allows this query type?
    └─────────────────────┘
    If ANY gate fails → ComplianceRevertError (query blocked)

Design Decision:
    We use Python rather than Solidity because the VANTAGE-STR network
    runs a custom sidechain (ForensicLedger) — not EVM. The logic is
    identical to what a Solidity contract would enforce, but executes
    natively in the Orchestrator's process space for sub-millisecond
    latency.

Usage:
    engine = JurisdictionalComplianceEngine()
    result = engine.authorize_query(
        court_order_id="CO-2026-DE-4492",
        agency_id="BKA-DE",
        crime_category="HOMICIDE",
        source_country="DE",
        target_country="NL",
        query_type="cross_border_str_match",
    )
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class ComplianceRevertError(Exception):
    """
    Raised when a query fails jurisdictional compliance.

    Equivalent to a Solidity 'revert' — the query is blocked and the
    reason is logged to the forensic ledger.
    """

    def __init__(self, gate: str, reason: str, details: Dict[str, Any] = None):
        self.gate = gate
        self.reason = reason
        self.details = details or {}
        super().__init__(f"COMPLIANCE REVERT [{gate}]: {reason}")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CrimeCategory(str, Enum):
    """Standardized crime categories for authorization matrix."""
    HOMICIDE = "HOMICIDE"
    SEXUAL_ASSAULT = "SEXUAL_ASSAULT"
    TERRORISM = "TERRORISM"
    KIDNAPPING = "KIDNAPPING"
    ORGANIZED_CRIME = "ORGANIZED_CRIME"
    MISSING_PERSONS = "MISSING_PERSONS"
    WAR_CRIMES = "WAR_CRIMES"
    DRUG_TRAFFICKING = "DRUG_TRAFFICKING"
    FRAUD = "FRAUD"
    OTHER = "OTHER"


class QueryType(str, Enum):
    """Types of queries that can be made across borders."""
    DOMESTIC_STR_MATCH = "domestic_str_match"
    CROSS_BORDER_STR_MATCH = "cross_border_str_match"
    INTERPOL_RED_NOTICE = "interpol_red_notice"
    MUTUAL_LEGAL_ASSISTANCE = "mutual_legal_assistance"


class CourtOrder(BaseModel):
    """Registered court order authorizing a DNA query."""
    order_id: str
    issued_by: str  # Issuing court/authority
    agency_id: str  # Requesting agency
    crime_category: str
    issued_at: str
    expires_at: str
    target_countries: List[str] = Field(default_factory=list)
    revoked: bool = False

    @property
    def is_expired(self) -> bool:
        try:
            exp = datetime.fromisoformat(self.expires_at)
            return datetime.now(timezone.utc) > exp
        except (ValueError, TypeError):
            return True

    @property
    def is_valid(self) -> bool:
        return not self.revoked and not self.is_expired


class AuthorizationResult(BaseModel):
    """Result of a full compliance check."""
    authorized: bool = False
    gate_results: Dict[str, bool] = Field(default_factory=dict)
    revert_reason: str = ""
    revert_gate: str = ""
    court_order_id: str = ""
    agency_id: str = ""
    crime_category: str = ""
    source_country: str = ""
    target_country: str = ""
    checked_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# JURISDICTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Agency → allowed crime categories
DEFAULT_AGENCY_PERMISSIONS: Dict[str, Set[str]] = {
    "INTERPOL-EU": {c.value for c in CrimeCategory},  # Full access
    "BKA-DE": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "KIDNAPPING",
               "ORGANIZED_CRIME", "WAR_CRIMES", "MISSING_PERSONS"},
    "FBI-US": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "KIDNAPPING",
               "ORGANIZED_CRIME", "DRUG_TRAFFICKING"},
    "NCA-UK": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "ORGANIZED_CRIME",
               "MISSING_PERSONS"},
    "EUROPOL-NL": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "ORGANIZED_CRIME",
                   "WAR_CRIMES", "DRUG_TRAFFICKING"},
    "AFP-AU": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "MISSING_PERSONS"},
    "NPA-JP": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "KIDNAPPING"},
    "RCMP-CA": {"HOMICIDE", "SEXUAL_ASSAULT", "TERRORISM", "MISSING_PERSONS",
                "ORGANIZED_CRIME"},
}

# (source_country, target_country) → allowed query types
# Based on real bilateral MLAT (Mutual Legal Assistance Treaty) relationships
DEFAULT_BILATERAL_TREATIES: Dict[tuple, Set[str]] = {
    # EU Prüm Convention — automatic cross-border DNA sharing
    ("DE", "NL"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("NL", "DE"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("DE", "FR"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("FR", "DE"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("DE", "AT"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("NL", "FR"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},

    # Five Eyes — US/UK/CA/AU MLAT
    ("US", "UK"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("UK", "US"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("US", "CA"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("CA", "US"): {QueryType.CROSS_BORDER_STR_MATCH.value, QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("US", "AU"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("AU", "US"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("UK", "AU"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("AU", "UK"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},

    # INTERPOL — universal access
    ("INTERPOL", "*"): {QueryType.INTERPOL_RED_NOTICE.value, QueryType.CROSS_BORDER_STR_MATCH.value},

    # Japan — restrictive, MLA only
    ("JP", "US"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
    ("US", "JP"): {QueryType.MUTUAL_LEGAL_ASSISTANCE.value},
}


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class JurisdictionalComplianceEngine:
    """
    Smart contract equivalent enforcing jurisdictional DNA query compliance.

    Three gates must pass before a query is authorized:
        Gate 1 — Court Order: valid, not expired, not revoked.
        Gate 2 — Agency Authorization: agency allowed for this crime category.
        Gate 3 — Cross-Border Treaty: bilateral treaty permits this query type.

    If any gate fails, the engine 'reverts' (raises ComplianceRevertError)
    and the query is blocked. All decisions are logged to the ForensicLedger.

    Usage:
        engine = JurisdictionalComplianceEngine()
        engine.register_court_order(CourtOrder(...))
        result = engine.authorize_query(...)
    """

    def __init__(
        self,
        agency_permissions: Optional[Dict[str, Set[str]]] = None,
        bilateral_treaties: Optional[Dict[tuple, Set[str]]] = None,
    ) -> None:
        self._court_orders: Dict[str, CourtOrder] = {}
        self._agency_permissions = agency_permissions or DEFAULT_AGENCY_PERMISSIONS
        self._bilateral_treaties = bilateral_treaties or DEFAULT_BILATERAL_TREATIES
        self._decision_log: List[AuthorizationResult] = []

    # ── Court Order Registry ──

    def register_court_order(self, order: CourtOrder) -> None:
        """Register a court order in the compliance engine."""
        self._court_orders[order.order_id] = order
        logger.info(
            f"[COMPLIANCE] Court order {order.order_id} registered — "
            f"agency={order.agency_id} crime={order.crime_category} "
            f"expires={order.expires_at}"
        )

    def revoke_court_order(self, order_id: str) -> bool:
        """Revoke a court order. Returns True if found and revoked."""
        if order_id in self._court_orders:
            order = self._court_orders[order_id]
            self._court_orders[order_id] = order.model_copy(
                update={"revoked": True}
            )
            logger.warning(f"[COMPLIANCE] Court order {order_id} REVOKED")
            return True
        return False

    # ── Authorization (Main Entry Point) ──

    def authorize_query(
        self,
        court_order_id: str,
        agency_id: str,
        crime_category: str,
        source_country: str,
        target_country: str,
        query_type: str = QueryType.DOMESTIC_STR_MATCH.value,
    ) -> AuthorizationResult:
        """
        Run the full three-gate compliance check.

        If any gate fails, raises ComplianceRevertError (query blocked).
        If all gates pass, returns an AuthorizationResult with authorized=True.

        Args:
            court_order_id: Unique court order identifier.
            agency_id: Requesting law enforcement agency.
            crime_category: Category of the crime under investigation.
            source_country: Country of the requesting agency.
            target_country: Country of the target node.
            query_type: Type of query being made.

        Returns:
            AuthorizationResult with gate-by-gate outcomes.

        Raises:
            ComplianceRevertError: If any gate fails.
        """
        t_start = time.perf_counter()

        gate_results: Dict[str, bool] = {}
        result_kwargs = {
            "court_order_id": court_order_id,
            "agency_id": agency_id,
            "crime_category": crime_category,
            "source_country": source_country,
            "target_country": target_country,
        }

        try:
            # ── Gate 1: Court Order Validation ──
            gate_results["court_order"] = self._gate_court_order(
                court_order_id, agency_id,
            )

            # ── Gate 2: Agency Authorization ──
            gate_results["agency_auth"] = self._gate_agency_authorization(
                agency_id, crime_category,
            )

            # ── Gate 3: Cross-Border Treaty ──
            is_cross_border = source_country != target_country
            if is_cross_border:
                gate_results["cross_border"] = self._gate_cross_border(
                    source_country, target_country, query_type,
                )
            else:
                gate_results["cross_border"] = True  # Domestic — auto-pass

            latency = (time.perf_counter() - t_start) * 1000

            result = AuthorizationResult(
                authorized=True,
                gate_results=gate_results,
                latency_ms=round(latency, 3),
                **result_kwargs,
            )

            self._decision_log.append(result)
            logger.info(
                f"[COMPLIANCE] AUTHORIZED — agency={agency_id} "
                f"order={court_order_id} crime={crime_category} "
                f"route={source_country}→{target_country} "
                f"({latency:.2f}ms)"
            )
            return result

        except ComplianceRevertError as exc:
            latency = (time.perf_counter() - t_start) * 1000
            gate_results[exc.gate] = False

            result = AuthorizationResult(
                authorized=False,
                gate_results=gate_results,
                revert_reason=exc.reason,
                revert_gate=exc.gate,
                latency_ms=round(latency, 3),
                **result_kwargs,
            )

            self._decision_log.append(result)
            logger.warning(
                f"[COMPLIANCE] REVERTED — gate={exc.gate} "
                f"reason='{exc.reason}' agency={agency_id} "
                f"({latency:.2f}ms)"
            )
            raise  # Re-raise for caller to handle

    # ── Gate Implementations ──

    def _gate_court_order(
        self, court_order_id: str, agency_id: str,
    ) -> bool:
        """
        Gate 1: Validate the court order.

        Checks:
            - Order exists in the registry
            - Order is not revoked
            - Order is not expired
            - Requesting agency matches the order's agency
        """
        if court_order_id not in self._court_orders:
            raise ComplianceRevertError(
                gate="court_order",
                reason=f"Court order '{court_order_id}' not found in registry",
                details={"order_id": court_order_id},
            )

        order = self._court_orders[court_order_id]

        if order.revoked:
            raise ComplianceRevertError(
                gate="court_order",
                reason=f"Court order '{court_order_id}' has been revoked",
                details={"order_id": court_order_id, "revoked": True},
            )

        if order.is_expired:
            raise ComplianceRevertError(
                gate="court_order",
                reason=f"Court order '{court_order_id}' expired at {order.expires_at}",
                details={"order_id": court_order_id, "expires_at": order.expires_at},
            )

        if order.agency_id != agency_id:
            raise ComplianceRevertError(
                gate="court_order",
                reason=f"Agency '{agency_id}' is not the authorized requestor "
                       f"(expected '{order.agency_id}')",
                details={"expected": order.agency_id, "actual": agency_id},
            )

        return True

    def _gate_agency_authorization(
        self, agency_id: str, crime_category: str,
    ) -> bool:
        """
        Gate 2: Check agency authorization for the crime category.

        Uses the permission matrix to determine if the agency is
        allowed to query for this type of crime.
        """
        # Extract base agency (strip country suffix variations)
        agency_base = agency_id.split("-")[0] + "-" + agency_id.split("-")[1] if "-" in agency_id else agency_id

        allowed_categories = self._agency_permissions.get(agency_base)

        if allowed_categories is None:
            raise ComplianceRevertError(
                gate="agency_auth",
                reason=f"Agency '{agency_id}' is not registered in the permission matrix",
                details={"agency_id": agency_id},
            )

        if crime_category not in allowed_categories:
            raise ComplianceRevertError(
                gate="agency_auth",
                reason=f"Agency '{agency_id}' is not authorized for "
                       f"crime category '{crime_category}'",
                details={
                    "agency_id": agency_id,
                    "crime_category": crime_category,
                    "allowed": sorted(allowed_categories),
                },
            )

        return True

    def _gate_cross_border(
        self, source: str, target: str, query_type: str,
    ) -> bool:
        """
        Gate 3: Check bilateral treaty for cross-border query.

        Verifies that a mutual legal assistance treaty or equivalent
        agreement exists between the two countries for the requested
        query type.
        """
        # Check direct bilateral treaty
        key = (source, target)
        allowed_types = self._bilateral_treaties.get(key)

        # Check INTERPOL wildcard
        if allowed_types is None:
            key_wildcard = (source, "*")
            allowed_types = self._bilateral_treaties.get(key_wildcard)

        if allowed_types is None:
            raise ComplianceRevertError(
                gate="cross_border",
                reason=f"No bilateral treaty between '{source}' and '{target}'",
                details={"source": source, "target": target},
            )

        if query_type not in allowed_types:
            raise ComplianceRevertError(
                gate="cross_border",
                reason=f"Treaty between '{source}' and '{target}' does not "
                       f"permit query type '{query_type}'",
                details={
                    "source": source,
                    "target": target,
                    "query_type": query_type,
                    "allowed_types": sorted(allowed_types),
                },
            )

        return True

    # ── Accessors ──

    @property
    def decision_count(self) -> int:
        """Number of compliance decisions made."""
        return len(self._decision_log)

    @property
    def revert_count(self) -> int:
        """Number of reverted (blocked) queries."""
        return sum(1 for d in self._decision_log if not d.authorized)

    def get_decision_log(self, limit: int = 50) -> List[Dict]:
        """Get recent compliance decisions."""
        return [
            d.model_dump() for d in reversed(self._decision_log[-limit:])
        ]
