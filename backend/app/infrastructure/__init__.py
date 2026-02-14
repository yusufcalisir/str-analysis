"""
VANTAGE-STR Infrastructure Module.

Exports core infrastructure components for the forensic network:
    - ForensicLedger: Tamper-evident audit chain with Merkle tree
    - JurisdictionalComplianceEngine: Smart contract compliance gates
    - ComplianceRevertError: Exception for blocked queries
"""

from app.infrastructure.blockchain.ledger import (
    ForensicLedger,
    LedgerEntry,
    MerkleTree,
    IntegrityReport,
    LedgerStats,
    hash_query_params,
)
from app.infrastructure.blockchain.contracts.compliance import (
    JurisdictionalComplianceEngine,
    ComplianceRevertError,
    CourtOrder,
    AuthorizationResult,
)

__all__ = [
    "ForensicLedger",
    "LedgerEntry",
    "MerkleTree",
    "IntegrityReport",
    "LedgerStats",
    "hash_query_params",
    "JurisdictionalComplianceEngine",
    "ComplianceRevertError",
    "CourtOrder",
    "AuthorizationResult",
]
