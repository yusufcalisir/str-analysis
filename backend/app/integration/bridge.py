"""
Inter-Agency Interoperability Bridge — INTERPOL & Prüm XML Formatters.

Translates VANTAGE-STR match results into internationally recognized
XML formats so agencies NOT on the VANTAGE network can receive and
process cryptographically verified DNA match alerts.

Supported Standards:
    1. INTERPOL I-24/7 DNA Profile Exchange (XML)
    2. EU Prüm Convention DNA Step 1 (anonymous hit) & Step 2 (identity request)

Architecture:
    MatchResult → InteroperabilityBridge → XML string
                                         → format auto-detected by target

Usage:
    bridge = InteroperabilityBridge()
    xml = bridge.format("interpol", match_data)
    xml = bridge.format("prum_step1", match_data)
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class MatchExportData:
    """Normalized match data for export formatting."""

    def __init__(
        self,
        query_id: str,
        match_id: str,
        requesting_agency: str,
        requesting_country: str,
        target_node_id: str,
        target_country: str,
        similarity_score: float,
        zkp_status: str,
        zkp_proof_hash: str = "",
        court_order_id: str = "",
        crime_category: str = "",
        loci_compared: int = 0,
        audit_entry_hash: str = "",
        timestamp: Optional[str] = None,
    ):
        self.query_id = query_id
        self.match_id = match_id
        self.requesting_agency = requesting_agency
        self.requesting_country = requesting_country
        self.target_node_id = target_node_id
        self.target_country = target_country
        self.similarity_score = similarity_score
        self.zkp_status = zkp_status
        self.zkp_proof_hash = zkp_proof_hash
        self.court_order_id = court_order_id
        self.crime_category = crime_category
        self.loci_compared = loci_compared
        self.audit_entry_hash = audit_entry_hash
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPOL I-24/7 XML FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

class InterpolXMLFormatter:
    """
    Formats match results into INTERPOL I-24/7 DNA exchange XML.

    Based on INTERPOL's DNA Gateway standards. Includes:
        - Match header with requesting NCB (National Central Bureau)
        - Cryptographic verification status (ZKP proof hash)
        - Score and loci comparison summary
        - Audit trail reference
    """

    NAMESPACE = "urn:interpol:dna:exchange:2024"

    @classmethod
    def format(cls, data: MatchExportData) -> str:
        """Generate INTERPOL I-24/7 compliant XML."""
        root = Element("InterpolDNAAlert")
        root.set("xmlns", cls.NAMESPACE)
        root.set("version", "2.1")
        root.set("timestamp", data.timestamp)

        # Header
        header = SubElement(root, "AlertHeader")
        SubElement(header, "AlertID").text = f"INTERPOL-{data.query_id}"
        SubElement(header, "AlertType").text = "DNA_MATCH"
        SubElement(header, "Priority").text = (
            "URGENT" if data.similarity_score >= 0.95 else "STANDARD"
        )
        SubElement(header, "GeneratedBy").text = "VANTAGE-STR"

        # Requesting party
        requestor = SubElement(root, "RequestingParty")
        SubElement(requestor, "NCBCode").text = data.requesting_agency
        SubElement(requestor, "Country").text = data.requesting_country
        SubElement(requestor, "CourtOrderRef").text = data.court_order_id
        SubElement(requestor, "CrimeCategory").text = data.crime_category

        # Match details
        match = SubElement(root, "MatchDetails")
        SubElement(match, "MatchID").text = data.match_id
        SubElement(match, "TargetNCB").text = data.target_node_id
        SubElement(match, "TargetCountry").text = data.target_country
        SubElement(match, "SimilarityScore").text = f"{data.similarity_score:.6f}"
        SubElement(match, "LociCompared").text = str(data.loci_compared)
        SubElement(match, "MatchClassification").text = cls._classify_match(
            data.similarity_score
        )

        # Cryptographic verification
        crypto = SubElement(root, "CryptographicVerification")
        SubElement(crypto, "ZKPStatus").text = data.zkp_status.upper()
        SubElement(crypto, "ProofHash").text = data.zkp_proof_hash
        SubElement(crypto, "AuditChainRef").text = data.audit_entry_hash
        SubElement(crypto, "VerificationProtocol").text = "Groth16-BN254"

        return cls._prettify(root)

    @staticmethod
    def _classify_match(score: float) -> str:
        if score >= 0.99:
            return "IDENTITY"
        if score >= 0.95:
            return "NEAR_IDENTITY"
        if score >= 0.85:
            return "HIGH_SIMILARITY"
        if score >= 0.70:
            return "MODERATE_SIMILARITY"
        return "LOW_SIMILARITY"

    @staticmethod
    def _prettify(element: Element) -> str:
        raw = tostring(element, encoding="unicode", xml_declaration=False)
        return parseString(raw).toprettyxml(indent="  ", encoding=None)


# ═══════════════════════════════════════════════════════════════════════════════
# EU PRÜM CONVENTION XML FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

class PrumXMLFormatter:
    """
    Formats match results into EU Prüm Convention DNA exchange XML.

    Two-step process:
        Step 1 — Anonymous hit notification (no personal data)
        Step 2 — Identity request (requires MLAT or Prüm authorization)
    """

    NAMESPACE = "urn:eu:prum:dna:exchange:2008"

    @classmethod
    def format_step1(cls, data: MatchExportData) -> str:
        """
        Step 1: Anonymous hit notification.

        Contains only the match score and loci count — no personal data.
        """
        root = Element("PrumDNAMatchNotification")
        root.set("xmlns", cls.NAMESPACE)
        root.set("version", "1.0")
        root.set("step", "1")
        root.set("timestamp", data.timestamp)

        # Reference
        ref = SubElement(root, "Reference")
        SubElement(ref, "RequestID").text = data.query_id
        SubElement(ref, "MatchID").text = data.match_id
        SubElement(ref, "RequestingState").text = data.requesting_country
        SubElement(ref, "MatchingState").text = data.target_country

        # Anonymous result
        result = SubElement(root, "MatchResult")
        SubElement(result, "HitType").text = cls._hit_type(data.similarity_score)
        SubElement(result, "QualityScore").text = f"{data.similarity_score:.6f}"
        SubElement(result, "LociCompared").text = str(data.loci_compared)

        # VANTAGE-STR extension
        ext = SubElement(root, "VantageExtension")
        SubElement(ext, "ZKPVerified").text = str(data.zkp_status == "verified").lower()
        SubElement(ext, "AuditRef").text = data.audit_entry_hash

        return cls._prettify(root)

    @classmethod
    def format_step2(cls, data: MatchExportData) -> str:
        """
        Step 2: Identity request (requires bilateral authorization).

        Adds court order reference and formal request for personal data exchange.
        """
        root = Element("PrumIdentityRequest")
        root.set("xmlns", cls.NAMESPACE)
        root.set("version", "1.0")
        root.set("step", "2")
        root.set("timestamp", data.timestamp)

        # Reference (same as Step 1)
        ref = SubElement(root, "Reference")
        SubElement(ref, "OriginalMatchID").text = data.match_id
        SubElement(ref, "RequestingState").text = data.requesting_country
        SubElement(ref, "MatchingState").text = data.target_country

        # Legal basis
        legal = SubElement(root, "LegalBasis")
        SubElement(legal, "CourtOrderID").text = data.court_order_id
        SubElement(legal, "CrimeCategory").text = data.crime_category
        SubElement(legal, "RequestingAgency").text = data.requesting_agency
        SubElement(legal, "LegalFramework").text = "PRUM_CONVENTION_2008"

        # Crypto proof
        crypto = SubElement(root, "CryptographicAttestation")
        SubElement(crypto, "ZKPProofHash").text = data.zkp_proof_hash
        SubElement(crypto, "VerificationStatus").text = data.zkp_status.upper()
        SubElement(crypto, "AuditChainRef").text = data.audit_entry_hash

        return cls._prettify(root)

    @staticmethod
    def _hit_type(score: float) -> str:
        if score >= 0.99:
            return "FULL_MATCH"
        if score >= 0.90:
            return "NEAR_MATCH"
        if score >= 0.70:
            return "PARTIAL_MATCH"
        return "NO_MATCH"

    @staticmethod
    def _prettify(element: Element) -> str:
        raw = tostring(element, encoding="unicode", xml_declaration=False)
        return parseString(raw).toprettyxml(indent="  ", encoding=None)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEROPERABILITY BRIDGE (FACADE)
# ═══════════════════════════════════════════════════════════════════════════════

class InteroperabilityBridge:
    """
    Unified facade for all international DNA exchange format converters.

    Routes to the correct formatter based on the target standard:
        "interpol"    → INTERPOL I-24/7 XML
        "prum_step1"  → Prüm Convention Step 1 (anonymous hit)
        "prum_step2"  → Prüm Convention Step 2 (identity request)

    Usage:
        bridge = InteroperabilityBridge()
        xml = bridge.format("interpol", match_data)
    """

    FORMATTERS = {
        "interpol": lambda d: InterpolXMLFormatter.format(d),
        "prum_step1": lambda d: PrumXMLFormatter.format_step1(d),
        "prum_step2": lambda d: PrumXMLFormatter.format_step2(d),
    }

    @classmethod
    def format(cls, standard: str, data: MatchExportData) -> str:
        """
        Convert match data to the specified international XML format.

        Args:
            standard: One of "interpol", "prum_step1", "prum_step2".
            data: Normalized match data.

        Returns:
            Formatted XML string.

        Raises:
            ValueError: If the standard is not supported.
        """
        formatter = cls.FORMATTERS.get(standard)
        if formatter is None:
            supported = ", ".join(cls.FORMATTERS.keys())
            raise ValueError(
                f"Unsupported standard '{standard}'. Supported: {supported}"
            )

        xml = formatter(data)
        logger.info(
            f"[BRIDGE] Exported {standard} XML — "
            f"match={data.match_id} score={data.similarity_score:.4f}"
        )

        return xml

    @classmethod
    def supported_standards(cls) -> List[str]:
        """List of supported export standards."""
        return list(cls.FORMATTERS.keys())
