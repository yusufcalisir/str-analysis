"""
AuditorAgent — DSPy-Powered Anomaly Detection for the Forensic Ledger.

Scans the ForensicLedger for suspicious access patterns and generates
structured anomaly alerts for system administrators.

Detection Layers:
    Layer 1 — Rule-Based Pre-Filter (deterministic, fast):
        - Burst queries: >N queries from one agency in T seconds
        - After-hours access: queries outside operational windows
        - Cross-border frequency spikes: unusual volume to a single country
        - Repeated reverted queries: agency keeps hitting compliance gates

    Layer 2 — DSPy Chain-of-Thought Analysis (nuanced, LLM):
        - Evaluates edge cases and contextual anomalies
        - Produces human-readable reasoning for administrators

Architecture mirrors ForensicValidator: rule-based first, DSPy second.

Usage:
    from app.infrastructure.blockchain.ledger import ForensicLedger
    agent = AuditorAgent()
    alerts = agent.scan(ledger.get_chain(limit=500))
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import dspy

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Rule-based thresholds
BURST_QUERY_THRESHOLD: int = 15       # Max queries from one agency per window
BURST_WINDOW_SECONDS: float = 300.0   # 5-minute window
AFTER_HOURS_START: int = 22           # 10 PM UTC
AFTER_HOURS_END: int = 5             # 5 AM UTC
REVERT_RATIO_THRESHOLD: float = 0.3   # >30% reverted queries = suspicious
CROSS_BORDER_SPIKE_RATIO: float = 3.0 # 3× above average = spike


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class AlertSeverity(str, Enum):
    """Anomaly alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Categories of detected anomalies."""
    BURST_QUERY = "burst_query"
    AFTER_HOURS = "after_hours"
    CROSS_BORDER_SPIKE = "cross_border_spike"
    REPEATED_REVERTS = "repeated_reverts"
    UNUSUAL_PATTERN = "unusual_pattern"  # Catch-all for DSPy-detected


@dataclass
class AnomalyAlert:
    """A single anomaly alert for the system administrator."""
    anomaly_type: str
    severity: str
    agency_id: str
    description: str
    evidence: Dict[str, Any]
    recommended_action: str
    detected_at: str = ""
    ai_reasoning: Optional[str] = None  # Populated by DSPy layer

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "agency_id": self.agency_id,
            "description": self.description,
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "detected_at": self.detected_at,
            "ai_reasoning": self.ai_reasoning,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DSPY SIGNATURE
# ═══════════════════════════════════════════════════════════════════════════════

class AuditAnomalySignature(dspy.Signature):
    """
    Analyze forensic ledger activity for suspicious access patterns.

    You are a cybersecurity analyst reviewing audit logs from a federated
    forensic DNA matching network. Your task is to identify anomalous
    behavior that could indicate abuse, unauthorized access, or policy
    violations.

    Consider: query frequency, timing, cross-border patterns, compliance
    failures, and any combinations that seem unusual or concerning.
    """

    # ── Inputs ──
    ledger_summary: str = dspy.InputField(
        desc=(
            "Statistical summary of recent ledger activity. Includes: "
            "query counts per agency, compliance revert rates, cross-border "
            "query distributions, time-of-day patterns, and ZKP verification "
            "outcomes. Format: structured text with labeled sections."
        )
    )
    flagged_patterns: str = dspy.InputField(
        desc=(
            "Pre-identified patterns from rule-based analysis that need "
            "deeper contextual evaluation. Each pattern includes: agency ID, "
            "pattern type, raw counts, and threshold comparison."
        )
    )

    # ── Outputs ──
    risk_assessment: str = dspy.OutputField(
        desc=(
            "Overall risk assessment: 'LOW', 'MEDIUM', 'HIGH', or 'CRITICAL'. "
            "Based on the combination and severity of detected anomalies."
        )
    )
    anomaly_analysis: str = dspy.OutputField(
        desc=(
            "Detailed analysis of each anomaly. For each: explain the concern, "
            "assess legitimacy probability, and provide context-aware reasoning. "
            "Consider operational norms, time zones, and legitimate surge scenarios."
        )
    )
    recommended_actions: str = dspy.OutputField(
        desc=(
            "Specific actions for the system administrator. One per anomaly. "
            "Options include: 'MONITOR', 'RATE_LIMIT', 'SUSPEND_ACCESS', "
            "'REQUIRE_REAUTHORIZATION', 'ESCALATE_TO_SUPERVISOR', 'NO_ACTION'."
        )
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED PRE-FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class RuleBasedAuditFilter:
    """
    Deterministic rule-based anomaly detection.

    Fast checks that catch obvious patterns before invoking the LLM.
    Mirrors the ForensicValidator pattern: rules first, DSPy second.
    """

    @staticmethod
    def scan(entries: List[Dict]) -> List[AnomalyAlert]:
        """
        Scan ledger entries for rule-based anomalies.

        Args:
            entries: List of serialized LedgerEntry dicts.

        Returns:
            List of detected anomaly alerts.
        """
        alerts: List[AnomalyAlert] = []

        if not entries:
            return alerts

        # Group entries by agency (node_id as proxy)
        by_agency: Dict[str, List[Dict]] = defaultdict(list)
        for entry in entries:
            by_agency[entry.get("node_id", "unknown")].append(entry)

        # ── Check 1: Burst Queries ──
        for agency_id, agency_entries in by_agency.items():
            timestamps = []
            for e in agency_entries:
                try:
                    ts = datetime.fromisoformat(e.get("timestamp", ""))
                    timestamps.append(ts.timestamp())
                except (ValueError, TypeError):
                    continue

            if len(timestamps) < 2:
                continue

            timestamps.sort()

            # Sliding window burst detection
            window_counts: List[int] = []
            for i, t in enumerate(timestamps):
                count = sum(
                    1 for t2 in timestamps[i:]
                    if t2 - t <= BURST_WINDOW_SECONDS
                )
                window_counts.append(count)

            max_burst = max(window_counts) if window_counts else 0

            if max_burst >= BURST_QUERY_THRESHOLD:
                alerts.append(AnomalyAlert(
                    anomaly_type=AnomalyType.BURST_QUERY.value,
                    severity=AlertSeverity.WARNING.value,
                    agency_id=agency_id,
                    description=(
                        f"Agency '{agency_id}' issued {max_burst} queries "
                        f"within a {BURST_WINDOW_SECONDS:.0f}s window "
                        f"(threshold: {BURST_QUERY_THRESHOLD})"
                    ),
                    evidence={
                        "max_burst_count": max_burst,
                        "threshold": BURST_QUERY_THRESHOLD,
                        "window_seconds": BURST_WINDOW_SECONDS,
                        "total_queries": len(agency_entries),
                    },
                    recommended_action="RATE_LIMIT",
                ))

        # ── Check 2: After-Hours Access ──
        after_hours_by_agency: Dict[str, int] = Counter()
        for entry in entries:
            try:
                ts = datetime.fromisoformat(entry.get("timestamp", ""))
                hour = ts.hour
                if hour >= AFTER_HOURS_START or hour < AFTER_HOURS_END:
                    agency = entry.get("node_id", "unknown")
                    after_hours_by_agency[agency] += 1
            except (ValueError, TypeError):
                continue

        for agency_id, count in after_hours_by_agency.items():
            total = len(by_agency.get(agency_id, []))
            ratio = count / total if total > 0 else 0

            if count >= 5 and ratio > 0.4:
                alerts.append(AnomalyAlert(
                    anomaly_type=AnomalyType.AFTER_HOURS.value,
                    severity=AlertSeverity.INFO.value,
                    agency_id=agency_id,
                    description=(
                        f"Agency '{agency_id}' made {count}/{total} queries "
                        f"({ratio:.0%}) outside operational hours "
                        f"({AFTER_HOURS_START}:00-{AFTER_HOURS_END}:00 UTC)"
                    ),
                    evidence={
                        "after_hours_count": count,
                        "total_queries": total,
                        "ratio": round(ratio, 3),
                    },
                    recommended_action="MONITOR",
                ))

        # ── Check 3: Repeated Compliance Reverts ──
        for agency_id, agency_entries in by_agency.items():
            reverted = sum(
                1 for e in agency_entries
                if e.get("compliance_decision") == "reverted"
            )
            total = len(agency_entries)
            ratio = reverted / total if total > 0 else 0

            if reverted >= 3 and ratio >= REVERT_RATIO_THRESHOLD:
                alerts.append(AnomalyAlert(
                    anomaly_type=AnomalyType.REPEATED_REVERTS.value,
                    severity=AlertSeverity.CRITICAL.value,
                    agency_id=agency_id,
                    description=(
                        f"Agency '{agency_id}' has {reverted}/{total} queries "
                        f"reverted ({ratio:.0%}) — possible unauthorized access attempt"
                    ),
                    evidence={
                        "reverted_count": reverted,
                        "total_queries": total,
                        "ratio": round(ratio, 3),
                        "threshold": REVERT_RATIO_THRESHOLD,
                    },
                    recommended_action="SUSPEND_ACCESS",
                ))

        # ── Check 4: Cross-Border Frequency Spike ──
        cross_border_counts: Dict[str, int] = Counter()
        for entry in entries:
            if entry.get("metadata", {}).get("cross_border", False):
                target = entry.get("metadata", {}).get("target_country", "??")
                source_agency = entry.get("node_id", "unknown")
                cross_border_counts[f"{source_agency}→{target}"] += 1

        if cross_border_counts:
            avg_count = sum(cross_border_counts.values()) / len(cross_border_counts)
            for route, count in cross_border_counts.items():
                if avg_count > 0 and count / avg_count >= CROSS_BORDER_SPIKE_RATIO:
                    alerts.append(AnomalyAlert(
                        anomaly_type=AnomalyType.CROSS_BORDER_SPIKE.value,
                        severity=AlertSeverity.WARNING.value,
                        agency_id=route.split("→")[0],
                        description=(
                            f"Cross-border route '{route}' has {count} queries "
                            f"({count / avg_count:.1f}× above average)"
                        ),
                        evidence={
                            "route": route,
                            "count": count,
                            "average": round(avg_count, 1),
                            "spike_ratio": round(count / avg_count, 2),
                        },
                        recommended_action="ESCALATE_TO_SUPERVISOR",
                    ))

        return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# AUDITOR AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class AuditorAgent:
    """
    DSPy-powered audit agent combining rule-based and LLM analysis.

    Layer 1 (Rules):  Fast, deterministic anomaly detection.
    Layer 2 (DSPy):   Contextual analysis of flagged patterns.

    Usage:
        agent = AuditorAgent()
        alerts = agent.scan(ledger_entries)

    The agent can operate in rule-only mode (no LLM) for environments
    where DSPy is not configured, by setting use_dspy=False.
    """

    def __init__(self, use_dspy: bool = True) -> None:
        self._rule_filter = RuleBasedAuditFilter()
        self._use_dspy = use_dspy
        self._predictor: Optional[dspy.ChainOfThought] = None

        if use_dspy:
            try:
                self._predictor = dspy.ChainOfThought(AuditAnomalySignature)
            except Exception as exc:
                logger.warning(
                    f"[AUDITOR] DSPy initialization failed: {exc} — "
                    f"falling back to rule-only mode"
                )
                self._use_dspy = False

    def scan(
        self,
        entries: List[Dict],
        enrich_with_dspy: bool = True,
    ) -> List[AnomalyAlert]:
        """
        Scan ledger entries for anomalies.

        Args:
            entries: List of serialized LedgerEntry dicts.
            enrich_with_dspy: Whether to run DSPy enrichment on results.

        Returns:
            List of AnomalyAlert objects, sorted by severity.
        """
        t_start = time.perf_counter()

        # Layer 1: Rule-based detection
        alerts = self._rule_filter.scan(entries)

        # Layer 2: DSPy enrichment (if available and requested)
        if self._use_dspy and self._predictor and enrich_with_dspy and alerts:
            alerts = self._enrich_with_dspy(entries, alerts)

        # Sort by severity: CRITICAL > WARNING > INFO
        severity_order = {
            AlertSeverity.CRITICAL.value: 0,
            AlertSeverity.WARNING.value: 1,
            AlertSeverity.INFO.value: 2,
        }
        alerts.sort(key=lambda a: severity_order.get(a.severity, 99))

        elapsed = (time.perf_counter() - t_start) * 1000
        logger.info(
            f"[AUDITOR] Scan complete — {len(alerts)} anomalies detected "
            f"across {len(entries)} entries ({elapsed:.1f}ms)"
        )

        return alerts

    def _enrich_with_dspy(
        self,
        entries: List[Dict],
        alerts: List[AnomalyAlert],
    ) -> List[AnomalyAlert]:
        """
        Run DSPy analysis to add contextual reasoning to alerts.

        Builds a statistical summary of the ledger and the flagged
        patterns, then asks the LLM for nuanced assessment.
        """
        try:
            # Build ledger summary
            summary = self._build_ledger_summary(entries)

            # Format flagged patterns
            flagged = "\n".join(
                f"- [{a.severity.upper()}] {a.anomaly_type}: {a.description}"
                for a in alerts
            )

            # Run DSPy predictor
            result = self._predictor(
                ledger_summary=summary,
                flagged_patterns=flagged,
            )

            # Attach AI reasoning to alerts
            ai_reasoning = getattr(result, "anomaly_analysis", "")
            for alert in alerts:
                alert.ai_reasoning = ai_reasoning

            logger.info(
                f"[AUDITOR] DSPy enrichment complete — "
                f"risk_level={getattr(result, 'risk_assessment', 'UNKNOWN')}"
            )

        except Exception as exc:
            logger.warning(f"[AUDITOR] DSPy enrichment failed: {exc}")

        return alerts

    @staticmethod
    def _build_ledger_summary(entries: List[Dict]) -> str:
        """Build a statistical summary of ledger entries for DSPy input."""
        if not entries:
            return "No entries to analyze."

        total = len(entries)
        agencies = Counter(e.get("node_id", "?") for e in entries)
        zkp_statuses = Counter(e.get("zkp_status", "?") for e in entries)
        compliance = Counter(e.get("compliance_decision", "?") for e in entries)

        cross_border = sum(
            1 for e in entries
            if e.get("metadata", {}).get("cross_border", False)
        )

        lines = [
            f"Total entries: {total}",
            f"Cross-border queries: {cross_border} ({cross_border/total:.0%})",
            "",
            "Queries per agency:",
            *[f"  {a}: {c}" for a, c in agencies.most_common()],
            "",
            "ZKP verification outcomes:",
            *[f"  {s}: {c}" for s, c in zkp_statuses.most_common()],
            "",
            "Compliance decisions:",
            *[f"  {d}: {c}" for d, c in compliance.most_common()],
        ]

        return "\n".join(lines)
