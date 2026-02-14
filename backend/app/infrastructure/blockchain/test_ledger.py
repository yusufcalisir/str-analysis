"""
Ledger Verification Tests — Phase 5.1.

Tests ForensicLedger integrity, tamper detection, MerkleTree, and
compliance engine import validation.

Run: python backend/app/infrastructure/blockchain/test_ledger.py
"""

import asyncio
import hashlib
import json
import sys
import time
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))

from app.infrastructure.blockchain.ledger import (
    ForensicLedger,
    MerkleTree,
    _sha256,
    _compute_entry_hash,
    hash_query_params,
    GENESIS_HASH,
)
from app.infrastructure.blockchain.contracts.compliance import (
    JurisdictionalComplianceEngine,
    ComplianceRevertError,
    CourtOrder,
)


PASS = 0
FAIL = 0


def test(name: str, passed: bool):
    global PASS, FAIL
    status = "\033[32m✓ PASS\033[0m" if passed else "\033[31m✗ FAIL\033[0m"
    print(f"  {status}  {name}")
    if passed:
        PASS += 1
    else:
        FAIL += 1


# ═══════════════════════════════════════════════════════════════════════════════
# MERKLE TREE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_merkle_empty():
    tree = MerkleTree()
    test("MerkleTree: empty root is genesis hash", tree.root == GENESIS_HASH)


def test_merkle_single_leaf():
    tree = MerkleTree()
    h = _sha256("entry_0")
    root = tree.add_leaf(h)
    test("MerkleTree: single leaf root equals the leaf hash", root == h)


def test_merkle_two_leaves():
    tree = MerkleTree()
    h1 = _sha256("entry_0")
    h2 = _sha256("entry_1")
    tree.add_leaf(h1)
    root = tree.add_leaf(h2)
    expected = _sha256(h1 + h2)
    test("MerkleTree: two leaves produce correct root", root == expected)


def test_merkle_verify():
    tree = MerkleTree()
    hashes = [_sha256(f"entry_{i}") for i in range(5)]
    for h in hashes:
        tree.add_leaf(h)
    test("MerkleTree: verify returns True for correct hashes", tree.verify(hashes))
    test("MerkleTree: verify returns False for tampered hashes",
         not tree.verify(hashes[:-1] + [_sha256("tampered")]))


# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC LEDGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════

async def test_ledger_append_and_integrity():
    ForensicLedger.reset_instance()
    ledger = ForensicLedger.get_instance()
    await ledger.start()

    # Record 3 events
    for i in range(3):
        await ledger.record_event(
            query_hash=_sha256(f"query_{i}"),
            node_id=f"NODE-{i}",
            zkp_status="verified",
        )

    # Give consumer time to process
    await asyncio.sleep(0.5)

    test("Ledger: 3 entries committed", ledger.chain_length == 3)

    # Verify integrity
    report = ledger.verify_integrity()
    test("Ledger: chain integrity verified", report.is_valid)
    test("Ledger: merkle root is non-genesis", ledger.merkle_root != GENESIS_HASH)

    await ledger.stop()
    ForensicLedger.reset_instance()


async def test_ledger_tamper_detection():
    ForensicLedger.reset_instance()
    ledger = ForensicLedger.get_instance()
    await ledger.start()

    await ledger.record_event(
        query_hash=_sha256("legit_query"),
        node_id="BKA-DE",
        zkp_status="verified",
    )
    await ledger.record_event(
        query_hash=_sha256("another_query"),
        node_id="EUROPOL-NL",
        zkp_status="verified",
    )
    await asyncio.sleep(0.5)

    # Tamper with the chain (simulate modifying a frozen dataclass by replacing it)
    original = ledger._chain[0]
    from dataclasses import replace
    tampered = replace(original, query_hash="TAMPERED_HASH")
    ledger._chain[0] = tampered

    report = ledger.verify_integrity()
    test("Ledger: tamper detection catches modified entry", not report.is_valid)
    test("Ledger: tamper at index 0 detected", report.first_invalid_index == 0)

    await ledger.stop()
    ForensicLedger.reset_instance()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLIANCE ENGINE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_compliance_authorized():
    engine = JurisdictionalComplianceEngine()
    from datetime import datetime, timezone, timedelta
    engine.register_court_order(CourtOrder(
        order_id="CO-2026-DE-001",
        issued_by="Berlin District Court",
        agency_id="BKA-DE",
        crime_category="HOMICIDE",
        issued_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
        target_countries=["NL"],
    ))

    result = engine.authorize_query(
        court_order_id="CO-2026-DE-001",
        agency_id="BKA-DE",
        crime_category="HOMICIDE",
        source_country="DE",
        target_country="NL",
        query_type="cross_border_str_match",
    )
    test("Compliance: valid query authorized", result.authorized)
    test("Compliance: all gates passed", all(result.gate_results.values()))


def test_compliance_reverts_missing_order():
    engine = JurisdictionalComplianceEngine()
    reverted = False
    try:
        engine.authorize_query(
            court_order_id="NONEXISTENT",
            agency_id="BKA-DE",
            crime_category="HOMICIDE",
            source_country="DE",
            target_country="DE",
        )
    except ComplianceRevertError as e:
        reverted = True
        test("Compliance: revert gate is 'court_order'", e.gate == "court_order")

    test("Compliance: missing court order reverts", reverted)


def test_compliance_reverts_unauthorized_crime():
    engine = JurisdictionalComplianceEngine()
    from datetime import datetime, timezone, timedelta
    engine.register_court_order(CourtOrder(
        order_id="CO-2026-AU-001",
        issued_by="Sydney Court",
        agency_id="AFP-AU",
        crime_category="FRAUD",
        issued_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    ))
    reverted = False
    try:
        engine.authorize_query(
            court_order_id="CO-2026-AU-001",
            agency_id="AFP-AU",
            crime_category="FRAUD",
            source_country="AU",
            target_country="AU",
        )
    except ComplianceRevertError as e:
        reverted = True
        test("Compliance: revert gate is 'agency_auth'", e.gate == "agency_auth")

    test("Compliance: unauthorized crime category reverts", reverted)


def test_compliance_reverts_no_treaty():
    engine = JurisdictionalComplianceEngine()
    from datetime import datetime, timezone, timedelta
    engine.register_court_order(CourtOrder(
        order_id="CO-2026-JP-001",
        issued_by="Tokyo District Court",
        agency_id="NPA-JP",
        crime_category="HOMICIDE",
        issued_at=datetime.now(timezone.utc).isoformat(),
        expires_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    ))
    reverted = False
    try:
        engine.authorize_query(
            court_order_id="CO-2026-JP-001",
            agency_id="NPA-JP",
            crime_category="HOMICIDE",
            source_country="JP",
            target_country="DE",
            query_type="cross_border_str_match",
        )
    except ComplianceRevertError as e:
        reverted = True
        test("Compliance: revert gate is 'cross_border'", e.gate == "cross_border")

    test("Compliance: no bilateral treaty reverts", reverted)


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  VANTAGE-STR Phase 5.1 — Ledger & Compliance Tests")
    print("=" * 60)

    print("\n[Merkle Tree]")
    test_merkle_empty()
    test_merkle_single_leaf()
    test_merkle_two_leaves()
    test_merkle_verify()

    print("\n[Forensic Ledger]")
    await test_ledger_append_and_integrity()
    await test_ledger_tamper_detection()

    print("\n[Compliance Engine]")
    test_compliance_authorized()
    test_compliance_reverts_missing_order()
    test_compliance_reverts_unauthorized_crime()
    test_compliance_reverts_no_treaty()

    print("\n" + "=" * 60)
    total = PASS + FAIL
    status = "PASSED" if FAIL == 0 else "FAILED"
    color = "\033[32m" if FAIL == 0 else "\033[31m"
    print(f"  Results: {color}{PASS}/{total} passed\033[0m  ({status})")
    print("=" * 60)

    return FAIL == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
