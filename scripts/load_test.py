"""
Load Test — Async Batch Uploader for VANTAGE-STR Pipeline Stress Testing.

Generates synthetic genomic profiles and pushes them to the FastAPI
/profile/ingest endpoint in parallel, measuring per-stage latency
for serialization, validation, and vectorization.

Usage:
    python scripts/load_test.py --count 10000 --concurrency 50
    python scripts/load_test.py --count 100 --dry-run
"""

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import aiohttp
except ImportError:
    aiohttp = None

import numpy as np

from scripts.synthetic_genetics_engine import SyntheticGeneticsEngine, SyntheticProfile


# ═══════════════════════════════════════════════════════════════════════════════
# TIMING & METRICS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class IngestMetrics:
    """Accumulated timing metrics for the load test."""
    serialization_ms: List[float] = field(default_factory=list)
    request_ms: List[float] = field(default_factory=list)
    total_sent: int = 0
    total_accepted: int = 0
    total_quarantined: int = 0
    total_rejected: int = 0
    total_failed: int = 0
    errors: List[str] = field(default_factory=list)

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        ser = np.array(self.serialization_ms) if self.serialization_ms else np.array([0.0])
        req = np.array(self.request_ms) if self.request_ms else np.array([0.0])
        return {
            "total_sent": self.total_sent,
            "total_accepted": self.total_accepted,
            "total_quarantined": self.total_quarantined,
            "total_rejected": self.total_rejected,
            "total_failed": self.total_failed,
            "serialization_ms": {
                "mean": float(np.mean(ser)),
                "p50": float(np.percentile(ser, 50)),
                "p95": float(np.percentile(ser, 95)),
                "p99": float(np.percentile(ser, 99)),
            },
            "request_ms": {
                "mean": float(np.mean(req)),
                "p50": float(np.percentile(req, 50)),
                "p95": float(np.percentile(req, 95)),
                "p99": float(np.percentile(req, 99)),
            },
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ASYNC UPLOADER
# ═══════════════════════════════════════════════════════════════════════════════

async def upload_profile(
    session: "aiohttp.ClientSession",
    profile: SyntheticProfile,
    base_url: str,
    metrics: IngestMetrics,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Upload a single profile to the ingestion endpoint.

    Measures serialization time (dict→JSON) and round-trip request latency
    separately for pipeline stage profiling.
    """
    async with semaphore:
        # ── Serialization timing ──
        t_ser_start = time.perf_counter()
        payload = profile.to_ingest_payload()
        json_body = json.dumps(payload)
        t_ser_end = time.perf_counter()

        ser_ms = (t_ser_end - t_ser_start) * 1000
        metrics.serialization_ms.append(ser_ms)

        # ── Request timing ──
        url = f"{base_url}/profile/ingest"
        t_req_start = time.perf_counter()

        try:
            async with session.post(
                url,
                data=json_body,
                headers={"Content-Type": "application/json"},
            ) as response:
                t_req_end = time.perf_counter()
                req_ms = (t_req_end - t_req_start) * 1000
                metrics.request_ms.append(req_ms)
                metrics.total_sent += 1

                if response.status == 201:
                    body = await response.json()
                    decision = body.get("decision", "ACCEPTED")
                    if decision == "ACCEPTED":
                        metrics.total_accepted += 1
                    elif decision == "QUARANTINED":
                        metrics.total_quarantined += 1
                elif response.status == 403:
                    metrics.total_rejected += 1
                else:
                    metrics.total_failed += 1
                    text = await response.text()
                    metrics.errors.append(f"HTTP {response.status}: {text[:200]}")

        except Exception as exc:
            t_req_end = time.perf_counter()
            metrics.request_ms.append((t_req_end - t_req_start) * 1000)
            metrics.total_sent += 1
            metrics.total_failed += 1
            metrics.errors.append(str(exc)[:200])


async def run_load_test(
    profiles: List[SyntheticProfile],
    base_url: str,
    concurrency: int,
    metrics: IngestMetrics,
) -> None:
    """
    Execute the async load test across all profiles.

    Uses a semaphore to limit concurrent connections to the configured
    concurrency level, preventing server overload.
    """
    if aiohttp is None:
        print("ERROR: aiohttp is required. Install with: pip install aiohttp")
        return

    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=concurrency)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            upload_profile(session, p, base_url, metrics, semaphore)
            for p in profiles
        ]

        # Progress reporting
        total = len(tasks)
        completed = 0

        for batch_start in range(0, total, concurrency * 2):
            batch_end = min(batch_start + concurrency * 2, total)
            batch = tasks[batch_start:batch_end]
            await asyncio.gather(*batch)
            completed += len(batch)

            pct = (completed / total) * 100
            print(
                f"\r  ▸ Progress: {completed:>6,}/{total:>6,} ({pct:5.1f}%) "
                f"| Accepted: {metrics.total_accepted:>5,} "
                f"| Quarantined: {metrics.total_quarantined:>3,} "
                f"| Failed: {metrics.total_failed:>3,}",
                end="", flush=True,
            )

        print()  # Newline after progress bar


# ═══════════════════════════════════════════════════════════════════════════════
# DRY RUN (no server required)
# ═══════════════════════════════════════════════════════════════════════════════

def dry_run(profiles: List[SyntheticProfile]) -> None:
    """
    Validate serialization without sending requests.

    Useful for testing the generator and measuring serialization
    throughput independently of network latency.
    """
    print("  ▸ Dry run — serializing profiles without sending...")

    t0 = time.perf_counter()
    for p in profiles:
        payload = p.to_ingest_payload()
        _ = json.dumps(payload)
    elapsed = time.perf_counter() - t0

    print(f"  ▸ Serialized {len(profiles):,} profiles in {elapsed*1000:.1f} ms")
    print(f"  ▸ Throughput: {len(profiles)/elapsed:,.0f} serializations/sec")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="VANTAGE-STR Load Test")
    parser.add_argument("--count", type=int, default=10000, help="Number of profiles")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--population", type=str, default=None, help="Fixed population (or mixed)")
    parser.add_argument("--noisy-ratio", type=float, default=0.15, help="Fraction of noisy profiles")
    parser.add_argument("--sibling-ratio", type=float, default=0.05, help="Fraction of sibling profiles")
    parser.add_argument("--dry-run", action="store_true", help="Serialize only, no network")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  VANTAGE-STR  Load Test Runner                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Profiles:    {args.count:>8,}                            ║")
    print(f"║  Concurrency: {args.concurrency:>8}                            ║")
    print(f"║  Target:      {args.base_url:<38s} ║")
    print(f"║  Seed:        {args.seed:>8}                            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Generate ──
    print("─── Phase 1: Generating Synthetic Profiles ───")
    engine = SyntheticGeneticsEngine(seed=args.seed)
    t_gen_start = time.perf_counter()
    profiles = engine.generate_batch(
        count=args.count,
        population=args.population,
        noisy_ratio=args.noisy_ratio,
        sibling_ratio=args.sibling_ratio,
    )
    t_gen_end = time.perf_counter()
    gen_ms = (t_gen_end - t_gen_start) * 1000

    types = {"clean": 0, "noisy": 0, "sibling": 0}
    for p in profiles:
        types[p.profile_type] += 1

    print(f"  ▸ Generated {len(profiles):,} profiles in {gen_ms:.1f} ms")
    print(f"  ▸ Clean: {types['clean']:,}  Noisy: {types['noisy']:,}  Sibling: {types['sibling']:,}")
    print()

    if args.dry_run:
        dry_run(profiles)
        return

    # ── Upload ──
    print("─── Phase 2: Async Batch Upload ───")
    metrics = IngestMetrics()
    t_upload_start = time.perf_counter()
    asyncio.run(run_load_test(profiles, args.base_url, args.concurrency, metrics))
    t_upload_end = time.perf_counter()
    upload_ms = (t_upload_end - t_upload_start) * 1000

    # ── Report ──
    print()
    print("─── Results ───")
    summary = metrics.summary()

    print(f"  Total upload time:  {upload_ms:>10,.1f} ms")
    print(f"  Profiles sent:      {summary['total_sent']:>10,}")
    print(f"  Accepted:           {summary['total_accepted']:>10,}")
    print(f"  Quarantined:        {summary['total_quarantined']:>10,}")
    print(f"  Rejected:           {summary['total_rejected']:>10,}")
    print(f"  Failed:             {summary['total_failed']:>10,}")
    print()
    print("  Serialization Latency:")
    ser = summary["serialization_ms"]
    print(f"    Mean:  {ser['mean']:>8.2f} ms")
    print(f"    P50:   {ser['p50']:>8.2f} ms")
    print(f"    P95:   {ser['p95']:>8.2f} ms")
    print(f"    P99:   {ser['p99']:>8.2f} ms")
    print()
    print("  Request Latency:")
    req = summary["request_ms"]
    print(f"    Mean:  {req['mean']:>8.2f} ms")
    print(f"    P50:   {req['p50']:>8.2f} ms")
    print(f"    P95:   {req['p95']:>8.2f} ms")
    print(f"    P99:   {req['p99']:>8.2f} ms")

    if metrics.errors:
        print()
        print(f"  Errors ({len(metrics.errors)}):")
        for err in metrics.errors[:5]:
            print(f"    ▸ {err}")
        if len(metrics.errors) > 5:
            print(f"    ... and {len(metrics.errors) - 5} more")


# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK JITTER SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class JitterScenario:
    """Definition of a network jitter test scenario."""
    name: str
    description: str
    latency_range_ms: tuple  # (min, max)
    packet_loss_pct: float
    malformed_pct: float
    timeout_pct: float


@dataclass
class JitterMetrics:
    """Aggregated metrics from jitter testing."""
    scenario: str = ""
    total_broadcasts: int = 0
    total_node_queries: int = 0
    nodes_responded: int = 0
    nodes_timed_out: int = 0
    nodes_errored: int = 0
    total_matches: int = 0
    broadcast_times_ms: List[float] = field(default_factory=list)
    response_times_ms: List[float] = field(default_factory=list)
    malformed_injected: int = 0
    timeouts_injected: int = 0

    def summary(self) -> Dict[str, Any]:
        bt = np.array(self.broadcast_times_ms) if self.broadcast_times_ms else np.array([0.0])
        rt = np.array(self.response_times_ms) if self.response_times_ms else np.array([0.0])
        return {
            "scenario": self.scenario,
            "total_broadcasts": self.total_broadcasts,
            "total_node_queries": self.total_node_queries,
            "nodes_responded": self.nodes_responded,
            "nodes_timed_out": self.nodes_timed_out,
            "nodes_errored": self.nodes_errored,
            "total_matches": self.total_matches,
            "malformed_injected": self.malformed_injected,
            "timeouts_injected": self.timeouts_injected,
            "broadcast_ms": {
                "mean": float(np.mean(bt)),
                "p50": float(np.percentile(bt, 50)),
                "p95": float(np.percentile(bt, 95)),
                "p99": float(np.percentile(bt, 99)),
            },
            "response_ms": {
                "mean": float(np.mean(rt)),
                "p50": float(np.percentile(rt, 50)),
                "p95": float(np.percentile(rt, 95)),
                "p99": float(np.percentile(rt, 99)),
            },
        }


JITTER_SCENARIOS: List[JitterScenario] = [
    JitterScenario(
        name="nominal",
        description="Baseline — normal network conditions",
        latency_range_ms=(20, 150),
        packet_loss_pct=0.0,
        malformed_pct=0.0,
        timeout_pct=0.0,
    ),
    JitterScenario(
        name="mild_jitter",
        description="Mild jitter — occasional latency spikes",
        latency_range_ms=(30, 400),
        packet_loss_pct=2.0,
        malformed_pct=0.0,
        timeout_pct=1.0,
    ),
    JitterScenario(
        name="heavy_jitter",
        description="Heavy jitter — unstable transcontinental links",
        latency_range_ms=(50, 1500),
        packet_loss_pct=8.0,
        malformed_pct=1.0,
        timeout_pct=5.0,
    ),
    JitterScenario(
        name="adversarial",
        description="Adversarial — compromised/malicious node behavior",
        latency_range_ms=(100, 3000),
        packet_loss_pct=15.0,
        malformed_pct=10.0,
        timeout_pct=12.0,
    ),
    JitterScenario(
        name="catastrophic",
        description="Catastrophic — network near failure (DDoS / cable cut)",
        latency_range_ms=(500, 5000),
        packet_loss_pct=40.0,
        malformed_pct=20.0,
        timeout_pct=30.0,
    ),
]


async def simulate_jittery_node(
    node_id: str,
    embedding: List[float],
    scenario: JitterScenario,
    metrics: JitterMetrics,
) -> "NodeQueryResult":
    """
    Simulate a node response with configurable network impairments.

    Applies the scenario's latency range, packet loss probability,
    malformed packet injection, and timeout probability to create
    realistic degraded network conditions.
    """
    import random as rng

    # Late import to avoid circular dependency issues
    from backend.app.federated.orchestrator import NodeQueryResult

    # Determine impairment type
    roll = rng.random() * 100

    # Timeout injection
    if roll < scenario.timeout_pct:
        metrics.timeouts_injected += 1
        await asyncio.sleep(3.0)  # Will be caught by broadcast_query timeout
        return NodeQueryResult(
            node_id=node_id,
            responded=False,
            timed_out=True,
            error="Injected timeout",
            response_time_ms=3000.0,
        )

    # Malformed packet injection
    if roll < (scenario.timeout_pct + scenario.malformed_pct):
        metrics.malformed_injected += 1
        latency = rng.uniform(*scenario.latency_range_ms) / 1000
        await asyncio.sleep(latency)
        raise ValueError(
            f"Malformed gRPC packet from {node_id}: "
            f"corrupted header bytes"
        )

    # Packet loss
    if roll < (scenario.timeout_pct + scenario.malformed_pct + scenario.packet_loss_pct):
        latency = rng.uniform(*scenario.latency_range_ms) / 1000
        await asyncio.sleep(latency)
        return NodeQueryResult(
            node_id=node_id,
            responded=False,
            error="Connection reset by peer",
            response_time_ms=latency * 1000,
        )

    # Normal response with jittery latency
    latency = rng.uniform(*scenario.latency_range_ms) / 1000
    await asyncio.sleep(latency)

    n_matches = rng.randint(0, 3)
    import secrets
    matches = [
        {
            "local_reference_token": f"LRT-{secrets.token_hex(12)}",
            "match_score": round(rng.uniform(0.70, 0.99), 4),
        }
        for _ in range(n_matches)
    ]

    return NodeQueryResult(
        node_id=node_id,
        matches=matches,
        responded=True,
        response_time_ms=round(latency * 1000, 2),
    )


async def run_jitter_test(
    num_broadcasts: int = 20,
    num_nodes: int = 8,
) -> None:
    """
    Execute the full jitter stress test suite.

    Registers simulated nodes in a NodeManager, then runs broadcast_query
    through each JitterScenario to measure how the orchestrator handles
    degraded network conditions.
    """
    from backend.app.federated.orchestrator import (
        NodeManager,
        HandshakeRequest,
        broadcast_query,
        QueryBroadcastConfig,
    )
    import secrets

    NODE_CONFIGS = [
        ("EUROPOL-NL", "NL", 52.07, 4.30),
        ("FBI-US-DC", "US", 38.91, -77.04),
        ("NCA-UK", "GB", 51.51, -0.13),
        ("BKA-DE", "DE", 50.73, 7.10),
        ("DGPN-FR", "FR", 48.86, 2.35),
        ("POLIZIA-IT", "IT", 41.90, 12.50),
        ("AFP-AU", "AU", -35.28, 149.13),
        ("NPA-JP", "JP", 35.68, 139.69),
    ][:num_nodes]

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  VANTAGE-STR  Network Jitter Stress Test            ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Nodes:       {num_nodes:>8}                            ║")
    print(f"║  Broadcasts:  {num_broadcasts:>8}  per scenario             ║")
    print(f"║  Scenarios:   {len(JITTER_SCENARIOS):>8}                            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Register nodes
    manager = NodeManager()
    for nid, cc, lat, lng in NODE_CONFIGS:
        cert = f"-----BEGIN CERTIFICATE-----\n{secrets.token_hex(64)}\n-----END CERTIFICATE-----"
        manager.handle_handshake(HandshakeRequest(
            node_id=nid,
            display_name=nid,
            country_code=cc,
            cert_pem=cert,
            ip_address="127.0.0.1",
            latitude=lat,
            longitude=lng,
        ))

    dummy_embedding = [0.01] * 48

    for scenario in JITTER_SCENARIOS:
        print(f"─── Scenario: {scenario.name} ───")
        print(f"    {scenario.description}")
        print(f"    Latency: {scenario.latency_range_ms[0]}–{scenario.latency_range_ms[1]}ms | "
              f"Loss: {scenario.packet_loss_pct}% | "
              f"Malformed: {scenario.malformed_pct}% | "
              f"Timeout: {scenario.timeout_pct}%")

        metrics = JitterMetrics(scenario=scenario.name)

        async def query_fn(node_id: str, embedding: List[float]):
            return await simulate_jittery_node(node_id, embedding, scenario, metrics)

        t_start = time.perf_counter()

        for i in range(num_broadcasts):
            import uuid
            result = await broadcast_query(
                manager=manager,
                query_id=f"jitter-{scenario.name}-{i}",
                query_embedding=dummy_embedding,
                config=QueryBroadcastConfig(timeout_seconds=2.5),
                query_node_fn=query_fn,
            )
            metrics.total_broadcasts += 1
            metrics.total_node_queries += result.total_nodes_queried
            metrics.nodes_responded += result.nodes_responded
            metrics.nodes_timed_out += result.nodes_timed_out
            metrics.nodes_errored += result.nodes_errored
            metrics.total_matches += result.total_matches
            metrics.broadcast_times_ms.append(result.total_time_ms)
            for nr in result.node_results:
                if nr.responded:
                    metrics.response_times_ms.append(nr.response_time_ms)

            pct = ((i + 1) / num_broadcasts) * 100
            print(f"\r    ▸ Progress: {i+1:>3}/{num_broadcasts} ({pct:5.1f}%)"
                  f" | responded: {metrics.nodes_responded}"
                  f" | timed_out: {metrics.nodes_timed_out}"
                  f" | errored: {metrics.nodes_errored}", end="", flush=True)

        elapsed = (time.perf_counter() - t_start) * 1000
        print()

        s = metrics.summary()
        print(f"    Elapsed:      {elapsed:>10,.1f} ms")
        print(f"    Broadcasts:   {s['total_broadcasts']:>10}")
        print(f"    Node Queries: {s['total_node_queries']:>10}")
        print(f"    Responded:    {s['nodes_responded']:>10}")
        print(f"    Timed Out:    {s['nodes_timed_out']:>10}")
        print(f"    Errored:      {s['nodes_errored']:>10}")
        print(f"    Matches:      {s['total_matches']:>10}")
        print(f"    Malformed:    {s['malformed_injected']:>10}")
        print(f"    Timeouts Inj: {s['timeouts_injected']:>10}")
        bc = s["broadcast_ms"]
        print(f"    Broadcast Latency:  mean={bc['mean']:.1f}ms  P50={bc['p50']:.1f}ms  P95={bc['p95']:.1f}ms  P99={bc['p99']:.1f}ms")
        if s["response_ms"]["mean"] > 0:
            rr = s["response_ms"]
            print(f"    Response Latency:   mean={rr['mean']:.1f}ms  P50={rr['p50']:.1f}ms  P95={rr['p95']:.1f}ms  P99={rr['p99']:.1f}ms")
        print()

    print("═══ Jitter Stress Test Complete ═══")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="VANTAGE-STR Load Test")
    parser.add_argument("--count", type=int, default=10000, help="Number of profiles")
    parser.add_argument("--concurrency", type=int, default=50, help="Max concurrent requests")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000", help="API base URL")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--population", type=str, default=None, help="Fixed population (or mixed)")
    parser.add_argument("--noisy-ratio", type=float, default=0.15, help="Fraction of noisy profiles")
    parser.add_argument("--sibling-ratio", type=float, default=0.05, help="Fraction of sibling profiles")
    parser.add_argument("--dry-run", action="store_true", help="Serialize only, no network")
    parser.add_argument("--jitter", action="store_true", help="Run network jitter stress test")
    parser.add_argument("--jitter-broadcasts", type=int, default=20, help="Broadcasts per jitter scenario")
    parser.add_argument("--jitter-nodes", type=int, default=8, help="Number of simulated nodes")
    args = parser.parse_args()

    if args.jitter:
        asyncio.run(run_jitter_test(
            num_broadcasts=args.jitter_broadcasts,
            num_nodes=args.jitter_nodes,
        ))
        return

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  VANTAGE-STR  Load Test Runner                      ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Profiles:    {args.count:>8,}                            ║")
    print(f"║  Concurrency: {args.concurrency:>8}                            ║")
    print(f"║  Target:      {args.base_url:<38s} ║")
    print(f"║  Seed:        {args.seed:>8}                            ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # ── Generate ──
    print("─── Phase 1: Generating Synthetic Profiles ───")
    engine = SyntheticGeneticsEngine(seed=args.seed)
    t_gen_start = time.perf_counter()
    profiles = engine.generate_batch(
        count=args.count,
        population=args.population,
        noisy_ratio=args.noisy_ratio,
        sibling_ratio=args.sibling_ratio,
    )
    t_gen_end = time.perf_counter()
    gen_ms = (t_gen_end - t_gen_start) * 1000

    types = {"clean": 0, "noisy": 0, "sibling": 0}
    for p in profiles:
        types[p.profile_type] += 1

    print(f"  ▸ Generated {len(profiles):,} profiles in {gen_ms:.1f} ms")
    print(f"  ▸ Clean: {types['clean']:,}  Noisy: {types['noisy']:,}  Sibling: {types['sibling']:,}")
    print()

    if args.dry_run:
        dry_run(profiles)
        return

    # ── Upload ──
    print("─── Phase 2: Async Batch Upload ───")
    metrics = IngestMetrics()
    t_upload_start = time.perf_counter()
    asyncio.run(run_load_test(profiles, args.base_url, args.concurrency, metrics))
    t_upload_end = time.perf_counter()
    upload_ms = (t_upload_end - t_upload_start) * 1000

    # ── Report ──
    print()
    print("─── Results ───")
    summary = metrics.summary()

    print(f"  Total upload time:  {upload_ms:>10,.1f} ms")
    print(f"  Profiles sent:      {summary['total_sent']:>10,}")
    print(f"  Accepted:           {summary['total_accepted']:>10,}")
    print(f"  Quarantined:        {summary['total_quarantined']:>10,}")
    print(f"  Rejected:           {summary['total_rejected']:>10,}")
    print(f"  Failed:             {summary['total_failed']:>10,}")
    print()
    print("  Serialization Latency:")
    ser = summary["serialization_ms"]
    print(f"    Mean:  {ser['mean']:>8.2f} ms")
    print(f"    P50:   {ser['p50']:>8.2f} ms")
    print(f"    P95:   {ser['p95']:>8.2f} ms")
    print(f"    P99:   {ser['p99']:>8.2f} ms")
    print()
    print("  Request Latency:")
    req = summary["request_ms"]
    print(f"    Mean:  {req['mean']:>8.2f} ms")
    print(f"    P50:   {req['p50']:>8.2f} ms")
    print(f"    P95:   {req['p95']:>8.2f} ms")
    print(f"    P99:   {req['p99']:>8.2f} ms")

    if metrics.errors:
        print()
        print(f"  Errors ({len(metrics.errors)}):")
        for err in metrics.errors[:5]:
            print(f"    ▸ {err}")
        if len(metrics.errors) > 5:
            print(f"    ... and {len(metrics.errors) - 5} more")


if __name__ == "__main__":
    main()
