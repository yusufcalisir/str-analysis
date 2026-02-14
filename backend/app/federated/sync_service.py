"""
Global Sync Service — RTT Monitoring & Latency-Aware Query Routing.

Continuously pings all registered nodes to measure round-trip time (RTT),
stores metrics for historical analysis, and provides a latency-aware
routing table so the Orchestrator can prioritize faster nodes.

Architecture:
    SyncPinger → every 10s → gRPC/HTTP ping to each node → RTT measured
    RTTStore → stores per-node latency windows → rolling averages
    LatencyRouter → sorts nodes by avg RTT → fan-out priority ordering

Metrics Persistence:
    In production, metrics are written to PostgreSQL via async batch inserts.
    Currently uses an in-memory ring buffer per node (last 100 measurements).
"""

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

PING_INTERVAL_SECONDS: float = 10.0
PING_TIMEOUT_SECONDS: float = 5.0
RTT_WINDOW_SIZE: int = 100
BOTTLENECK_THRESHOLD_MS: float = 500.0
UNREACHABLE_THRESHOLD_MS: float = 2500.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class NodeHealth(str, Enum):
    """Health state derived from RTT measurements."""
    HEALTHY = "healthy"          # RTT < 200ms
    DEGRADED = "degraded"        # RTT 200–500ms
    BOTTLENECK = "bottleneck"    # RTT 500–2500ms
    UNREACHABLE = "unreachable"  # RTT > 2500ms or timeout


class PingResult(BaseModel):
    """Result of a single RTT ping."""
    node_id: str
    rtt_ms: float
    success: bool
    timestamp: float = Field(default_factory=time.time)
    error: str = ""


class NodeLatencyProfile(BaseModel):
    """Aggregated latency profile for a single node."""
    node_id: str
    avg_rtt_ms: float = 0.0
    min_rtt_ms: float = 0.0
    max_rtt_ms: float = 0.0
    p50_rtt_ms: float = 0.0
    p95_rtt_ms: float = 0.0
    p99_rtt_ms: float = 0.0
    jitter_ms: float = 0.0
    packet_loss_pct: float = 0.0
    health: NodeHealth = NodeHealth.HEALTHY
    sample_count: int = 0
    last_ping: float = 0.0


class SyncTelemetry(BaseModel):
    """Network-wide sync telemetry for the dashboard."""
    avg_global_rtt_ms: float = 0.0
    active_tunnels: int = 0
    total_pings_sent: int = 0
    total_pings_failed: int = 0
    bottleneck_nodes: List[str] = Field(default_factory=list)
    unreachable_nodes: List[str] = Field(default_factory=list)
    healthiest_node: str = ""
    slowest_node: str = ""
    last_sweep: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# RTT STORE
# ═══════════════════════════════════════════════════════════════════════════════

class RTTStore:
    """
    Per-node ring buffer for RTT measurements.

    Stores the last RTT_WINDOW_SIZE ping results for each node and
    computes rolling statistics (mean, min, max, percentiles, jitter).

    In production, these measurements are also batch-flushed to
    PostgreSQL for long-term analysis and alerting.
    """

    def __init__(self, window_size: int = RTT_WINDOW_SIZE) -> None:
        self._window_size = window_size
        self._buffers: Dict[str, List[PingResult]] = {}
        self._total_pings: int = 0
        self._total_failures: int = 0

    def record(self, result: PingResult) -> None:
        """Record a ping result for a node."""
        if result.node_id not in self._buffers:
            self._buffers[result.node_id] = []

        buf = self._buffers[result.node_id]
        buf.append(result)

        # Ring buffer — evict oldest if over capacity
        if len(buf) > self._window_size:
            buf.pop(0)

        self._total_pings += 1
        if not result.success:
            self._total_failures += 1

    def get_profile(self, node_id: str) -> NodeLatencyProfile:
        """Compute latency profile from buffered measurements."""
        buf = self._buffers.get(node_id, [])

        if not buf:
            return NodeLatencyProfile(
                node_id=node_id,
                health=NodeHealth.UNREACHABLE,
            )

        successful = [p.rtt_ms for p in buf if p.success]
        total = len(buf)
        failures = total - len(successful)

        if not successful:
            return NodeLatencyProfile(
                node_id=node_id,
                packet_loss_pct=100.0,
                health=NodeHealth.UNREACHABLE,
                sample_count=total,
                last_ping=buf[-1].timestamp,
            )

        s = sorted(successful)
        n = len(s)
        avg = sum(s) / n
        jitter = max(s) - min(s) if n > 1 else 0.0
        loss_pct = (failures / total) * 100 if total > 0 else 0.0

        # Classify health
        if avg > UNREACHABLE_THRESHOLD_MS or loss_pct > 80:
            health = NodeHealth.UNREACHABLE
        elif avg > BOTTLENECK_THRESHOLD_MS or loss_pct > 30:
            health = NodeHealth.BOTTLENECK
        elif avg > 200:
            health = NodeHealth.DEGRADED
        else:
            health = NodeHealth.HEALTHY

        return NodeLatencyProfile(
            node_id=node_id,
            avg_rtt_ms=round(avg, 2),
            min_rtt_ms=round(s[0], 2),
            max_rtt_ms=round(s[-1], 2),
            p50_rtt_ms=round(s[n // 2], 2),
            p95_rtt_ms=round(s[int(n * 0.95)], 2),
            p99_rtt_ms=round(s[min(int(n * 0.99), n - 1)], 2),
            jitter_ms=round(jitter, 2),
            packet_loss_pct=round(loss_pct, 2),
            health=health,
            sample_count=total,
            last_ping=buf[-1].timestamp,
        )

    def get_all_profiles(self) -> List[NodeLatencyProfile]:
        """Compute profiles for all tracked nodes."""
        return [self.get_profile(nid) for nid in self._buffers]

    def get_routing_order(self) -> List[str]:
        """
        Return node_ids sorted by average RTT (fastest first).

        Used by the Orchestrator to prioritize fan-out order so
        faster nodes are queried first, reducing tail latency.
        """
        profiles = self.get_all_profiles()
        reachable = [p for p in profiles if p.health != NodeHealth.UNREACHABLE]
        reachable.sort(key=lambda p: p.avg_rtt_ms)
        return [p.node_id for p in reachable]

    @property
    def total_pings(self) -> int:
        return self._total_pings

    @property
    def total_failures(self) -> int:
        return self._total_failures

    @property
    def node_count(self) -> int:
        return len(self._buffers)


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC PINGER
# ═══════════════════════════════════════════════════════════════════════════════

class SyncPinger:
    """
    Asynchronous RTT pinger for all registered VANTAGE nodes.

    Runs a background loop that pings every node at PING_INTERVAL_SECONDS
    intervals, recording results in the RTTStore. Supports both real
    gRPC pings and simulated pings for development.

    Usage:
        pinger = SyncPinger(rtt_store, get_nodes_fn)
        await pinger.start()   # Starts background loop
        pinger.stop()          # Graceful shutdown
    """

    def __init__(
        self,
        rtt_store: RTTStore,
        get_node_ids: Callable[[], List[str]],
        ping_fn: Optional[Callable[[str], Coroutine[Any, Any, PingResult]]] = None,
    ) -> None:
        """
        Args:
            rtt_store: Where to store ping results.
            get_node_ids: Callable returning current node IDs.
            ping_fn: Optional async function to actually ping a node.
                     If None, simulates pings for development.
        """
        self._store = rtt_store
        self._get_nodes = get_node_ids
        self._ping_fn = ping_fn
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sweep_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def sweep_count(self) -> int:
        return self._sweep_count

    async def start(self) -> None:
        """Start the background ping loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._ping_loop())
        logger.info("[SYNC] Pinger started | interval=%.1fs", PING_INTERVAL_SECONDS)

    def stop(self) -> None:
        """Stop the background ping loop."""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("[SYNC] Pinger stopped")

    async def ping_all(self) -> List[PingResult]:
        """Execute one ping sweep across all nodes."""
        node_ids = self._get_nodes()
        if not node_ids:
            return []

        # Ping all nodes concurrently
        tasks = [self._ping_node(nid) for nid in node_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ping_results: List[PingResult] = []
        for r in results:
            if isinstance(r, PingResult):
                self._store.record(r)
                ping_results.append(r)
            elif isinstance(r, Exception):
                logger.error(f"[SYNC] Ping exception: {r}")

        self._sweep_count += 1
        return ping_results

    async def _ping_loop(self) -> None:
        """Background loop that pings all nodes periodically."""
        while self._running:
            try:
                await self.ping_all()
            except Exception as exc:
                logger.error(f"[SYNC] Ping sweep error: {exc}")
            await asyncio.sleep(PING_INTERVAL_SECONDS)

    async def _ping_node(self, node_id: str) -> PingResult:
        """Ping a single node and return the result."""
        if self._ping_fn:
            try:
                return await asyncio.wait_for(
                    self._ping_fn(node_id),
                    timeout=PING_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                return PingResult(
                    node_id=node_id,
                    rtt_ms=PING_TIMEOUT_SECONDS * 1000,
                    success=False,
                    error="Timeout",
                )
            except Exception as exc:
                return PingResult(
                    node_id=node_id,
                    rtt_ms=0.0,
                    success=False,
                    error=str(exc),
                )

        # Simulation mode
        return self._simulate_ping(node_id)

    def _simulate_ping(self, node_id: str) -> PingResult:
        """Generate a realistic simulated ping result."""
        # Base latency varies by "geographic distance" (hash-based)
        base = (hash(node_id) % 150) + 20  # 20–170ms base
        jitter = random.gauss(0, base * 0.15)
        rtt = max(5.0, base + jitter)

        # Occasional packet loss (2% chance)
        success = random.random() > 0.02

        return PingResult(
            node_id=node_id,
            rtt_ms=round(rtt, 2) if success else 0.0,
            success=success,
            error="" if success else "Simulated packet loss",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SYNC SERVICE
# ═══════════════════════════════════════════════════════════════════════════════

class SyncService:
    """
    Top-level service combining RTT monitoring and latency-aware routing.

    Provides the Orchestrator with:
        - Real-time node health via continuous pinging.
        - Latency-sorted routing tables for fan-out optimization.
        - Network telemetry for the GlobalPulse dashboard.

    Usage:
        sync = SyncService()
        sync.register_node("EUROPOL-NL")
        sync.register_node("FBI-US-DC")
        await sync.start()  # Starts background pinger

        # Get routing priority
        order = sync.get_routing_order()

        # Get telemetry for UI
        telemetry = sync.get_telemetry()
    """

    def __init__(self) -> None:
        self._store = RTTStore()
        self._node_ids: List[str] = []
        self._pinger = SyncPinger(
            rtt_store=self._store,
            get_node_ids=lambda: self._node_ids,
        )

    def register_node(self, node_id: str) -> None:
        """Register a node for RTT monitoring."""
        if node_id not in self._node_ids:
            self._node_ids.append(node_id)

    def unregister_node(self, node_id: str) -> None:
        """Remove a node from monitoring."""
        if node_id in self._node_ids:
            self._node_ids.remove(node_id)

    async def start(self) -> None:
        """Start the background pinger."""
        await self._pinger.start()

    def stop(self) -> None:
        """Stop the background pinger."""
        self._pinger.stop()

    async def ping_all_now(self) -> List[PingResult]:
        """Execute an immediate ping sweep (on-demand)."""
        return await self._pinger.ping_all()

    def get_routing_order(self) -> List[str]:
        """Return node IDs sorted by latency (fastest first)."""
        return self._store.get_routing_order()

    def get_node_profile(self, node_id: str) -> NodeLatencyProfile:
        """Get latency profile for a specific node."""
        return self._store.get_profile(node_id)

    def get_all_profiles(self) -> List[NodeLatencyProfile]:
        """Get latency profiles for all nodes."""
        return self._store.get_all_profiles()

    def get_telemetry(self) -> SyncTelemetry:
        """Aggregate network telemetry for the dashboard."""
        profiles = self._store.get_all_profiles()

        if not profiles:
            return SyncTelemetry()

        reachable = [p for p in profiles if p.health != NodeHealth.UNREACHABLE]
        avg_rtt = (
            sum(p.avg_rtt_ms for p in reachable) / len(reachable)
            if reachable else 0.0
        )

        bottlenecks = [p.node_id for p in profiles if p.health == NodeHealth.BOTTLENECK]
        unreachable = [p.node_id for p in profiles if p.health == NodeHealth.UNREACHABLE]
        healthiest = min(reachable, key=lambda p: p.avg_rtt_ms).node_id if reachable else ""
        slowest = max(reachable, key=lambda p: p.avg_rtt_ms).node_id if reachable else ""

        return SyncTelemetry(
            avg_global_rtt_ms=round(avg_rtt, 2),
            active_tunnels=len(reachable),
            total_pings_sent=self._store.total_pings,
            total_pings_failed=self._store.total_failures,
            bottleneck_nodes=bottlenecks,
            unreachable_nodes=unreachable,
            healthiest_node=healthiest,
            slowest_node=slowest,
            last_sweep=time.time(),
        )
