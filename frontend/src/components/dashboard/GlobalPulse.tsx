"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import {
    Activity,
    Shield,
    Zap,
    Globe,
    Lock,
    Radio,
    Clock,
    TrendingUp,
    AlertTriangle,
    CheckCircle,
    XCircle,
    Wifi,
    WifiOff,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface NodePulse {
    id: string;
    label: string;
    countryCode: string;
    lat: number;
    lng: number;
    avgRttMs: number;
    health: "healthy" | "degraded" | "bottleneck" | "unreachable";
    activeTunnel: boolean;
    lastPing: number;
    packetLoss: number;
}

interface BroadcastWave {
    id: string;
    sourceNodeId: string;
    targetNodeIds: string[];
    startedAt: number;
    progress: number; // 0–1
}

interface PulseStats {
    avgGlobalRtt: number;
    activeTunnels: number;
    totalPings: number;
    totalFailures: number;
    bottleneckCount: number;
    healthiestNode: string;
    slowestNode: string;
}

// ─── Mock Data ──────────────────────────────────────────────────────────────

const NODES: NodePulse[] = [
    { id: "EUROPOL-NL", label: "EUROPOL", countryCode: "NL", lat: 52.07, lng: 4.30, avgRttMs: 42, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 0 },
    { id: "FBI-US-DC", label: "FBI", countryCode: "US", lat: 38.91, lng: -77.04, avgRttMs: 128, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 1.2 },
    { id: "NCA-UK", label: "NCA", countryCode: "GB", lat: 51.51, lng: -0.13, avgRttMs: 38, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 0 },
    { id: "BKA-DE", label: "BKA", countryCode: "DE", lat: 50.73, lng: 7.10, avgRttMs: 35, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 0.5 },
    { id: "DGPN-FR", label: "DGPN", countryCode: "FR", lat: 48.86, lng: 2.35, avgRttMs: 45, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 0 },
    { id: "POLIZIA-IT", label: "Polizia", countryCode: "IT", lat: 41.90, lng: 12.50, avgRttMs: 52, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 0 },
    { id: "AFP-AU", label: "AFP", countryCode: "AU", lat: -35.28, lng: 149.13, avgRttMs: 285, health: "degraded", activeTunnel: true, lastPing: Date.now(), packetLoss: 3.1 },
    { id: "NPA-JP", label: "NPA", countryCode: "JP", lat: 35.68, lng: 139.69, avgRttMs: 195, health: "degraded", activeTunnel: true, lastPing: Date.now(), packetLoss: 1.8 },
    { id: "SAPS-ZA", label: "SAPS", countryCode: "ZA", lat: -33.93, lng: 18.42, avgRttMs: 310, health: "degraded", activeTunnel: true, lastPing: Date.now(), packetLoss: 4.2 },
    { id: "PF-BR", label: "PF", countryCode: "BR", lat: -15.79, lng: -47.88, avgRttMs: 520, health: "bottleneck", activeTunnel: true, lastPing: Date.now(), packetLoss: 8.5 },
    { id: "RCMP-CA", label: "RCMP", countryCode: "CA", lat: 45.42, lng: -75.70, avgRttMs: 115, health: "healthy", activeTunnel: true, lastPing: Date.now(), packetLoss: 0.8 },
    { id: "KNP-KR", label: "KNP", countryCode: "KR", lat: 37.57, lng: 126.98, avgRttMs: 2800, health: "unreachable", activeTunnel: false, lastPing: Date.now() - 60000, packetLoss: 85 },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function project(lat: number, lng: number): [number, number] {
    const x = (lng + 180) * (800 / 360);
    const latRad = (lat * Math.PI) / 180;
    const mercN = Math.log(Math.tan(Math.PI / 4 + latRad / 2));
    const y = 250 - (mercN * 800) / (2 * Math.PI);
    return [x, Math.max(20, Math.min(480, y))];
}

function healthColor(h: string): string {
    switch (h) {
        case "healthy": return "#22c55e";
        case "degraded": return "#f59e0b";
        case "bottleneck": return "#f97316";
        case "unreachable": return "#ef4444";
        default: return "#6b7280";
    }
}

function healthBg(h: string): string {
    switch (h) {
        case "healthy": return "bg-emerald-500/10 border-emerald-500/20";
        case "degraded": return "bg-amber-500/10 border-amber-500/20";
        case "bottleneck": return "bg-orange-500/10 border-orange-500/20";
        case "unreachable": return "bg-red-500/10 border-red-500/20";
        default: return "bg-zinc-500/10 border-zinc-500/20";
    }
}

// ─── SVG Map with Pulse Visualization ────────────────────────────────────────

function PulseMap({
    nodes,
    waves,
}: {
    nodes: NodePulse[];
    waves: BroadcastWave[];
}) {
    return (
        <svg viewBox="0 0 800 500" className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
            <defs>
                {/* Pulse glow filter */}
                <filter id="pulseGlow" x="-50%" y="-50%" width="200%" height="200%">
                    <feGaussianBlur stdDeviation="3" result="blur" />
                    <feMerge>
                        <feMergeNode in="blur" />
                        <feMergeNode in="SourceGraphic" />
                    </feMerge>
                </filter>
                {/* Wave gradient */}
                <radialGradient id="waveGrad">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity="0.6" />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity="0" />
                </radialGradient>
                {/* Tunnel line gradient */}
                <linearGradient id="tunnelGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#22c55e" stopOpacity="0.15" />
                    <stop offset="50%" stopColor="#22c55e" stopOpacity="0.4" />
                    <stop offset="100%" stopColor="#22c55e" stopOpacity="0.15" />
                </linearGradient>
            </defs>

            {/* Grid */}
            <rect width="800" height="500" fill="#0a0a0a" />
            {Array.from({ length: 20 }, (_, i) => (
                <line key={`vg-${i}`} x1={i * 40} y1="0" x2={i * 40} y2="500" stroke="#1a1a2e" strokeWidth="0.5" />
            ))}
            {Array.from({ length: 13 }, (_, i) => (
                <line key={`hg-${i}`} x1="0" y1={i * 40} x2="800" y2={i * 40} stroke="#1a1a2e" strokeWidth="0.5" />
            ))}

            {/* Encrypted tunnel connections between active nodes */}
            {nodes
                .filter((n) => n.activeTunnel)
                .map((a, i, arr) => {
                    const [ax, ay] = project(a.lat, a.lng);
                    return arr.slice(i + 1).map((b) => {
                        if (!b.activeTunnel) return null;
                        const [bx, by] = project(b.lat, b.lng);
                        return (
                            <line
                                key={`tunnel-${a.id}-${b.id}`}
                                x1={ax} y1={ay} x2={bx} y2={by}
                                stroke="url(#tunnelGrad)"
                                strokeWidth="0.4"
                                opacity="0.3"
                            />
                        );
                    });
                })}

            {/* Broadcast waves */}
            {waves.map((wave) => {
                const source = nodes.find((n) => n.id === wave.sourceNodeId);
                if (!source) return null;
                const [sx, sy] = project(source.lat, source.lng);

                return wave.targetNodeIds.map((tid) => {
                    const target = nodes.find((n) => n.id === tid);
                    if (!target) return null;
                    const [tx, ty] = project(target.lat, target.lng);

                    // Animated position along the line
                    const cx = sx + (tx - sx) * wave.progress;
                    const cy = sy + (ty - sy) * wave.progress;
                    const opacity = 1 - wave.progress * 0.7;

                    return (
                        <g key={`wave-${wave.id}-${tid}`}>
                            <line
                                x1={sx} y1={sy} x2={cx} y2={cy}
                                stroke="#22c55e"
                                strokeWidth="1.5"
                                opacity={opacity}
                                strokeDasharray="4 2"
                            />
                            <circle cx={cx} cy={cy} r={3} fill="#22c55e" opacity={opacity} filter="url(#pulseGlow)">
                                <animate attributeName="r" values="2;5;2" dur="0.6s" repeatCount="indefinite" />
                            </circle>
                        </g>
                    );
                });
            })}

            {/* Node markers */}
            {nodes.map((node) => {
                const [x, y] = project(node.lat, node.lng);
                const color = healthColor(node.health);
                const r = node.activeTunnel ? 5 : 3;

                return (
                    <g key={node.id}>
                        {/* Pulse ring for active tunnels */}
                        {node.activeTunnel && (
                            <circle cx={x} cy={y} r={r + 2} fill="none" stroke={color} strokeWidth="1" opacity="0.4">
                                <animate attributeName="r" values={`${r + 2};${r + 10};${r + 2}`} dur="2.5s" repeatCount="indefinite" />
                                <animate attributeName="opacity" values="0.4;0;0.4" dur="2.5s" repeatCount="indefinite" />
                            </circle>
                        )}
                        {/* Node dot */}
                        <circle cx={x} cy={y} r={r} fill={color} opacity={node.activeTunnel ? 1 : 0.4} filter={node.activeTunnel ? "url(#pulseGlow)" : ""} />
                        {/* Label */}
                        <text x={x} y={y - 10} textAnchor="middle" fill={color} fontSize="7" fontFamily="monospace" opacity="0.8">
                            {node.label}
                        </text>
                        {/* RTT badge */}
                        <text x={x + 10} y={y + 3} textAnchor="start" fill="#888" fontSize="6" fontFamily="monospace">
                            {node.activeTunnel ? `${node.avgRttMs}ms` : "—"}
                        </text>
                    </g>
                );
            })}

            {/* Legend */}
            <g transform="translate(15, 460)">
                {[
                    { label: "Healthy", color: "#22c55e" },
                    { label: "Degraded", color: "#f59e0b" },
                    { label: "Bottleneck", color: "#f97316" },
                    { label: "Offline", color: "#ef4444" },
                ].map((item, i) => (
                    <g key={item.label} transform={`translate(${i * 100}, 0)`}>
                        <circle cx="0" cy="0" r="3" fill={item.color} />
                        <text x="8" y="3" fill="#888" fontSize="7" fontFamily="monospace">{item.label}</text>
                    </g>
                ))}
            </g>
        </svg>
    );
}

// ─── RTT Sparkline ───────────────────────────────────────────────────────────

function RTTSparkline({ values, color }: { values: number[]; color: string }) {
    if (values.length < 2) return null;
    const max = Math.max(...values, 1);
    const w = 80;
    const h = 20;
    const points = values
        .map((v, i) => `${(i / (values.length - 1)) * w},${h - (v / max) * h}`)
        .join(" ");

    return (
        <svg width={w} height={h} className="inline-block">
            <polyline
                points={points}
                fill="none"
                stroke={color}
                strokeWidth="1"
                strokeLinecap="round"
                strokeLinejoin="round"
                opacity="0.7"
            />
        </svg>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function GlobalPulse() {
    const [nodes, setNodes] = useState<NodePulse[]>(NODES);
    const [waves, setWaves] = useState<BroadcastWave[]>([]);
    const [stats, setStats] = useState<PulseStats>({
        avgGlobalRtt: 0,
        activeTunnels: 0,
        totalPings: 0,
        totalFailures: 0,
        bottleneckCount: 0,
        healthiestNode: "",
        slowestNode: "",
    });
    const [rttHistory, setRttHistory] = useState<Record<string, number[]>>({});
    const waveCounter = useRef(0);

    // Compute stats
    useEffect(() => {
        const active = nodes.filter((n) => n.activeTunnel);
        const avg = active.length > 0
            ? active.reduce((s, n) => s + n.avgRttMs, 0) / active.length
            : 0;
        const bottlenecks = nodes.filter((n) => n.health === "bottleneck" || n.health === "unreachable");
        const healthiest = active.length > 0
            ? active.reduce((a, b) => a.avgRttMs < b.avgRttMs ? a : b)
            : null;
        const slowest = active.length > 0
            ? active.reduce((a, b) => a.avgRttMs > b.avgRttMs ? a : b)
            : null;

        setStats({
            avgGlobalRtt: Math.round(avg),
            activeTunnels: active.length,
            totalPings: nodes.length * 42, // Simulated count
            totalFailures: 3,
            bottleneckCount: bottlenecks.length,
            healthiestNode: healthiest?.label ?? "",
            slowestNode: slowest?.label ?? "",
        });
    }, [nodes]);

    // Simulate live RTT updates
    useEffect(() => {
        const interval = setInterval(() => {
            setNodes((prev) =>
                prev.map((n) => {
                    if (!n.activeTunnel) return n;
                    const jitter = (Math.random() - 0.5) * n.avgRttMs * 0.3;
                    const newRtt = Math.max(10, Math.round(n.avgRttMs + jitter));
                    return { ...n, avgRttMs: newRtt, lastPing: Date.now() };
                })
            );

            // Record RTT history
            setRttHistory((prev) => {
                const next = { ...prev };
                NODES.forEach((n) => {
                    const history = next[n.id] ?? [];
                    const jitter = (Math.random() - 0.5) * n.avgRttMs * 0.3;
                    history.push(Math.max(10, Math.round(n.avgRttMs + jitter)));
                    if (history.length > 20) history.shift();
                    next[n.id] = history;
                });
                return next;
            });
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    // Simulate broadcast waves periodically
    useEffect(() => {
        const interval = setInterval(() => {
            const availableNodes = nodes.filter((n) => n.activeTunnel);
            if (availableNodes.length < 2) return;

            const source = availableNodes[Math.floor(Math.random() * availableNodes.length)];
            const targets = availableNodes.filter((n) => n.id !== source.id).map((n) => n.id);

            waveCounter.current += 1;
            const wave: BroadcastWave = {
                id: `wave-${waveCounter.current}`,
                sourceNodeId: source.id,
                targetNodeIds: targets,
                startedAt: Date.now(),
                progress: 0,
            };

            setWaves((prev) => [...prev, wave]);

            // Animate wave progress
            let progress = 0;
            const animInterval = setInterval(() => {
                progress += 0.05;
                if (progress >= 1) {
                    clearInterval(animInterval);
                    setWaves((prev) => prev.filter((w) => w.id !== wave.id));
                    return;
                }
                setWaves((prev) =>
                    prev.map((w) => w.id === wave.id ? { ...w, progress } : w)
                );
            }, 60);
        }, 8000);
        return () => clearInterval(interval);
    }, [nodes]);

    return (
        <div className="flex flex-col gap-4">
            {/* ── Telemetry Bar ── */}
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
                {[
                    { label: "Avg Response", value: `${stats.avgGlobalRtt}ms`, icon: Clock, accent: "text-tactical-primary" },
                    { label: "Active Tunnels", value: stats.activeTunnels.toString(), icon: Lock, accent: "text-emerald-400" },
                    { label: "Total Pings", value: stats.totalPings.toLocaleString(), icon: Radio, accent: "text-cyan-400" },
                    { label: "Failures", value: stats.totalFailures.toString(), icon: AlertTriangle, accent: "text-red-400" },
                    { label: "Bottlenecks", value: stats.bottleneckCount.toString(), icon: TrendingUp, accent: "text-orange-400" },
                    { label: "Fastest Node", value: stats.healthiestNode, icon: Zap, accent: "text-emerald-400" },
                    { label: "Slowest Node", value: stats.slowestNode, icon: WifiOff, accent: "text-amber-400" },
                ].map((s) => (
                    <div key={s.label} className="flex flex-col gap-0.5 bg-tactical-surface border border-tactical-border rounded px-2.5 py-2">
                        <div className="flex items-center gap-1">
                            <s.icon className={`w-3 h-3 ${s.accent} flex-shrink-0`} />
                            <span className="font-data text-[7px] uppercase tracking-wider text-tactical-text-dim truncate">{s.label}</span>
                        </div>
                        <p className={`font-data text-sm font-bold tabular-nums ${s.accent} truncate`}>{s.value}</p>
                    </div>
                ))}
            </div>

            {/* ── Map + Node Table ── */}
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_360px] gap-0">
                {/* Map */}
                <div className="relative border border-tactical-border rounded-l overflow-hidden bg-[#0a0a0a] min-h-[400px]">
                    <PulseMap nodes={nodes} waves={waves} />

                    {/* Broadcast indicator */}
                    {waves.length > 0 && (
                        <div className="absolute top-3 left-3 flex items-center gap-2 bg-tactical-primary/10 border border-tactical-primary/30 rounded px-2.5 py-1">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-tactical-primary opacity-75" />
                                <span className="relative inline-flex h-2 w-2 rounded-full bg-tactical-primary" />
                            </span>
                            <span className="font-data text-[9px] font-bold uppercase tracking-wider text-tactical-primary">
                                Broadcasting Query
                            </span>
                        </div>
                    )}

                    {/* mTLS badge */}
                    <div className="absolute bottom-3 right-3 flex items-center gap-1.5 bg-zinc-900/80 border border-zinc-800 rounded px-2 py-1">
                        <Shield className="w-3 h-3 text-emerald-400" />
                        <span className="font-data text-[8px] text-zinc-500">mTLS 1.3 • AES-256-GCM</span>
                    </div>
                </div>

                {/* Node RTT Table */}
                <div className="bg-tactical-surface border border-tactical-border border-l-0 rounded-r overflow-hidden flex flex-col">
                    <div className="flex items-center justify-between px-3 py-2 border-b border-tactical-border">
                        <div className="flex items-center gap-2">
                            <Wifi className="w-3.5 h-3.5 text-tactical-primary" />
                            <span className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text-muted">
                                Node Latency
                            </span>
                        </div>
                        <span className="font-data text-[8px] text-tactical-text-dim">
                            RTT • ms
                        </span>
                    </div>

                    <div className="flex-1 overflow-y-auto divide-y divide-tactical-border/30">
                        {nodes
                            .slice()
                            .sort((a, b) => {
                                if (!a.activeTunnel && b.activeTunnel) return 1;
                                if (a.activeTunnel && !b.activeTunnel) return -1;
                                return a.avgRttMs - b.avgRttMs;
                            })
                            .map((node) => (
                                <div
                                    key={node.id}
                                    className="flex items-center justify-between px-3 py-2 hover:bg-zinc-800/30 transition-colors"
                                >
                                    <div className="flex items-center gap-2 min-w-0">
                                        {node.activeTunnel ? (
                                            <CheckCircle className="w-3 h-3 flex-shrink-0" style={{ color: healthColor(node.health) }} />
                                        ) : (
                                            <XCircle className="w-3 h-3 text-red-400 flex-shrink-0" />
                                        )}
                                        <div className="min-w-0">
                                            <p className="font-data text-[10px] font-medium text-tactical-text truncate">
                                                {node.label}
                                                <span className="text-zinc-600 ml-1">{node.countryCode}</span>
                                            </p>
                                            <p className="font-data text-[7px] text-zinc-600">
                                                loss: {node.packetLoss}%
                                            </p>
                                        </div>
                                    </div>

                                    <div className="flex items-center gap-2">
                                        <RTTSparkline
                                            values={rttHistory[node.id] ?? []}
                                            color={healthColor(node.health)}
                                        />
                                        <div className="text-right w-14">
                                            <p
                                                className="font-data text-xs font-bold tabular-nums"
                                                style={{ color: healthColor(node.health) }}
                                            >
                                                {node.activeTunnel ? `${node.avgRttMs}` : "—"}
                                            </p>
                                            <p className="font-data text-[7px] text-zinc-600">
                                                {node.health}
                                            </p>
                                        </div>
                                    </div>
                                </div>
                            ))}
                    </div>

                    {/* Summary */}
                    <div className="flex items-center justify-between px-3 py-2 border-t border-tactical-border bg-zinc-900/50">
                        <div className="flex items-center gap-1.5">
                            <Globe className="w-3 h-3 text-tactical-primary" />
                            <span className="font-data text-[8px] text-zinc-500">
                                {stats.activeTunnels} encrypted tunnels active
                            </span>
                        </div>
                        <span className="font-data text-[8px] text-zinc-500">
                            avg {stats.avgGlobalRtt}ms
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
