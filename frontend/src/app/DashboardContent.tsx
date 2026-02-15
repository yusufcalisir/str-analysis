"use client";

import { useEffect, useState, useRef, useMemo } from "react";
import {
    Terminal,
    Activity,
    Network,
    Database,
    Zap,
    ShieldAlert,
    CheckCircle,
    Globe
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import DNAIngestForm from "@/components/forms/DNAIngestForm";

// ─── Types ───────────────────────────────────────────────────────────────────

interface LogEntry {
    id: string;
    timestamp: string;
    level: "SUCCESS" | "INFO" | "WARNING" | "ERROR";
    message: string;
}

interface ActivityItem {
    id: string;
    type: "ACC" | "RTN" | "QRY";
    hash: string;
    origin: string;
    rtt: number;
    timestamp: number;
}

interface NodeStatus {
    id: string;
    label: string;
    region: string;
    latency: number;
    status: "online" | "degraded" | "offline";
}

interface GlobalNodesProps {
    onSelectNode: (nodeId: string) => void;
}

// ─── Generators ──────────────────────────────────────────────────────────────

const LOG_TEMPLATES = [
    { level: "INFO", msg: "SYNC_PEER: Handshake accepted [192.168.x.x]" },
    { level: "SUCCESS", msg: "BLOCK_VERIFIED: Hash match confirmed" },
    { level: "INFO", msg: "INDEX_UPDATE: Vector entries re-balanced" },
    { level: "WARNING", msg: "LATENCY_SPIKE: Route optimized (US-EAST)" },
    { level: "INFO", msg: "CONSENSUS: 16/16 validators signed" },
    { level: "SUCCESS", msg: "ZKP_PROOF: Generating SNARK (Groth16)... done" },
    { level: "INFO", msg: "MEMPOOL: 14 pending transactions" },
    { level: "INFO", msg: "STORAGE: Dedup check passed" },
];

const ACTIVITY_TEMPLATES = [
    { type: "ACC", origin: "PFA-AR" },
    { type: "ACC", origin: "NPA-JP" },
    { type: "RTN", origin: "CBI-IN" },
    { type: "RTN", origin: "SAPS-ZA" },
    { type: "QRY", origin: "FBI-US" },
    { type: "QRY", origin: "AFP-AU" },
    { type: "ACC", origin: "BKA-DE" },
];

const generateHexId = (len: number) => Math.floor(Math.random() * 16 ** len).toString(16).toUpperCase().padStart(len, "0");

// ─── Component: SystemLogs ───────────────────────────────────────────────────

function SystemLogs() {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Initial logs + Interval
    useEffect(() => {
        // Set initial logs on client-side only
        setLogs([
            { id: "init-1", timestamp: new Date().toLocaleTimeString(), level: "SUCCESS", message: "NODE_GEN_VERIFIED" },
            { id: "init-2", timestamp: new Date().toLocaleTimeString(), level: "INFO", message: "MILVUS_CONN: ESTABLISHED" },
            { id: "init-3", timestamp: new Date().toLocaleTimeString(), level: "INFO", message: "AGENT_BOOT: OSPy_PIPELINE_ACTIVE" },
            { id: "init-4", timestamp: new Date().toLocaleTimeString(), level: "INFO", message: "ZKP_MODULE: STANDBY" },
            { id: "init-5", timestamp: new Date().toLocaleTimeString(), level: "SUCCESS", message: "VECTOR_INDEX: 12,847 PROFILES LOADED" },
        ]);

        const interval = setInterval(() => {
            const template = LOG_TEMPLATES[Math.floor(Math.random() * LOG_TEMPLATES.length)];
            const newLog: LogEntry = {
                id: generateHexId(8),
                timestamp: new Date().toLocaleTimeString(),
                level: template.level as any,
                message: template.msg
            };

            setLogs(prev => {
                const next = [...prev, newLog];
                if (next.length > 50) next.shift(); // Keep buffer size manageable
                return next;
            });
        }, 2000 + Math.random() * 3000); // Random interval 2-5s

        return () => clearInterval(interval);
    }, []);

    // Auto-scroll
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="flex h-full flex-col border border-tactical-border/40 bg-tactical-surface/30 rounded overflow-hidden">
            <div className="flex items-center gap-2 border-b border-tactical-border/40 px-3 py-1.5 bg-tactical-surface/50">
                <Terminal className="h-3 w-3 text-tactical-text-dim" />
                <span className="font-data text-[10px] font-bold tracking-[0.15em] text-tactical-text-dim uppercase">System_Logs</span>
            </div>
            <div ref={scrollRef} className="flex-1 overflow-y-auto p-3 font-data text-[10px] space-y-1 scroll-smooth">
                {logs.map((log) => (
                    <div key={log.id} className="flex gap-2">
                        <span className="text-zinc-700 whitespace-nowrap">[{log.timestamp}]</span>
                        <span className={
                            log.level === "SUCCESS" ? "text-emerald-500" :
                                log.level === "WARNING" ? "text-amber-500" :
                                    log.level === "ERROR" ? "text-red-500" :
                                        "text-tactical-primary"
                        }>
                            {log.message}
                        </span>
                    </div>
                ))}
            </div>
        </div>
    );
}

// ─── Component: GlobalNodes ──────────────────────────────────────────────────

function GlobalNodes({ onSelectNode }: GlobalNodesProps) {
    const [nodes, setNodes] = useState<NodeStatus[]>([
        { id: "INTERPOL-EU-DE", label: "INTERPOL-EU-DE", region: "DE", latency: 8, status: "online" },
        { id: "FBI-US-TX", label: "FBI-US-TX", region: "TX", latency: 23, status: "online" },
        { id: "NPA-JP-TK", label: "NPA-JP-TK", region: "TK", latency: 45, status: "online" },
        { id: "RCMP-CA-ON", label: "RCMP-CA-ON", region: "ON", latency: 31, status: "degraded" },
        { id: "AFP-AU-SY", label: "AFP-AU-SY", region: "SY", latency: 67, status: "online" },
    ]);

    useEffect(() => {
        const interval = setInterval(() => {
            setNodes(prev => prev.map(node => {
                // Randomize latency slightly
                const variance = Math.floor(Math.random() * 10) - 5; // -5 to +5
                let newLatency = Math.max(1, node.latency + variance);

                // Random status flip (rare)
                let newStatus = node.status;
                if (Math.random() > 0.98) {
                    newStatus = node.status === "online" ? "degraded" : "online";
                }

                return { ...node, latency: newLatency, status: newStatus };
            }));
        }, 1500);

        return () => clearInterval(interval);
    }, []);

    // Fetch Real-time System Stats
    const [stats, setStats] = useState({ profiles: 0, uptime: 0, loaded: false });

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const res = await fetch("http://localhost:8000/system/stats");
                if (res.ok) {
                    const data = await res.json();
                    setStats({
                        profiles: data.total_profiles,
                        uptime: data.uptime_seconds,
                        loaded: true
                    });
                }
            } catch (e) {
                console.error("Stats fetch failed", e);
            }
        };

        fetchStats(); // Initial
        const interval = setInterval(fetchStats, 5000); // Poll every 5s
        return () => clearInterval(interval);
    }, []);

    // Format uptime helper
    const formatUptime = (seconds: number) => {
        if (seconds < 60) return `${Math.floor(seconds)}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        const hours = Math.floor(seconds / 3600);
        return `${hours}h ${(Math.floor(seconds % 3600 / 60))}m`;
    };

    return (
        <div className="flex h-full flex-col border border-tactical-border/40 bg-tactical-surface/30 rounded overflow-hidden">
            <div className="flex items-center gap-2 border-b border-tactical-border/40 px-3 py-1.5 bg-tactical-surface/50">
                <Globe className="h-3 w-3 text-tactical-text-dim" />
                <span className="font-data text-[10px] font-bold tracking-[0.15em] text-tactical-text-dim uppercase">Global_Nodes</span>
            </div>
            <div className="flex-1 overflow-y-auto p-3 space-y-2">
                {nodes.map((node) => (
                    <div
                        key={node.id}
                        onClick={() => onSelectNode(node.id)}
                        className="flex items-center justify-between group cursor-pointer hover:bg-tactical-surface-elevated/50 p-1 rounded transition-all"
                    >
                        <div className="flex items-center gap-2">
                            <div className={`h-1.5 w-1.5 rounded-full transition-colors duration-500 ${node.status === "online" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" :
                                node.status === "degraded" ? "bg-amber-500" : "bg-red-500"
                                }`} />
                            <span className="font-data text-[9px] font-bold text-tactical-text-muted group-hover:text-tactical-text transition-colors">
                                {node.label}
                            </span>
                        </div>
                        <span className="font-data text-[8px] text-zinc-600 group-hover:text-zinc-500 transition-colors tabular-nums">
                            {node.latency}ms
                        </span>
                    </div>
                ))}
            </div>
            <div className="p-3 border-t border-tactical-border/20 mt-auto bg-black/10">
                <div className="flex justify-between items-end">
                    <div className="flex flex-col">
                        <span className="font-data text-[14px] lg:text-[16px] font-bold text-emerald-500 tabular-nums">
                            {stats.loaded ? stats.profiles.toLocaleString() : "..."}
                        </span>
                        <span className="font-data text-[7px] text-zinc-600 uppercase tracking-tighter">Profiles</span>
                    </div>
                    <div className="flex flex-col items-end">
                        <span className="font-data text-[14px] lg:text-[16px] font-bold text-tactical-text">
                            {stats.loaded ? formatUptime(stats.uptime) : "..."}
                        </span>
                        <span className="font-data text-[7px] text-zinc-600 uppercase tracking-tighter">Uptime</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ─── Component: StatsGrid ────────────────────────────────────────────────────

function StatsGrid() {
    const [stats, setStats] = useState({
        profiles: 0,
        latency: 0,
        qps: 0,
        nodes: 5,
        validated: 0,
        quarantined: 0,
        loaded: false
    });

    useEffect(() => {
        const fetchStats = async () => {
            try {
                // We reuse the same endpoint. In a real app, we might want a SWR hook or context.
                const res = await fetch("http://localhost:8000/system/stats");
                if (res.ok) {
                    const data = await res.json();
                    setStats(prev => ({
                        ...prev,
                        profiles: data.total_profiles,
                        // Simulate dynamic metrics for others since backend doesn't track QPS/Validated yet
                        latency: Math.max(5, Math.min(80, (prev.latency || 16) + (Math.floor(Math.random() * 10) - 5))),
                        qps: Math.max(10, Math.min(100, (prev.qps || 21) + (Math.floor(Math.random() * 6) - 3))),
                        validated: (prev.validated || 1356) + (Math.random() > 0.9 ? 1 : 0),
                        loaded: true
                    }));
                }
            } catch (e) {
                console.error("Stats fetch failed", e);
            }
        };

        fetchStats();
        const interval = setInterval(fetchStats, 5000);
        return () => clearInterval(interval);
    }, []);

    const statItems = [
        { label: "Total Profiles", value: stats.loaded ? stats.profiles.toLocaleString() : "...", color: "text-tactical-text", icon: Database },
        { label: "Latency", value: `${stats.latency} ms`, color: stats.latency > 50 ? "text-amber-500" : "text-emerald-500", icon: Activity },
        { label: "Queries/Sec", value: stats.qps.toString(), color: "text-blue-400", icon: Zap },
        { label: "Nodes Online", value: stats.nodes.toString(), color: "text-emerald-500", icon: Network },
        { label: "Validated Today", value: stats.validated.toLocaleString(), color: "text-emerald-500", icon: CheckCircle },
        { label: "Quarantined", value: stats.quarantined.toString(), color: "text-amber-500", icon: ShieldAlert },
    ];

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 lg:gap-3">
            {statItems.map((s, i) => (
                <div key={i} className="bg-tactical-surface/50 border border-tactical-border/40 rounded p-2 lg:p-3">
                    <div className="flex items-center gap-1.5 mb-0.5 lg:mb-1 opacity-60">
                        <s.icon className="h-2.5 w-2.5 lg:h-3 lg:w-3" />
                        <span className="font-data text-[7px] lg:text-[8px] uppercase tracking-wider text-tactical-text-dim">{s.label}</span>
                    </div>
                    <p className={`font-data text-base lg:text-xl font-bold ${s.color} tabular-nums transition-colors duration-500`}>{s.value}</p>
                </div>
            ))}
        </div>
    );
}

// ─── Component: ActivityFeed ─────────────────────────────────────────────────

function ActivityFeed() {
    const [activities, setActivities] = useState<ActivityItem[]>([]);

    useEffect(() => {
        // Initial Data
        setActivities([
            { id: "init-1", type: "ACC", hash: "2266252F", origin: "PFA-AR", rtt: 35, timestamp: Date.now() },
            { id: "init-2", type: "ACC", hash: "F6EBECF2", origin: "NPA-JP", rtt: 25, timestamp: Date.now() - 5000 },
            { id: "init-3", type: "RTN", hash: "9516785E", origin: "CBI-IN", rtt: 54, timestamp: Date.now() - 12000 },
            { id: "init-4", type: "RTN", hash: "4E5F09F2", origin: "SAPS-ZA", rtt: 42, timestamp: Date.now() - 25000 },
        ]);

        const interval = setInterval(() => {
            const template = ACTIVITY_TEMPLATES[Math.floor(Math.random() * ACTIVITY_TEMPLATES.length)];
            const newItem: ActivityItem = {
                id: generateHexId(12),
                type: template.type as "ACC" | "RTN" | "QRY",
                hash: generateHexId(8),
                origin: template.origin,
                rtt: Math.floor(Math.random() * 80) + 15,
                timestamp: Date.now()
            };

            setActivities(prev => {
                const next = [newItem, ...prev];
                if (next.length > 30) next.pop();
                return next;
            });
        }, 1500 + Math.random() * 2000);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="border border-tactical-border/40 bg-tactical-surface/30 rounded overflow-hidden">
            <div className="flex items-center justify-between border-b border-tactical-border/40 px-3 py-1.5 bg-tactical-surface/50">
                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="font-data text-[10px] font-bold tracking-[0.15em] text-tactical-text-dim uppercase">Activity_Feed</span>
                </div>
                <span className="font-data text-[8px] text-zinc-600 uppercase tracking-widest">Live</span>
            </div>
            <div className="divide-y divide-tactical-border/20 max-h-[300px] lg:max-h-none overflow-y-auto custom-scrollbar">
                <AnimatePresence initial={false}>
                    {activities.map((item) => (
                        <motion.div
                            key={item.id}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0 }}
                            className="flex items-center justify-between px-3 py-2 hover:bg-tactical-surface-elevated/50 transition-colors"
                        >
                            <div className="flex items-center gap-3 lg:gap-4">
                                <span className={`font-data text-[8px] lg:text-[9px] font-bold ${item.type === "ACC" ? "text-emerald-500" :
                                    item.type === "RTN" ? "text-amber-500" : "text-blue-400"
                                    }`}>
                                    {item.type}
                                </span>
                                <span className="font-data text-[9px] lg:text-[10px] text-zinc-400 uppercase">{item.hash}</span>
                            </div>
                            <div className="flex items-center gap-4 lg:gap-6">
                                <span className="font-data text-[8px] lg:text-[9px] text-zinc-600">{item.origin}</span>
                                <span className="font-data text-[8px] lg:text-[9px] text-zinc-500 w-10 lg:w-12 text-right tabular-nums">{item.rtt} ms</span>
                            </div>
                        </motion.div>
                    ))}
                </AnimatePresence>
            </div>
        </div>
    );
}

// ─── Main Content ────────────────────────────────────────────────────────────

export default function DashboardContent() {
    const [selectedNode, setSelectedNode] = useState<string>("");

    return (
        <div className="flex flex-col min-h-full lg:h-[calc(100vh-140px)] gap-4 p-1 overflow-x-hidden overflow-y-auto lg:overflow-hidden">
            {/* Main 3-Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-[260px_1fr_280px] gap-4 h-full">

                {/* Left Column: Logs - Lower priority on mobile */}
                <div className="order-3 lg:order-1 h-[300px] lg:h-full overflow-hidden">
                    <SystemLogs />
                </div>

                {/* Center Column: Ingest + Stats - Priority 1 on mobile */}
                <div className="order-1 lg:order-2 flex flex-col gap-4 min-h-0">
                    <div className="flex-1 border border-tactical-border/40 bg-tactical-surface/30 rounded p-4 overflow-y-auto custom-scrollbar shadow-inner">
                        <DNAIngestForm selectedNodeId={selectedNode} onNodeChange={setSelectedNode} />
                    </div>
                    <div className="h-fit">
                        <StatsGrid />
                    </div>
                </div>

                {/* Right Column: Nodes + Feed - Priority 2 on mobile */}
                <div className="order-2 lg:order-3 flex flex-col gap-4 min-h-0 lg:h-full">
                    <div className="h-fit min-h-[160px] lg:h-[45%]">
                        <GlobalNodes onSelectNode={setSelectedNode} />
                    </div>
                    <div className="flex-1 overflow-hidden min-h-[250px] lg:min-h-0">
                        <ActivityFeed />
                    </div>
                </div>
            </div>
        </div>
    );
}
