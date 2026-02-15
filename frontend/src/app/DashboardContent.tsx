"use client";

import { useEffect, useState, useMemo } from "react";
import {
    Terminal,
    Activity,
    Network,
    Database,
    Zap,
    ShieldAlert,
    CheckCircle,
    Radio,
    Globe
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import DNAIngestForm from "@/components/forms/DNAIngestForm";

// ─── Types ───────────────────────────────────────────────────────────────────

interface LogEntry {
    timestamp: string;
    level: "SUCCESS" | "INFO" | "WARNING" | "ERROR";
    message: string;
}

interface ActivityItem {
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

// ─── Component: ConsoleLog ───────────────────────────────────────────────────

function SystemLogs() {
    const [logs, setLogs] = useState<LogEntry[]>([
        { timestamp: "21:04:15", level: "SUCCESS", message: "NODE_GEN_VERIFIED" },
        { timestamp: "21:04:16", level: "INFO", message: "MILVUS_CONN: ESTABLISHED" },
        { timestamp: "21:04:17", level: "INFO", message: "AGENT_BOOT: OSPy_PIPELINE_ACTIVE" },
        { timestamp: "21:04:18", level: "INFO", message: "ZKP_MODULE: STANDBY" },
        { timestamp: "21:04:19", level: "SUCCESS", message: "VECTOR_INDEX: 12,847 PROFILES LOADED" },
    ]);

    return (
        <div className="flex h-full flex-col border border-tactical-border/40 bg-tactical-surface/30 rounded overflow-hidden">
            <div className="flex items-center gap-2 border-b border-tactical-border/40 px-3 py-1.5 bg-tactical-surface/50">
                <Terminal className="h-3 w-3 text-tactical-text-dim" />
                <span className="font-data text-[10px] font-bold tracking-[0.15em] text-tactical-text-dim uppercase">System_Logs</span>
            </div>
            <div className="flex-1 overflow-y-auto p-3 font-data text-[10px] space-y-1">
                {logs.map((log, i) => (
                    <div key={i} className="flex gap-2">
                        <span className="text-zinc-700">[{log.timestamp}]</span>
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

// ─── Component: GlobalNodes ──────────────────────────────────────────────────

function GlobalNodes({ onSelectNode }: GlobalNodesProps) {
    const nodes: NodeStatus[] = [
        { id: "INTERPOL-EU-DE", label: "INTERPOL-EU-DE", region: "DE", latency: 8, status: "online" },
        { id: "FBI-US-TX", label: "FBI-US-TX", region: "TX", latency: 23, status: "online" },
        { id: "NPA-JP-TK", label: "NPA-JP-TK", region: "TK", latency: 45, status: "online" },
        { id: "RCMP-CA-ON", label: "RCMP-CA-ON", region: "ON", latency: 31, status: "degraded" },
        { id: "AFP-AU-SY", label: "AFP-AU-SY", region: "SY", latency: 67, status: "online" },
    ];

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
                            <div className={`h-1.5 w-1.5 rounded-full ${node.status === "online" ? "bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.5)]" :
                                node.status === "degraded" ? "bg-amber-500" : "bg-red-500"
                                }`} />
                            <span className="font-data text-[9px] font-bold text-tactical-text-muted group-hover:text-tactical-text transition-colors">
                                {node.label}
                            </span>
                        </div>
                        <span className="font-data text-[8px] text-zinc-600 group-hover:text-zinc-500 transition-colors">
                            {node.latency}ms
                        </span>
                    </div>
                ))}
            </div>
            <div className="p-3 border-t border-tactical-border/20 mt-auto bg-black/10">
                <div className="flex justify-between items-end">
                    <div className="flex flex-col">
                        <span className="font-data text-[14px] lg:text-[16px] font-bold text-emerald-500">12.8K</span>
                        <span className="font-data text-[7px] text-zinc-600 uppercase tracking-tighter">Profiles</span>
                    </div>
                    <div className="flex flex-col items-end">
                        <span className="font-data text-[14px] lg:text-[16px] font-bold text-tactical-text">99.7%</span>
                        <span className="font-data text-[7px] text-zinc-600 uppercase tracking-tighter">Uptime</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ─── Component: StatsGrid ────────────────────────────────────────────────────

function StatsGrid() {
    const stats = [
        { label: "Total Profiles", value: "24,956", color: "text-tactical-text", icon: Database },
        { label: "Latency", value: "16 ms", color: "text-emerald-500", icon: Activity },
        { label: "Queries/Sec", value: "21", color: "text-blue-400", icon: Zap },
        { label: "Nodes Online", value: "1", color: "text-emerald-500", icon: Network },
        { label: "Validated Today", value: "1,356", color: "text-emerald-500", icon: CheckCircle },
        { label: "Quarantined", value: "53", color: "text-amber-500", icon: ShieldAlert },
    ];

    return (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 lg:gap-3">
            {stats.map((s, i) => (
                <div key={i} className="bg-tactical-surface/50 border border-tactical-border/40 rounded p-2 lg:p-3">
                    <div className="flex items-center gap-1.5 mb-0.5 lg:mb-1 opacity-60">
                        <s.icon className="h-2.5 w-2.5 lg:h-3 lg:w-3" />
                        <span className="font-data text-[7px] lg:text-[8px] uppercase tracking-wider text-tactical-text-dim">{s.label}</span>
                    </div>
                    <p className={`font-data text-base lg:text-xl font-bold ${s.color}`}>{s.value}</p>
                </div>
            ))}
        </div>
    );
}

// ─── Component: ActivityFeed ─────────────────────────────────────────────────

function ActivityFeed() {
    const activities: ActivityItem[] = [
        { type: "ACC", hash: "2266252F", origin: "PFA-AR", rtt: 35, timestamp: Date.now() },
        { type: "ACC", hash: "F6EBECF2", origin: "NPA-JP", rtt: 25, timestamp: Date.now() - 5000 },
        { type: "RTN", hash: "9516785E", origin: "CBI-IN", rtt: 54, timestamp: Date.now() - 12000 },
        { type: "RTN", hash: "4E5F09F2", origin: "SAPS-ZA", rtt: 42, timestamp: Date.now() - 25000 },
        { type: "ACC", hash: "778AF6CC", origin: "KNPA-KR", rtt: 21, timestamp: Date.now() - 35000 },
        { type: "ACC", hash: "9E33D200", origin: "SAPS-ZA", rtt: 62, timestamp: Date.now() - 48000 },
        { type: "ACC", hash: "64A02166", origin: "SAPS-ZA", rtt: 37, timestamp: Date.now() - 60000 },
        { type: "ACC", hash: "4AEEA43C", origin: "SAPS-ZA", rtt: 23, timestamp: Date.now() - 80000 },
    ];

    return (
        <div className="border border-tactical-border/40 bg-tactical-surface/30 rounded overflow-hidden">
            <div className="flex items-center justify-between border-b border-tactical-border/40 px-3 py-1.5 bg-tactical-surface/50">
                <div className="flex items-center gap-2">
                    <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                    <span className="font-data text-[10px] font-bold tracking-[0.15em] text-tactical-text-dim uppercase">Activity_Feed</span>
                </div>
                <span className="font-data text-[8px] text-zinc-600 uppercase tracking-widest">Live</span>
            </div>
            <div className="divide-y divide-tactical-border/20 max-h-[300px] lg:max-h-none overflow-y-auto">
                {activities.map((item, i) => (
                    <div key={i} className="flex items-center justify-between px-3 py-2 hover:bg-tactical-surface-elevated/50 transition-colors">
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
                            <span className="font-data text-[8px] lg:text-[9px] text-zinc-500 w-10 lg:w-12 text-right">{item.rtt} ms</span>
                        </div>
                    </div>
                ))}
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
