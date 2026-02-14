"use client";

import { useEffect, useState, useCallback } from "react";
import {
    Shield,
    Database,
    Search,
    Lock,
    Unlock,
    AlertTriangle,
    Activity,
    Globe,
    Clock,
    BarChart3,
    Power,
    Eye,
    EyeOff,
    CheckCircle,
    XCircle,
    Zap,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

type SecurityLevel = "normal" | "elevated" | "critical" | "lockdown";

interface QueryType {
    id: string;
    label: string;
    description: string;
    enabled: boolean;
}

interface RecentQuery {
    id: string;
    queryType: string;
    requestingNode: string;
    matches: number;
    timestamp: number;
    blocked: boolean;
}

interface NodeStats {
    localDatabaseSize: number;
    incomingQueriesLast24h: number;
    successfulMatches: number;
    blockedByLaw: number;
    tokensActive: number;
    avgSearchLatencyMs: number;
    securityLevel: SecurityLevel;
    flRoundsCompleted: number;
    modelVersion: number;
    uptime: string;
}

// ─── Mock Data ───────────────────────────────────────────────────────────────

const INITIAL_QUERY_TYPES: QueryType[] = [
    { id: "criminal_search", label: "Criminal Search", description: "Standard forensic criminal identification queries", enabled: true },
    { id: "missing_persons", label: "Missing Persons", description: "Humanitarian search for unidentified remains", enabled: true },
    { id: "disaster_identification", label: "Disaster ID", description: "Mass disaster victim identification", enabled: true },
    { id: "kinship_analysis", label: "Kinship Analysis", description: "Familial DNA searching across borders", enabled: false },
    { id: "interpol_red_notice", label: "INTERPOL Red Notice", description: "Priority queries from INTERPOL", enabled: true },
];

const MOCK_RECENT_QUERIES: RecentQuery[] = [
    { id: "Q-7281", queryType: "criminal_search", requestingNode: "EUROPOL-NL", matches: 2, timestamp: Date.now() - 180_000, blocked: false },
    { id: "Q-7280", queryType: "kinship_analysis", requestingNode: "FBI-US-DC", matches: 0, timestamp: Date.now() - 420_000, blocked: true },
    { id: "Q-7279", queryType: "missing_persons", requestingNode: "NCA-UK", matches: 1, timestamp: Date.now() - 900_000, blocked: false },
    { id: "Q-7278", queryType: "interpol_red_notice", requestingNode: "DGPN-FR", matches: 3, timestamp: Date.now() - 1_500_000, blocked: false },
    { id: "Q-7277", queryType: "criminal_search", requestingNode: "BKA-DE", matches: 0, timestamp: Date.now() - 2_100_000, blocked: false },
    { id: "Q-7276", queryType: "disaster_identification", requestingNode: "AFP-AU", matches: 1, timestamp: Date.now() - 3_000_000, blocked: false },
];

// ─── Helper ──────────────────────────────────────────────────────────────────

function timeAgo(ts: number): string {
    const s = Math.floor((Date.now() - ts) / 1000);
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
}

// ─── Component ───────────────────────────────────────────────────────────────

export default function LocalNodeStatus() {
    const [stats, setStats] = useState<NodeStats>({
        localDatabaseSize: 45_127,
        incomingQueriesLast24h: 342,
        successfulMatches: 28,
        blockedByLaw: 14,
        tokensActive: 56,
        avgSearchLatencyMs: 18.4,
        securityLevel: "normal",
        flRoundsCompleted: 7,
        modelVersion: 3,
        uptime: "14d 7h 23m",
    });

    const [queryTypes, setQueryTypes] = useState<QueryType[]>(INITIAL_QUERY_TYPES);
    const [recentQueries] = useState<RecentQuery[]>(MOCK_RECENT_QUERIES);
    const [killSwitchArmed, setKillSwitchArmed] = useState(false);

    // Simulate live stat updates
    useEffect(() => {
        const interval = setInterval(() => {
            setStats(prev => ({
                ...prev,
                incomingQueriesLast24h: prev.incomingQueriesLast24h + (Math.random() > 0.7 ? 1 : 0),
                tokensActive: Math.max(0, prev.tokensActive + Math.round((Math.random() - 0.4) * 3)),
                avgSearchLatencyMs: Math.max(5, prev.avgSearchLatencyMs + (Math.random() - 0.5) * 2),
            }));
        }, 4000);
        return () => clearInterval(interval);
    }, []);

    const toggleQueryType = useCallback((id: string) => {
        setQueryTypes(prev =>
            prev.map(qt => qt.id === id ? { ...qt, enabled: !qt.enabled } : qt)
        );
    }, []);

    const securityColor = (level: SecurityLevel) => {
        switch (level) {
            case "normal": return "text-tactical-primary";
            case "elevated": return "text-amber-400";
            case "critical": return "text-orange-500";
            case "lockdown": return "text-red-500";
        }
    };

    const securityBg = (level: SecurityLevel) => {
        switch (level) {
            case "normal": return "bg-tactical-primary/10 border-tactical-primary/20";
            case "elevated": return "bg-amber-500/10 border-amber-500/20";
            case "critical": return "bg-orange-500/10 border-orange-500/20";
            case "lockdown": return "bg-red-500/10 border-red-500/20";
        }
    };

    return (
        <div className="flex flex-col gap-4">
            {/* ── Security Banner ── */}
            <div className={`flex items-center justify-between px-4 py-2.5 rounded border ${securityBg(stats.securityLevel)}`}>
                <div className="flex items-center gap-2">
                    <Shield className={`w-4 h-4 ${securityColor(stats.securityLevel)}`} />
                    <span className={`font-data text-xs font-bold uppercase tracking-wider ${securityColor(stats.securityLevel)}`}>
                        Security: {stats.securityLevel}
                    </span>
                </div>
                <div className="flex items-center gap-3">
                    <span className="font-data text-[9px] text-tactical-text-dim">
                        Uptime: {stats.uptime}
                    </span>
                    <span className="font-data text-[9px] text-tactical-text-dim">
                        Model v{stats.modelVersion} • FL rounds: {stats.flRoundsCompleted}
                    </span>
                </div>
            </div>

            {/* ── Stats Grid ── */}
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
                {[
                    { label: "Local Database", value: stats.localDatabaseSize.toLocaleString(), sub: "profiles", icon: Database, accent: "text-tactical-primary" },
                    { label: "Queries (24h)", value: stats.incomingQueriesLast24h.toString(), sub: "incoming", icon: Search, accent: "text-cyan-400" },
                    { label: "Matches Found", value: stats.successfulMatches.toString(), sub: "above 0.90", icon: CheckCircle, accent: "text-emerald-400" },
                    { label: "Blocked by Law", value: stats.blockedByLaw.toString(), sub: "sovereignty", icon: Lock, accent: "text-amber-400" },
                    { label: "Active Tokens", value: stats.tokensActive.toString(), sub: "ephemeral", icon: Eye, accent: "text-violet-400" },
                    { label: "Avg Latency", value: `${stats.avgSearchLatencyMs.toFixed(1)}ms`, sub: "per search", icon: Zap, accent: "text-tactical-primary" },
                ].map(stat => (
                    <div key={stat.label} className="flex flex-col gap-1 bg-tactical-surface border border-tactical-border rounded px-3 py-2.5">
                        <div className="flex items-center gap-1.5">
                            <stat.icon className={`w-3 h-3 ${stat.accent} flex-shrink-0`} />
                            <span className="font-data text-[7px] uppercase tracking-wider text-tactical-text-dim truncate">
                                {stat.label}
                            </span>
                        </div>
                        <p className={`font-data text-lg font-bold tabular-nums ${stat.accent}`}>
                            {stat.value}
                        </p>
                        <p className="font-data text-[8px] text-zinc-600">{stat.sub}</p>
                    </div>
                ))}
            </div>

            {/* ── Two-Column Layout ── */}
            <div className="grid grid-cols-1 lg:grid-cols-[1fr_1fr] gap-4">

                {/* Sovereignty Toggle */}
                <div className="bg-tactical-surface border border-tactical-border rounded overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2 border-b border-tactical-border">
                        <div className="flex items-center gap-2">
                            <Globe className="w-3.5 h-3.5 text-tactical-primary" />
                            <span className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text-muted">
                                Sovereignty Controls
                            </span>
                        </div>
                        <span className="font-data text-[8px] text-tactical-text-dim">
                            {queryTypes.filter(q => q.enabled).length}/{queryTypes.length} active
                        </span>
                    </div>

                    <div className="divide-y divide-tactical-border/40">
                        {queryTypes.map(qt => (
                            <div key={qt.id} className="flex items-center justify-between px-4 py-2.5 group">
                                <div className="flex-1 min-w-0 mr-3">
                                    <p className={`font-data text-xs font-medium ${qt.enabled ? "text-tactical-text" : "text-zinc-600"}`}>
                                        {qt.label}
                                    </p>
                                    <p className="font-data text-[8px] text-zinc-600 truncate">{qt.description}</p>
                                </div>
                                <button
                                    onClick={() => toggleQueryType(qt.id)}
                                    className={`relative w-9 h-5 rounded-full transition-colors duration-200 flex-shrink-0 ${qt.enabled
                                            ? "bg-tactical-primary/80"
                                            : "bg-zinc-700"
                                        }`}
                                >
                                    <span
                                        className={`absolute top-0.5 w-4 h-4 rounded-full transition-transform duration-200 ${qt.enabled
                                                ? "translate-x-4 bg-white"
                                                : "translate-x-0.5 bg-zinc-500"
                                            }`}
                                    />
                                </button>
                            </div>
                        ))}
                    </div>

                    {/* Kill Switch */}
                    <div className="border-t border-tactical-border px-4 py-3">
                        <button
                            onClick={() => setKillSwitchArmed(!killSwitchArmed)}
                            className={`w-full flex items-center justify-center gap-2 py-2 rounded text-xs font-bold uppercase tracking-wider transition-all duration-300 ${killSwitchArmed
                                    ? "bg-red-500/20 border border-red-500/50 text-red-400 hover:bg-red-500/30"
                                    : "bg-zinc-800 border border-zinc-700 text-zinc-500 hover:border-zinc-600"
                                }`}
                        >
                            {killSwitchArmed ? (
                                <>
                                    <AlertTriangle className="w-3.5 h-3.5" />
                                    <span className="font-data">Kill Switch Armed</span>
                                </>
                            ) : (
                                <>
                                    <Power className="w-3.5 h-3.5" />
                                    <span className="font-data">Arm Kill Switch</span>
                                </>
                            )}
                        </button>
                        {killSwitchArmed && (
                            <p className="font-data text-[8px] text-red-400/60 text-center mt-1.5">
                                Activating will purge all in-memory tokens and disconnect from the orchestrator
                            </p>
                        )}
                    </div>
                </div>

                {/* Recent Queries */}
                <div className="bg-tactical-surface border border-tactical-border rounded overflow-hidden">
                    <div className="flex items-center justify-between px-4 py-2 border-b border-tactical-border">
                        <div className="flex items-center gap-2">
                            <Activity className="w-3.5 h-3.5 text-cyan-400" />
                            <span className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text-muted">
                                Incoming Queries
                            </span>
                        </div>
                        <span className="font-data text-[8px] text-tactical-text-dim">
                            Last 24h
                        </span>
                    </div>

                    <div className="divide-y divide-tactical-border/30 max-h-[320px] overflow-y-auto">
                        {recentQueries.map(q => (
                            <div key={q.id} className="flex items-center justify-between px-4 py-2.5">
                                <div className="flex items-center gap-2.5 min-w-0">
                                    {q.blocked ? (
                                        <XCircle className="w-3.5 h-3.5 text-red-400 flex-shrink-0" />
                                    ) : (
                                        <CheckCircle className="w-3.5 h-3.5 text-tactical-primary flex-shrink-0" />
                                    )}
                                    <div className="min-w-0">
                                        <div className="flex items-center gap-2">
                                            <span className="font-data text-[10px] font-medium text-tactical-text">{q.id}</span>
                                            <span className={`font-data text-[7px] px-1.5 py-0.5 rounded uppercase tracking-wider ${q.blocked
                                                    ? "bg-red-500/15 text-red-400"
                                                    : "bg-tactical-primary/10 text-tactical-primary"
                                                }`}>
                                                {q.blocked ? "BLOCKED" : q.queryType.replace("_", " ")}
                                            </span>
                                        </div>
                                        <p className="font-data text-[8px] text-zinc-600">
                                            from {q.requestingNode} • {q.matches} match{q.matches !== 1 ? "es" : ""}
                                        </p>
                                    </div>
                                </div>
                                <span className="font-data text-[9px] text-tactical-text-dim tabular-nums flex-shrink-0">
                                    {timeAgo(q.timestamp)}
                                </span>
                            </div>
                        ))}
                    </div>

                    {/* Summary bar */}
                    <div className="flex items-center justify-between px-4 py-2 border-t border-tactical-border bg-zinc-900/50">
                        <div className="flex items-center gap-3">
                            <div className="flex items-center gap-1">
                                <div className="w-1.5 h-1.5 rounded-full bg-tactical-primary" />
                                <span className="font-data text-[8px] text-zinc-500">
                                    {recentQueries.filter(q => !q.blocked).length} approved
                                </span>
                            </div>
                            <div className="flex items-center gap-1">
                                <div className="w-1.5 h-1.5 rounded-full bg-red-400" />
                                <span className="font-data text-[8px] text-zinc-500">
                                    {recentQueries.filter(q => q.blocked).length} blocked
                                </span>
                            </div>
                        </div>
                        <div className="flex items-center gap-1">
                            <BarChart3 className="w-3 h-3 text-zinc-600" />
                            <span className="font-data text-[8px] text-zinc-500">
                                {stats.blockedByLaw} total blocked by sovereignty
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
