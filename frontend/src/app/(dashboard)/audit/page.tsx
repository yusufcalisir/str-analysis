"use client";

import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Shield,
    ShieldCheck,
    ShieldAlert,
    ShieldX,
    Activity,
    Clock,
    Hash,
    AlertTriangle,
    Globe,
    Zap,
    XCircle,
    Eye,
    Link2,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface LedgerEntry {
    index: number;
    timestamp: string;
    query_hash: string;
    node_id: string;
    zkp_status: string;
    authorization_token: string;
    compliance_decision: string;
    metadata: Record<string, unknown>;
    previous_hash: string;
    entry_hash: string;
    merkle_root: string;
}

interface LedgerStats {
    total_entries: number;
    verified_proofs: number;
    invalid_proofs: number;
    reverted_queries: number;
    authorized_queries: number;
    chain_age_seconds: number;
    merkle_root: string;
    is_chain_valid: boolean;
}

type FilterType = "all" | "failed" | "cross_border" | "anomalies";

// ─── Mock Data ───────────────────────────────────────────────────────────────

const MOCK_STATS: LedgerStats = {
    total_entries: 1_247,
    verified_proofs: 1_089,
    invalid_proofs: 23,
    reverted_queries: 47,
    authorized_queries: 1_177,
    chain_age_seconds: 604_800,
    merkle_root: "a8f3c91d2e7b405689012345abcdef67890abcde12345678fedcba0987654321",
    is_chain_valid: true,
};

const MOCK_ENTRIES: LedgerEntry[] = [
    {
        index: 1246,
        timestamp: "2026-02-12T20:44:12+00:00",
        query_hash: "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        node_id: "BKA-DE",
        zkp_status: "verified",
        authorization_token: "LRT-a3f8c91d",
        compliance_decision: "authorized",
        metadata: { cross_border: true, target_country: "NL", crime: "HOMICIDE" },
        previous_hash: "7d2c3f8a1b4e5d6c9a0b8e7f6d5c4a3b2d1e0f9c8b7a6e5d4c3b2a1f0e9d8c",
        entry_hash: "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b",
        merkle_root: "a8f3c91d2e7b405689012345abcdef67890abcde12345678fedcba0987654321",
    },
    {
        index: 1245,
        timestamp: "2026-02-12T20:43:58+00:00",
        query_hash: "f4a1b5c3d8e2f6709a3b4c5d6e7f8091a2b3c4d5e6f70819a2b3c4d5e6f7081",
        node_id: "EUROPOL-NL",
        zkp_status: "verified",
        authorization_token: "LRT-b4c92d11",
        compliance_decision: "authorized",
        metadata: { cross_border: false },
        previous_hash: "8e3d4f9b2c5e6d7a0b1c9e8f7d6c5a4b3e2d1f0a9c8b7e6d5c4b3a2f1e0d9c8",
        entry_hash: "7d2c3f8a1b4e5d6c9a0b8e7f6d5c4a3b2d1e0f9c8b7a6e5d4c3b2a1f0e9d8c",
        merkle_root: "b7e2d81c3f6a5049780123456bcdef7890abcde12345678fedcba0987654320",
    },
    {
        index: 1244,
        timestamp: "2026-02-12T20:43:41+00:00",
        query_hash: "d5b2c6d4e9f3a7810b4c5d6e7f80a1b2c3d4e5f6a7b8190c2d3e4f5a6b7c8d9",
        node_id: "FBI-US",
        zkp_status: "invalid",
        authorization_token: "",
        compliance_decision: "reverted",
        metadata: { cross_border: true, target_country: "JP", crime: "FRAUD", revert_gate: "cross_border" },
        previous_hash: "9f4e5a0c3d6e7b8a1c2d0f9e8d7c6b5a4e3d2c1b0a9f8e7d6c5b4a3e2d1c0b9",
        entry_hash: "8e3d4f9b2c5e6d7a0b1c9e8f7d6c5a4b3e2d1f0a9c8b7e6d5c4b3a2f1e0d9c8",
        merkle_root: "c6d1e70b2e5940387012345abcdef67890abcde12345678fedcba098765431f",
    },
    {
        index: 1243,
        timestamp: "2026-02-12T20:42:19+00:00",
        query_hash: "a6c3d7e5f0a4b891c5d6e7f8a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
        node_id: "NCA-UK",
        zkp_status: "verified",
        authorization_token: "LRT-e8f0a367",
        compliance_decision: "authorized",
        metadata: { cross_border: true, target_country: "DE", crime: "TERRORISM" },
        previous_hash: "0a5f6b1d4e7f8c9a2d3e1f0b9c8d7e6a5f4e3d2c1b0a9f8e7d6c5b4a3e2d1c0",
        entry_hash: "9f4e5a0c3d6e7b8a1c2d0f9e8d7c6b5a4e3d2c1b0a9f8e7d6c5b4a3e2d1c0b9",
        merkle_root: "d5e0f69a1d489327601234abcdef67890abcde12345678fedcba098765431e",
    },
    {
        index: 1242,
        timestamp: "2026-02-12T20:41:55+00:00",
        query_hash: "b7d4e8f6a1b5c902d6e7f8a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2",
        node_id: "INTERPOL-EU",
        zkp_status: "verified",
        authorization_token: "LRT-d4c7b290",
        compliance_decision: "authorized",
        metadata: { cross_border: true, target_country: "DE", crime: "WAR_CRIMES" },
        previous_hash: "1b6a7c2e5f8a9d0b3e4f2a1c0d9e8f7b6a5e4d3c2b1a0f9e8d7c6b5a4e3d2c1",
        entry_hash: "0a5f6b1d4e7f8c9a2d3e1f0b9c8d7e6a5f4e3d2c1b0a9f8e7d6c5b4a3e2d1c0",
        merkle_root: "e4f1e58b0c37821650123abcdef67890abcde12345678fedcba098765431d",
    },
];

// ─── Utility ─────────────────────────────────────────────────────────────────

function truncateHash(hash: string, start = 8, end = 6): string {
    if (hash.length <= start + end + 3) return hash;
    return `${hash.slice(0, start)}…${hash.slice(-end)}`;
}

function formatTimestamp(iso: string): string {
    try {
        const d = new Date(iso);
        return d.toLocaleTimeString("en-GB", { hour12: false }) + "." + String(d.getMilliseconds()).padStart(3, "0");
    } catch {
        return iso;
    }
}

function formatDuration(seconds: number): string {
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(0)}m`;
    if (seconds < 86400) return `${(seconds / 3600).toFixed(1)}h`;
    return `${(seconds / 86400).toFixed(1)}d`;
}

// ─── Chain Integrity Indicator ───────────────────────────────────────────────

function ChainIntegrity({ isValid, merkleRoot }: { isValid: boolean; merkleRoot: string }) {
    return (
        <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex items-center gap-3 px-4 py-3 rounded-lg border ${isValid
                ? "bg-emerald-500/5 border-emerald-500/20"
                : "bg-red-500/5 border-red-500/20"
                }`}
        >
            <motion.div
                animate={isValid ? { scale: [1, 1.15, 1] } : {}}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            >
                {isValid ? (
                    <ShieldCheck className="w-5 h-5 text-emerald-400" />
                ) : (
                    <ShieldX className="w-5 h-5 text-red-400" />
                )}
            </motion.div>
            <div className="flex-1 min-w-0">
                <p className={`font-mono text-[10px] font-bold uppercase tracking-[0.15em] ${isValid ? "text-emerald-400" : "text-red-400"
                    }`}>
                    {isValid ? "Chain Integrity Verified" : "Chain Integrity Compromised"}
                </p>
                <p className="font-mono text-[9px] text-zinc-600 truncate" title={merkleRoot}>
                    Merkle Root: {truncateHash(merkleRoot, 12, 8)}
                </p>
            </div>
            <div className={`w-2 h-2 rounded-full ${isValid ? "bg-emerald-400" : "bg-red-400"}`}>
                {isValid && (
                    <span className="block w-2 h-2 rounded-full bg-emerald-400 animate-ping opacity-40" />
                )}
            </div>
        </motion.div>
    );
}

// ─── Stats Strip ─────────────────────────────────────────────────────────────

function StatCard({ label, value, color = "text-tactical-text" }: { label: string; value: string | number; color?: string }) {
    return (
        <div className="text-center px-3 py-2">
            <p className={`font-mono text-base font-bold tabular-nums ${color}`}>{value}</p>
            <p className="font-mono text-[7px] uppercase tracking-[0.15em] text-zinc-600 mt-0.5">{label}</p>
        </div>
    );
}

function StatsStrip({ stats }: { stats: LedgerStats }) {
    return (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-px bg-zinc-800/50 rounded-lg overflow-hidden border border-tactical-border">
            <StatCard label="Total Entries" value={stats.total_entries.toLocaleString()} />
            <StatCard label="Verified Proofs" value={stats.verified_proofs.toLocaleString()} color="text-emerald-400" />
            <StatCard label="Invalid Proofs" value={stats.invalid_proofs} color="text-red-400" />
            <StatCard label="Reverted Queries" value={stats.reverted_queries} color="text-amber-400" />
            <StatCard label="Authorized" value={stats.authorized_queries.toLocaleString()} color="text-cyan-400" />
            <StatCard label="Chain Age" value={formatDuration(stats.chain_age_seconds)} />
        </div>
    );
}

// ─── Filter Bar ──────────────────────────────────────────────────────────────

const FILTER_OPTIONS: { value: FilterType; label: string; icon: typeof Activity }[] = [
    { value: "all", label: "All Entries", icon: Activity },
    { value: "failed", label: "Failed Auth", icon: XCircle },
    { value: "cross_border", label: "Cross-Border", icon: Globe },
    { value: "anomalies", label: "Anomalies", icon: AlertTriangle },
];

function FilterBar({ active, onChange }: { active: FilterType; onChange: (f: FilterType) => void }) {
    return (
        <div className="flex items-center gap-1 p-1 bg-zinc-900/50 rounded-lg border border-tactical-border-subtle">
            {FILTER_OPTIONS.map(({ value, label, icon: Icon }) => (
                <button
                    key={value}
                    onClick={() => onChange(value)}
                    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md font-mono text-[9px] font-bold uppercase tracking-wider transition-all ${active === value
                        ? "bg-tactical-primary/15 text-tactical-primary border border-tactical-primary/25"
                        : "text-zinc-600 hover:text-zinc-400 border border-transparent"
                        }`}
                >
                    <Icon className="w-3 h-3" />
                    {label}
                </button>
            ))}
        </div>
    );
}

// ─── Ledger Entry Row ────────────────────────────────────────────────────────

const ZKP_BADGE_MAP: Record<string, { color: string; bg: string; icon: typeof ShieldCheck }> = {
    verified: { color: "text-emerald-400", bg: "bg-emerald-500/10", icon: ShieldCheck },
    invalid: { color: "text-red-400", bg: "bg-red-500/10", icon: ShieldAlert },
    pending: { color: "text-cyan-400", bg: "bg-cyan-500/10", icon: Clock },
    skipped: { color: "text-zinc-600", bg: "bg-zinc-800/50", icon: Shield },
};

const COMPLIANCE_MAP: Record<string, { color: string; label: string }> = {
    authorized: { color: "text-emerald-400", label: "AUTH" },
    reverted: { color: "text-red-400", label: "REVERTED" },
    bypassed: { color: "text-amber-400", label: "BYPASS" },
};

function EntryRow({ entry, isNew = false }: { entry: LedgerEntry; isNew?: boolean }) {
    const zkp = ZKP_BADGE_MAP[entry.zkp_status] || ZKP_BADGE_MAP.skipped;
    const ZkpIcon = zkp.icon;
    const compliance = COMPLIANCE_MAP[entry.compliance_decision] || COMPLIANCE_MAP.authorized;
    const isCrossBorder = entry.metadata?.cross_border === true;

    return (
        <motion.div
            initial={isNew ? { opacity: 0, x: -12 } : { opacity: 1 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3 }}
            className={`flex items-center gap-3 px-3 py-2.5 rounded-md border transition-colors hover:bg-zinc-800/30 ${entry.compliance_decision === "reverted"
                ? "border-red-500/15 bg-red-500/5"
                : "border-tactical-border-subtle bg-tactical-surface/50"
                }`}
        >
            {/* Index */}
            <span className="font-mono text-[9px] text-zinc-700 tabular-nums w-10 text-right flex-shrink-0">
                #{entry.index}
            </span>

            {/* Timestamp */}
            <span className="font-mono text-[9px] text-zinc-500 tabular-nums w-24 flex-shrink-0">
                {formatTimestamp(entry.timestamp)}
            </span>

            {/* Node ID */}
            <span className="font-mono text-[10px] font-bold text-tactical-text w-28 truncate flex-shrink-0">
                {entry.node_id}
            </span>

            {/* ZKP Badge */}
            <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded ${zkp.bg} flex-shrink-0`}>
                <ZkpIcon className={`w-2.5 h-2.5 ${zkp.color}`} />
                <span className={`font-mono text-[7px] font-bold uppercase tracking-wider ${zkp.color}`}>
                    {entry.zkp_status}
                </span>
            </div>

            {/* Compliance */}
            <span className={`font-mono text-[7px] font-bold uppercase tracking-wider ${compliance.color} w-16 flex-shrink-0`}>
                {compliance.label}
            </span>

            {/* Cross-border indicator */}
            {isCrossBorder && (
                <div className="flex items-center gap-1 px-1.5 py-0.5 rounded bg-indigo-500/10 flex-shrink-0">
                    <Globe className="w-2.5 h-2.5 text-indigo-400" />
                    <span className="font-mono text-[7px] font-bold text-indigo-400">
                        {(entry.metadata?.target_country as string) || "??"}
                    </span>
                </div>
            )}

            {/* Entry Hash */}
            <div className="flex-1 min-w-0 flex items-center gap-1">
                <Hash className="w-2.5 h-2.5 text-zinc-700 flex-shrink-0" />
                <span className="font-mono text-[8px] text-zinc-600 truncate" title={entry.entry_hash}>
                    {truncateHash(entry.entry_hash)}
                </span>
            </div>

            {/* Chain link indicator */}
            <Link2 className="w-3 h-3 text-zinc-800 flex-shrink-0" />
        </motion.div>
    );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function AuditPage() {
    const [filter, setFilter] = useState<FilterType>("all");
    const [stats] = useState<LedgerStats>(MOCK_STATS);
    const [entries] = useState<LedgerEntry[]>(MOCK_ENTRIES);

    const filteredEntries = useMemo(() => {
        switch (filter) {
            case "failed":
                return entries.filter(
                    (e) => e.compliance_decision === "reverted" || e.zkp_status === "invalid"
                );
            case "cross_border":
                return entries.filter((e) => e.metadata?.cross_border === true);
            case "anomalies":
                return entries.filter((e) => e.metadata?.anomaly_flagged === true);
            default:
                return entries;
        }
    }, [entries, filter]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="space-y-5"
        >
            {/* ── Header ── */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-tactical-primary" />
                    <h1 className="font-data text-xs font-bold tracking-[0.2em] text-tactical-text uppercase">
                        Forensic_Audit_Ledger
                    </h1>
                    <span className="font-data text-[8px] text-zinc-600 tracking-wider uppercase">
                        Immutable Chain Explorer — Phase 5.1
                    </span>
                </div>
                <div className="flex items-center gap-2">
                    <Eye className="w-3.5 h-3.5 text-tactical-primary" />
                    <span className="font-mono text-[8px] font-bold uppercase tracking-[0.15em] text-tactical-primary">
                        Auditor View
                    </span>
                </div>
            </div>

            {/* Chain Integrity */}
            <ChainIntegrity isValid={stats.is_chain_valid} merkleRoot={stats.merkle_root} />

            {/* Stats */}
            <StatsStrip stats={stats} />

            {/* Filter + Feed */}
            <div className="space-y-3">
                <div className="flex items-center justify-between">
                    <FilterBar active={filter} onChange={setFilter} />
                    <span className="font-mono text-[8px] text-zinc-600">
                        {filteredEntries.length} entries
                    </span>
                </div>

                {/* Live Feed */}
                <div className="space-y-1.5 max-h-[520px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-zinc-800 scrollbar-track-transparent">
                    <AnimatePresence>
                        {filteredEntries.map((entry, i) => (
                            <EntryRow key={entry.index} entry={entry} isNew={i === 0} />
                        ))}
                    </AnimatePresence>

                    {filteredEntries.length === 0 && (
                        <div className="flex items-center justify-center py-12 text-zinc-700">
                            <p className="font-mono text-[10px] uppercase tracking-wider">
                                No entries match the active filter
                            </p>
                        </div>
                    )}
                </div>
            </div>

            {/* Merkle Root Footer */}
            <div className="flex items-center justify-between border-t border-tactical-border-subtle pt-3">
                <div className="flex items-center gap-2">
                    <Zap className="w-3 h-3 text-zinc-700" />
                    <span className="font-mono text-[7px] text-zinc-700 uppercase tracking-wider">
                        Powered by ForensicLedger v1 — SHA-256 Merkle Chain
                    </span>
                </div>
                <span className="font-mono text-[8px] text-zinc-700 tabular-nums">
                    {stats.total_entries.toLocaleString()} blocks
                </span>
            </div>
        </motion.div>
    );
}
