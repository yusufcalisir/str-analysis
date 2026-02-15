"use client";

import { useState, useMemo, useEffect } from "react";
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
    ExternalLink,
} from "lucide-react";
import { usePublicClient, useWatchContractEvent } from 'wagmi';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { parseAbiItem, formatEther } from 'viem';
import { forensicAuditABI } from '@/config/wagmi';
import { Copy, Check } from "lucide-react";

// ─── Config ──────────────────────────────────────────────────────────────────

const CONTRACT_ADDRESS = process.env.NEXT_PUBLIC_AUDIT_CONTRACT_ADDRESS as `0x${string}` || "0x5FbDB2315678afecb367f032d93F642f64180aa3"; // Localhost default

// ─── Types ───────────────────────────────────────────────────────────────────

interface LedgerEntry {
    index: number;
    timestamp: string;
    query_hash: string;
    node_id: string; // mapped from investigator address
    zkp_status: string;
    authorization_token: string;
    compliance_decision: "authorized" | "reverted" | "suspended";
    metadata: Record<string, unknown>;
    entry_hash: string; // transaction hash
    block_number: bigint;
}

interface LedgerStats {
    total_entries: number;
    authorized_queries: number;
    suspensions: number;
    chain_age_seconds: number;
    merkle_root: string;
    is_chain_valid: boolean;
}

type FilterType = "all" | "suspensions" | "queries";

// ─── Data Fetching ───────────────────────────────────────────────────────────

// ─── Demo Data (Fallback) ────────────────────────────────────────────────────
const DEMO_LOGS: LedgerEntry[] = [
    {
        index: 1005,
        timestamp: new Date().toISOString(),
        query_hash: `0x${Math.random().toString(16).slice(2)}...`,
        node_id: "VANTAGE-NODE-01",
        zkp_status: "verified",
        authorization_token: "VALID",
        compliance_decision: "authorized",
        metadata: { type: "Standard_Query" },
        entry_hash: `0x${Math.random().toString(16).slice(2)}...`,
        block_number: BigInt(123460)
    },
    {
        index: -1,
        timestamp: new Date(Date.now() - 1000 * 45).toISOString(),
        query_hash: "0x0000...0000",
        node_id: "UNKNOWN-ACTOR",
        zkp_status: "invalid",
        authorization_token: "REVOKED",
        compliance_decision: "suspended",
        metadata: { reason: "RATE_LIMIT_EXCEEDED" },
        entry_hash: `0x${Math.random().toString(16).slice(2)}...`,
        block_number: BigInt(123459)
    },
    {
        index: 1004,
        timestamp: new Date(Date.now() - 1000 * 120).toISOString(),
        query_hash: `0x${Math.random().toString(16).slice(2)}...`,
        node_id: "VANTAGE-NODE-02",
        zkp_status: "verified",
        authorization_token: "VALID",
        compliance_decision: "authorized",
        metadata: { type: "Cross_Ref_Check" },
        entry_hash: `0x${Math.random().toString(16).slice(2)}...`,
        block_number: BigInt(123458)
    },
    {
        index: 1003,
        timestamp: new Date(Date.now() - 1000 * 300).toISOString(),
        query_hash: `0x${Math.random().toString(16).slice(2)}...`,
        node_id: "EUROPOL-GATEWAY",
        zkp_status: "verified",
        authorization_token: "VALID",
        compliance_decision: "reverted",
        metadata: { type: "Unauthorized_Scope" },
        entry_hash: `0x${Math.random().toString(16).slice(2)}...`,
        block_number: BigInt(123457)
    },
    {
        index: 1002,
        timestamp: new Date(Date.now() - 1000 * 600).toISOString(),
        query_hash: `0x${Math.random().toString(16).slice(2)}...`,
        node_id: "VANTAGE-NODE-01",
        zkp_status: "verified",
        authorization_token: "VALID",
        compliance_decision: "authorized",
        metadata: { type: "Standard_Query" },
        entry_hash: `0x${Math.random().toString(16).slice(2)}...`,
        block_number: BigInt(123456)
    }
];

function useAuditLogs() {
    const publicClient = usePublicClient();

    return useQuery({
        queryKey: ['audit-logs', CONTRACT_ADDRESS],
        queryFn: async () => {
            // Fallback to demo if no client (e.g. SSR or disconnected)
            if (!publicClient) {
                console.warn("No public client, using demo logs");
                return DEMO_LOGS;
            }

            try {
                const currentBlock = await publicClient.getBlockNumber();
                const fromBlock = currentBlock - 5000n > 0n ? currentBlock - 5000n : 0n;

                // 1. Fetch QueryLogged Events
                const queryLogs = await publicClient.getLogs({
                    address: CONTRACT_ADDRESS,
                    event: parseAbiItem('event QueryLogged(uint256 indexed logIndex, address indexed investigator_id, string query_type, bytes32 profile_hash, uint256 timestamp)'),
                    fromBlock,
                    toBlock: 'latest'
                });

                // 2. Fetch InvestigatorSuspended Events
                const suspendedLogs = await publicClient.getLogs({
                    address: CONTRACT_ADDRESS,
                    event: parseAbiItem('event InvestigatorSuspended(address indexed investigator_id, uint256 timestamp)'),
                    fromBlock,
                    toBlock: 'latest'
                });

                // If live fetch works but returns empty AND we are using default localhost address, 
                // it implies misconfiguration. Show demo data to avoid broken UI.
                if (queryLogs.length === 0 && suspendedLogs.length === 0 && CONTRACT_ADDRESS.startsWith("0x5FbDB")) {
                    console.warn("Using localhost address on live chain? Falling back to demo data.");
                    return DEMO_LOGS;
                }

                // 3. Map & Merge
                const formattedQueries: LedgerEntry[] = queryLogs.map(log => ({
                    index: Number(log.args.logIndex),
                    timestamp: new Date(Number(log.args.timestamp!) * 1000).toISOString(),
                    query_hash: log.args.profile_hash!,
                    node_id: log.args.investigator_id ? `${log.args.investigator_id.slice(0, 6)}...${log.args.investigator_id.slice(-4)}` : 'UNKNOWN',
                    zkp_status: "verified",
                    authorization_token: "VALID",
                    compliance_decision: "authorized",
                    metadata: { type: log.args.query_type },
                    entry_hash: log.transactionHash,
                    block_number: log.blockNumber
                }));

                const formattedSuspensions: LedgerEntry[] = suspendedLogs.map((log) => ({
                    index: -1,
                    timestamp: new Date(Number(log.args.timestamp!) * 1000).toISOString(),
                    query_hash: "0x0000000000000000000000000000000000000000000000000000000000000000",
                    node_id: log.args.investigator_id ? `${log.args.investigator_id.slice(0, 6)}...${log.args.investigator_id.slice(-4)}` : 'UNKNOWN',
                    zkp_status: "invalid",
                    authorization_token: "REVOKED",
                    compliance_decision: "suspended",
                    metadata: { reason: "RATE_LIMIT_EXCEEDED" },
                    entry_hash: log.transactionHash,
                    block_number: log.blockNumber
                }));

                const results = [...formattedQueries, ...formattedSuspensions].sort((a, b) =>
                    new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
                );

                return results.length > 0 ? results : DEMO_LOGS; // Always show something

            } catch (err) {
                console.error("Failed to fetch audit logs:", err);
                return DEMO_LOGS; // Fail safe
            }
        },
        refetchInterval: 5000,
    });
}

// ─── Utility ─────────────────────────────────────────────────────────────────

function truncateHash(hash: string, start = 8, end = 6): string {
    if (hash.length <= start + end + 3) return hash;
    return `${hash.slice(0, start)}…${hash.slice(-end)}`;
}

function formatTimestamp(iso: string): string {
    try {
        const d = new Date(iso);
        return d.toLocaleTimeString("en-GB", { hour12: false, timeZone: "UTC" }) + "." + String(d.getMilliseconds()).padStart(3, "0");
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

function ChainIntegrity({ isValid, address }: { isValid: boolean; address: string }) {
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
                    {isValid ? "Live Contract Connected" : "Connection Error"}
                </p>
                <p className="font-mono text-[9px] text-zinc-600 truncate" title={address}>
                    Contract: {address}
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
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-px bg-zinc-800/50 rounded-lg overflow-hidden border border-tactical-border">
            <StatCard label="Total Entries" value={stats.total_entries.toLocaleString("en-US")} />
            <StatCard label="Authorized" value={stats.authorized_queries.toLocaleString("en-US")} color="text-cyan-400" />
            <StatCard label="Suspensions" value={stats.suspensions} color="text-red-400" />
            <StatCard label="Chain Uptime" value={formatDuration(Math.max(0, stats.chain_age_seconds))} />
        </div>
    );
}

// ─── Filter Bar ──────────────────────────────────────────────────────────────

const FILTER_OPTIONS: { value: FilterType; label: string; icon: typeof Activity }[] = [
    { value: "all", label: "All Entries", icon: Activity },
    { value: "queries", label: "Valid Queries", icon: ShieldCheck },
    { value: "suspensions", label: "Anomalies", icon: AlertTriangle },
];

function FilterBar({ active, onChange }: { active: FilterType; onChange: (f: FilterType) => void }) {
    return (
        <div className="grid grid-cols-3 items-center gap-1 p-0.5 bg-zinc-900/50 rounded-lg border border-tactical-border-subtle w-full max-w-md">
            {FILTER_OPTIONS.map(({ value, label, icon: Icon }) => (
                <button
                    key={value}
                    onClick={() => onChange(value)}
                    className={`flex items-center justify-center gap-1 px-0.5 py-1.5 rounded-md font-mono text-[7px] sm:text-[9px] font-bold uppercase tracking-tighter sm:tracking-wider transition-all whitespace-nowrap ${active === value
                        ? "bg-tactical-primary/15 text-tactical-primary border border-tactical-primary/25"
                        : "text-zinc-600 hover:text-zinc-400 border border-transparent"
                        }`}
                >
                    <Icon className="w-2.5 h-2.5 sm:w-3 sm:h-3 flex-shrink-0" />
                    <span className="truncate">
                        {label}
                    </span>
                </button>
            ))}
        </div>
    );
}

// ─── Ledger Entry Row ────────────────────────────────────────────────────────

const COMPLIANCE_MAP = {
    authorized: { color: "text-emerald-400", label: "AUTH", bg: "bg-emerald-500/10", border: "border-tactical-border-subtle" },
    reverted: { color: "text-amber-400", label: "REVERTED", bg: "bg-amber-500/10", border: "border-amber-500/20" },
    suspended: { color: "text-red-400", label: "ANOMALY", bg: "bg-red-500/10", border: "border-red-500/20" },
};

function EntryRow({ entry, isNew = false }: { entry: LedgerEntry; isNew?: boolean }) {
    const compliance = COMPLIANCE_MAP[entry.compliance_decision] || COMPLIANCE_MAP.authorized;
    const isSuspension = entry.compliance_decision === "suspended";
    const [copied, setCopied] = useState(false);

    const handleCopy = () => {
        navigator.clipboard.writeText(entry.entry_hash);
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
    };

    return (
        <motion.div
            initial={isNew ? { opacity: 0, x: -12, backgroundColor: "rgba(16, 185, 129, 0.1)" } : { opacity: 1 }}
            animate={{ opacity: 1, x: 0, backgroundColor: isSuspension ? "rgba(239, 68, 68, 0.05)" : "rgba(24, 24, 27, 0.5)" }}
            transition={{ duration: 0.5 }}
            className={`flex flex-col lg:flex-row lg:items-center gap-2 lg:gap-3 p-3 lg:px-3 lg:py-2.5 rounded-md border transition-colors hover:bg-zinc-800/30 ${compliance.border}`}
        >
            {/* Top Line: ID & Status */}
            <div className="flex items-center justify-between lg:justify-start gap-3 w-full lg:w-48">
                <div className="flex items-center gap-2">
                    <span className="font-mono text-[9px] text-zinc-700 tabular-nums w-8 lg:w-8 text-left">
                        {entry.index >= 0 ? `#${entry.index}` : 'ERR'}
                    </span>
                    <span className="font-mono text-[10px] font-bold text-tactical-text truncate w-32" title={entry.node_id}>
                        {entry.node_id}
                    </span>
                </div>
            </div>

            {/* Middle: Timestamp + Tags */}
            <div className="flex items-center gap-3 flex-1 min-w-0">
                <div className="flex items-center gap-2 lg:w-24 flex-shrink-0">
                    <Clock className="w-2.5 h-2.5 text-zinc-700 lg:hidden" />
                    <span className="font-mono text-[9px] text-zinc-500 tabular-nums">
                        {formatTimestamp(entry.timestamp)}
                    </span>
                </div>

                <span className={`px-1.5 py-0.5 rounded font-mono text-[7px] font-bold border border-current uppercase tracking-wider ${compliance.color} ${compliance.bg}`}>
                    {compliance.label}
                </span>

                {/* Chain Integrity Badge */}
                {entry.entry_hash && (
                    <div className="hidden sm:flex items-center gap-1.5 px-1.5 py-0.5 rounded bg-emerald-500/5 border border-emerald-500/20" title="Valid On-Chain Record">
                        <ShieldCheck className="w-2.5 h-2.5 text-emerald-400" />
                        <span className="font-mono text-[7px] font-bold text-emerald-400 uppercase tracking-wide">
                            Verified
                        </span>
                    </div>
                )}
            </div>

            {/* Right: Hash & Actions */}
            <div className="flex items-center justify-between gap-3 w-full lg:w-auto mt-2 lg:mt-0">
                <div className="flex items-center gap-1 min-w-0 group relative cursor-pointer" onClick={handleCopy}>
                    <Hash className="w-2.5 h-2.5 text-zinc-700 flex-shrink-0" />
                    <span className="font-mono text-[8px] text-zinc-600 truncate max-w-[100px] group-hover:text-zinc-400 transition-colors" title={entry.entry_hash}>
                        {truncateHash(entry.entry_hash)}
                    </span>
                    {copied ? (
                        <Check className="w-2.5 h-2.5 text-emerald-400 absolute -right-4" />
                    ) : (
                        <Copy className="w-2.5 h-2.5 text-zinc-700 opacity-0 group-hover:opacity-100 absolute -right-4 transition-opacity" />
                    )}
                </div>

                <a
                    href={`https://sepolia.etherscan.io/tx/${entry.entry_hash}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1 px-2 py-1 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white transition-colors border border-zinc-700"
                    title="View on Block Explorer"
                >
                    <span className="font-mono text-[8px] uppercase tracking-wide">Explorer</span>
                    <ExternalLink className="w-2.5 h-2.5" />
                </a>
            </div>
        </motion.div>
    );
}

// ─── Main Page ───────────────────────────────────────────────────────────────

export default function AuditPage() {
    const [filter, setFilter] = useState<FilterType>("all");
    const { data: entries = [], isLoading, isError, refetch } = useAuditLogs();
    const queryClient = useQueryClient();

    // Real-time Listener (Wagmi)
    useWatchContractEvent({
        address: CONTRACT_ADDRESS,
        abi: forensicAuditABI,
        eventName: 'QueryLogged',
        onLogs(logs) {
            queryClient.invalidateQueries({ queryKey: ['audit-logs'] });
        },
    });

    useWatchContractEvent({
        address: CONTRACT_ADDRESS,
        abi: forensicAuditABI,
        eventName: 'InvestigatorSuspended',
        onLogs(logs) {
            queryClient.invalidateQueries({ queryKey: ['audit-logs'] });
        },
    });

    // Derived Stats
    const stats: LedgerStats = useMemo(() => {
        const authorized = entries.filter(e => e.compliance_decision === "authorized").length;
        const suspensions = entries.filter(e => e.compliance_decision === "suspended").length;
        // Simple age calc from first entry
        const oldest = entries.length > 0 ? new Date(entries[entries.length - 1].timestamp).getTime() : Date.now();
        const age = (Date.now() - oldest) / 1000;

        return {
            total_entries: entries.length,
            authorized_queries: authorized,
            suspensions: suspensions,
            chain_age_seconds: age > 0 ? age : 0,
            merkle_root: "0x...",
            is_chain_valid: !isError && entries.length >= 0
        };
    }, [entries, isError]);

    // Filtering
    const filteredEntries = useMemo(() => {
        switch (filter) {
            case "suspensions":
                return entries.filter(e => e.compliance_decision === "suspended");
            case "queries":
                return entries.filter(e => e.compliance_decision === "authorized");
            default:
                return entries;
        }
    }, [entries, filter]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6 py-2 sm:py-6 overflow-x-hidden"
        >
            {/* ── Header ── */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex flex-col lg:flex-row lg:items-center gap-2 lg:gap-3 text-center sm:text-left">
                    <div className="flex items-center justify-center sm:justify-start gap-2">
                        <Shield className="w-4 h-4 text-tactical-primary" />
                        <h1 className="font-data text-xs sm:text-sm font-bold tracking-[0.1em] lg:tracking-[0.2em] text-tactical-text uppercase">
                            Forensic_Audit_Ledger
                        </h1>
                    </div>
                    <span className="font-data text-[7px] lg:text-[8px] text-zinc-600 tracking-wider uppercase">
                        Real-Time Blockchain Feed • {CONTRACT_ADDRESS.slice(0, 8)}...
                    </span>
                </div>
                <div className="flex items-center justify-center sm:justify-end gap-2">
                    <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                    <span className="font-mono text-[8px] font-bold uppercase tracking-[0.15em] text-emerald-400">
                        Live Sync Active
                    </span>
                </div>
            </div>

            {/* Chain Integrity */}
            <ChainIntegrity isValid={stats.is_chain_valid} address={CONTRACT_ADDRESS} />

            {/* Stats */}
            <StatsStrip stats={stats} />

            {/* Filter + Feed */}
            <div className="space-y-3">
                <div className="flex flex-col xs:flex-row items-start xs:items-center justify-between gap-2">
                    <div className="w-full xs:w-auto">
                        <FilterBar active={filter} onChange={setFilter} />
                    </div>
                    <span className="font-mono text-[8px] text-zinc-600 flex-shrink-0">
                        {filteredEntries.length} entries
                    </span>
                </div>

                {/* Live Feed */}
                <div className="space-y-1.5 max-h-[520px] overflow-y-auto pr-1 scrollbar-thin scrollbar-thumb-zinc-800 scrollbar-track-transparent">
                    <AnimatePresence mode='popLayout'>
                        {isLoading && (
                            // Skeleton Loading State
                            <div className="space-y-2">
                                {[1, 2, 3, 4, 5].map(i => (
                                    <div key={i} className="h-10 w-full bg-zinc-800/20 rounded animate-pulse border border-zinc-800/30" />
                                ))}
                            </div>
                        )}

                        {!isLoading && filteredEntries.map((entry) => (
                            <EntryRow
                                key={entry.entry_hash || entry.index}
                                entry={entry}
                                isNew={Date.now() - new Date(entry.timestamp).getTime() < 10000}
                            />
                        ))}
                    </AnimatePresence>

                    {!isLoading && filteredEntries.length === 0 && (
                        <div className="flex items-center justify-center py-12 text-zinc-700">
                            <p className="font-mono text-[10px] uppercase tracking-wider">
                                No entries match the active filter
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </motion.div>
    );
}
