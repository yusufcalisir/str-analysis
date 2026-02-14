"use client";

import { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Dna,
    AlertTriangle,
    CheckCircle,
    Shield,
    ShieldCheck,
    ShieldAlert,
    ShieldQuestion,
    Loader2,
    MapPin,
    Clock,
    ChevronDown,
    ChevronUp,
    Info,
    Fingerprint,
    Hash,
    Timer,
} from "lucide-react";
import { useState } from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ZKPMetadata {
    commitmentHash: string;
    proofSizeBytes: number;
    verificationMs: number;
    queryId: string;
}

type ZKPStatus = "verifying" | "verified" | "invalid" | "none";

interface MatchResultData {
    profileId: string;
    nodeId: string;
    rawSimilarity: number;
    penalizedScore: number;
    activeLoci: number;
    totalLoci: number;
    completenessRatio: number;
    qualityTier: "complete" | "partial" | "degraded";
    missingLociQuery: string[];
    missingLociTarget: string[];
    zkpStatus?: ZKPStatus;
    zkpMetadata?: ZKPMetadata;
}

// ─── Tier Config ─────────────────────────────────────────────────────────────

const TIER_CONFIG = {
    complete: {
        label: "Complete Profile",
        color: "text-emerald-400",
        bg: "bg-emerald-500",
        bgFaint: "bg-emerald-500/10",
        border: "border-emerald-500/20",
        barColor: "bg-emerald-500",
        icon: CheckCircle,
        description: "≥18 markers compared",
    },
    partial: {
        label: "Partial Profile",
        color: "text-amber-400",
        bg: "bg-amber-500",
        bgFaint: "bg-amber-500/10",
        border: "border-amber-500/20",
        barColor: "bg-amber-500",
        icon: AlertTriangle,
        description: "10–17 markers compared",
    },
    degraded: {
        label: "Critically Degraded",
        color: "text-red-400",
        bg: "bg-red-500",
        bgFaint: "bg-red-500/10",
        border: "border-red-500/20",
        barColor: "bg-red-500",
        icon: AlertTriangle,
        description: "<10 markers compared",
    },
} as const;

// ─── All 24 Standard CODIS+ESS+Penta Loci ───────────────────────────────────

const ALL_LOCI = [
    "AMEL", "CSF1PO", "D1S1656", "D2S441", "D2S1338", "D3S1358",
    "D5S818", "D7S820", "D8S1179", "D10S1248", "D12S391", "D13S317",
    "D16S539", "D18S51", "D19S433", "D21S11", "D22S1045", "FGA",
    "PENTA_D", "PENTA_E", "SE33", "TH01", "TPOX", "VWA",
];

// ─── Completeness Ring ───────────────────────────────────────────────────────

function CompletenessRing({
    ratio,
    tier,
    size = 52,
}: {
    ratio: number;
    tier: "complete" | "partial" | "degraded";
    size?: number;
}) {
    const config = TIER_CONFIG[tier];
    const radius = (size - 6) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - ratio);
    const pct = Math.round(ratio * 100);

    const strokeColor =
        tier === "complete" ? "#10b981"
            : tier === "partial" ? "#f59e0b"
                : "#ef4444";

    return (
        <div className="relative flex items-center justify-center" style={{ width: size, height: size }}>
            <svg width={size} height={size} className="transform -rotate-90">
                {/* Background ring */}
                <circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke="rgba(255,255,255,0.06)"
                    strokeWidth={3}
                />
                {/* Progress ring */}
                <motion.circle
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    fill="none"
                    stroke={strokeColor}
                    strokeWidth={3}
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    initial={{ strokeDashoffset: circumference }}
                    animate={{ strokeDashoffset: offset }}
                    transition={{ duration: 0.8, ease: "easeOut" }}
                />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
                <span className={`font-data text-[10px] font-bold tabular-nums ${config.color}`}>
                    {pct}%
                </span>
            </div>
        </div>
    );
}

// ─── Score Display ───────────────────────────────────────────────────────────

function ScoreDisplay({
    label,
    value,
    variant = "default",
}: {
    label: string;
    value: number;
    variant?: "default" | "primary" | "muted";
}) {
    const pct = (value * 100).toFixed(2);
    const colorClass =
        variant === "primary" ? "text-tactical-primary"
            : variant === "muted" ? "text-zinc-500"
                : value >= 0.95 ? "text-emerald-400"
                    : value >= 0.80 ? "text-cyan-400"
                        : value >= 0.60 ? "text-amber-400"
                            : "text-red-400";

    return (
        <div className="flex flex-col gap-0.5">
            <span className="font-data text-[7px] uppercase tracking-wider text-zinc-600">
                {label}
            </span>
            <span className={`font-data text-[14px] font-bold tabular-nums ${colorClass}`}>
                {pct}%
            </span>
        </div>
    );
}

// ─── Loci Grid ───────────────────────────────────────────────────────────────

function LociGrid({
    missingQuery,
    missingTarget,
}: {
    missingQuery: string[];
    missingTarget: string[];
}) {
    const missingQSet = new Set(missingQuery);
    const missingTSet = new Set(missingTarget);

    return (
        <div className="grid grid-cols-6 sm:grid-cols-8 gap-1">
            {ALL_LOCI.map((locus) => {
                const isMissingQ = missingQSet.has(locus);
                const isMissingT = missingTSet.has(locus);
                const isActive = !isMissingQ && !isMissingT;

                let bgClass = "bg-emerald-500/15 border-emerald-500/30 text-emerald-400";
                let title = "Compared";

                if (isMissingQ && isMissingT) {
                    bgClass = "bg-red-500/10 border-red-500/20 text-red-500/60";
                    title = "Missing from both";
                } else if (isMissingQ) {
                    bgClass = "bg-amber-500/10 border-amber-500/20 text-amber-500/60";
                    title = "Missing from query";
                } else if (isMissingT) {
                    bgClass = "bg-violet-500/10 border-violet-500/20 text-violet-500/60";
                    title = "Missing from target";
                }

                return (
                    <div
                        key={locus}
                        title={`${locus}: ${title}`}
                        className={`text-center text-[7px] font-data font-bold py-1 px-0.5 rounded border transition-all cursor-default ${bgClass}`}
                    >
                        {locus}
                    </div>
                );
            })}
        </div>
    );
}

// ─── Crypto Status Badge ─────────────────────────────────────────────────────

const ZKP_STATUS_CONFIG = {
    verifying: {
        label: "Verifying Proof",
        color: "text-cyan-400",
        bg: "bg-cyan-500/10",
        border: "border-cyan-500/25",
        icon: Loader2,
        pulse: true,
    },
    verified: {
        label: "Proof Verified",
        color: "text-emerald-400",
        bg: "bg-emerald-500/10",
        border: "border-emerald-500/25",
        icon: ShieldCheck,
        pulse: false,
    },
    invalid: {
        label: "Invalid Proof",
        color: "text-red-400",
        bg: "bg-red-500/10",
        border: "border-red-500/25",
        icon: ShieldAlert,
        pulse: false,
    },
    none: {
        label: "No ZKP",
        color: "text-zinc-600",
        bg: "bg-zinc-800/50",
        border: "border-zinc-700/30",
        icon: ShieldQuestion,
        pulse: false,
    },
} as const;

function CryptoStatusBadge({ status }: { status: ZKPStatus }) {
    const cfg = ZKP_STATUS_CONFIG[status];
    const Icon = cfg.icon;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className={`flex items-center gap-1.5 px-2 py-1 rounded border ${cfg.bg} ${cfg.border}`}
        >
            <motion.div
                animate={cfg.pulse ? { opacity: [1, 0.4, 1] } : {}}
                transition={cfg.pulse ? { duration: 1.5, repeat: Infinity, ease: "easeInOut" } : {}}
            >
                <Icon className={`w-3 h-3 ${cfg.color} ${status === "verifying" ? "animate-spin" : ""}`} />
            </motion.div>
            <span className={`font-data text-[7px] font-bold uppercase tracking-wider ${cfg.color}`}>
                {cfg.label}
            </span>
        </motion.div>
    );
}

// ─── Proof Metadata Panel ────────────────────────────────────────────────────

function ProofMetadataPanel({ metadata }: { metadata: ZKPMetadata }) {
    return (
        <div className="p-2.5 rounded bg-zinc-900/70 border border-emerald-500/10">
            <div className="flex items-center gap-1.5 mb-2">
                <Fingerprint className="w-3 h-3 text-emerald-500" />
                <span className="font-data text-[8px] font-bold uppercase tracking-[0.12em] text-emerald-400">
                    ZKP Proof Metadata
                </span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1.5">
                <div className="flex items-center gap-1.5">
                    <Hash className="w-2.5 h-2.5 text-zinc-600 flex-shrink-0" />
                    <div className="min-w-0">
                        <span className="block font-data text-[6px] uppercase tracking-wider text-zinc-600">Commitment</span>
                        <span className="block font-data text-[9px] text-zinc-400 truncate" title={metadata.commitmentHash}>
                            {metadata.commitmentHash.slice(0, 16)}…{metadata.commitmentHash.slice(-8)}
                        </span>
                    </div>
                </div>
                <div className="flex items-center gap-1.5">
                    <Fingerprint className="w-2.5 h-2.5 text-zinc-600 flex-shrink-0" />
                    <div>
                        <span className="block font-data text-[6px] uppercase tracking-wider text-zinc-600">Query ID</span>
                        <span className="block font-data text-[9px] text-zinc-400">{metadata.queryId}</span>
                    </div>
                </div>
                <div className="flex items-center gap-1.5">
                    <Shield className="w-2.5 h-2.5 text-zinc-600 flex-shrink-0" />
                    <div>
                        <span className="block font-data text-[6px] uppercase tracking-wider text-zinc-600">Proof Size</span>
                        <span className="block font-data text-[9px] text-zinc-400">{metadata.proofSizeBytes} bytes</span>
                    </div>
                </div>
                <div className="flex items-center gap-1.5">
                    <Timer className="w-2.5 h-2.5 text-zinc-600 flex-shrink-0" />
                    <div>
                        <span className="block font-data text-[6px] uppercase tracking-wider text-zinc-600">Verification</span>
                        <span className="block font-data text-[9px] text-emerald-400 font-bold">{metadata.verificationMs.toFixed(1)}ms</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function MatchResultCard({ match }: { match: MatchResultData }) {
    const [expanded, setExpanded] = useState(false);
    const tier = TIER_CONFIG[match.qualityTier];
    const TierIcon = tier.icon;

    const allMissing = useMemo(() => {
        const combined = new Set([...match.missingLociQuery, ...match.missingLociTarget]);
        return Array.from(combined).sort();
    }, [match.missingLociQuery, match.missingLociTarget]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            className={`border rounded overflow-hidden bg-tactical-bg ${tier.border}`}
        >
            {/* ── Main Row ── */}
            {/* ── Main Layout: Responsive Stack ── */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-4 px-4 py-4 sm:py-3">
                <div className="flex items-center gap-4 flex-1 min-w-0">
                    {/* Completeness Ring */}
                    <div className="flex-shrink-0">
                        <CompletenessRing
                            ratio={match.completenessRatio}
                            tier={match.qualityTier}
                            size={48}
                        />
                    </div>

                    {/* Profile Info */}
                    <div className="flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2 mb-1.5">
                            <span className="font-data text-[11px] font-bold text-tactical-text truncate max-w-[140px] sm:max-w-none">
                                {match.profileId}
                            </span>
                            <div className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[7px] font-data font-bold uppercase tracking-wider ${tier.bgFaint} ${tier.color} ${tier.border} border whitespace-nowrap`}>
                                <TierIcon className="w-2.5 h-2.5" />
                                {tier.label}
                            </div>
                        </div>
                        <div className="flex items-center gap-3 text-[9px] font-data text-zinc-500">
                            <span className="flex items-center gap-1">
                                <MapPin className="w-2.5 h-2.5" />
                                {match.nodeId}
                            </span>
                            <span className="flex items-center gap-1 border-l border-zinc-800 pl-3">
                                <Dna className="w-2.5 h-2.5" />
                                {match.activeLoci}/{match.totalLoci} loci
                            </span>
                        </div>
                    </div>
                </div>

                {/* Stats and Controls Row */}
                <div className="flex items-center justify-between sm:justify-end gap-6 w-full sm:w-auto mt-2 sm:mt-0 pt-3 sm:pt-0 border-t border-zinc-900/50 sm:border-0">
                    {/* Scores */}
                    <div className="flex items-center gap-4">
                        <ScoreDisplay label="Raw Sim" value={match.rawSimilarity} variant="muted" />
                        <ScoreDisplay label="Penalized" value={match.penalizedScore} variant="primary" />
                    </div>

                    <div className="flex items-center gap-3">
                        {/* ZKP Status Badge */}
                        {match.zkpStatus && match.zkpStatus !== "none" && (
                            <div className="hidden xs:block">
                                <CryptoStatusBadge status={match.zkpStatus} />
                            </div>
                        )}

                        {/* Expand toggle */}
                        <button
                            onClick={() => setExpanded(!expanded)}
                            className="flex items-center justify-center w-6 h-6 rounded border border-zinc-800 hover:border-zinc-700 text-zinc-500 hover:text-zinc-300 transition-colors bg-zinc-900/30"
                        >
                            {expanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        </button>
                    </div>
                </div>

                {/* Mobile ZKP (Visible only on mobile when hidden in row) */}
                {match.zkpStatus && match.zkpStatus !== "none" && (
                    <div className="xs:hidden block pt-1">
                        <CryptoStatusBadge status={match.zkpStatus} />
                    </div>
                )}
            </div>

            {/* ── Completeness Bar ── */}
            <div className="px-4 pb-2">
                <div className="flex items-center gap-2 mb-1">
                    <span className="font-data text-[7px] uppercase tracking-wider text-zinc-600">
                        Profile Completeness
                    </span>
                    <span className={`font-data text-[7px] font-bold ${tier.color}`}>
                        {match.activeLoci}/{match.totalLoci}
                    </span>
                </div>
                <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
                    <motion.div
                        className={`h-full rounded-full ${tier.barColor}`}
                        initial={{ width: 0 }}
                        animate={{ width: `${match.completenessRatio * 100}%` }}
                        transition={{ duration: 0.6, ease: "easeOut" }}
                    />
                </div>
            </div>

            {/* ── Expanded Detail ── */}
            {expanded && (
                <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="border-t border-zinc-800/50"
                >
                    <div className="px-4 py-3 space-y-3">
                        {/* Penalty explanation */}
                        <div className="flex items-start gap-2 p-2 rounded bg-zinc-900/50 border border-zinc-800/50">
                            <Info className="w-3 h-3 text-zinc-500 mt-0.5 flex-shrink-0" />
                            <p className="font-mono text-[10px] text-zinc-500 leading-relaxed">
                                <span className="text-zinc-400 font-bold">Completeness Penalty:</span> Penalized Score
                                = Raw Similarity × (Active Loci / Total Loci) ={" "}
                                <span className="text-tactical-primary">
                                    {(match.rawSimilarity * 100).toFixed(2)}% × {match.activeLoci}/{match.totalLoci}
                                    {" "}= {(match.penalizedScore * 100).toFixed(2)}%
                                </span>
                            </p>
                        </div>

                        {/* Missing loci breakdown */}
                        {allMissing.length > 0 && (
                            <div>
                                <div className="flex items-center gap-1.5 mb-2">
                                    <AlertTriangle className="w-3 h-3 text-amber-500" />
                                    <span className="font-data text-[9px] font-bold uppercase tracking-wider text-amber-400">
                                        Missing Loci ({allMissing.length})
                                    </span>
                                </div>
                                <LociGrid
                                    missingQuery={match.missingLociQuery}
                                    missingTarget={match.missingLociTarget}
                                />
                                <div className="flex items-center gap-4 mt-2">
                                    <div className="flex items-center gap-1.5">
                                        <div className="w-2 h-2 rounded-sm bg-emerald-500/30 border border-emerald-500/50" />
                                        <span className="font-data text-[7px] text-zinc-500">Compared</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <div className="w-2 h-2 rounded-sm bg-amber-500/30 border border-amber-500/50" />
                                        <span className="font-data text-[7px] text-zinc-500">Missing (Query)</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <div className="w-2 h-2 rounded-sm bg-violet-500/30 border border-violet-500/50" />
                                        <span className="font-data text-[7px] text-zinc-500">Missing (Target)</span>
                                    </div>
                                    <div className="flex items-center gap-1.5">
                                        <div className="w-2 h-2 rounded-sm bg-red-500/30 border border-red-500/50" />
                                        <span className="font-data text-[7px] text-zinc-500">Missing (Both)</span>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* ZKP Proof Metadata */}
                        {match.zkpMetadata && match.zkpStatus === "verified" && (
                            <ProofMetadataPanel metadata={match.zkpMetadata} />
                        )}
                    </div>
                </motion.div>
            )}
        </motion.div>
    );
}


// ─── Demo Export ──────────────────────────────────────────────────────────────

const DEMO_MATCHES: MatchResultData[] = [
    {
        profileId: "SIM-7A3F1902",
        nodeId: "EUROPOL-NL",
        rawSimilarity: 0.9847,
        penalizedScore: 0.9024,
        activeLoci: 22,
        totalLoci: 24,
        completenessRatio: 0.9167,
        qualityTier: "complete",
        missingLociQuery: [],
        missingLociTarget: ["PENTA_D", "PENTA_E"],
        zkpStatus: "verified",
        zkpMetadata: {
            commitmentHash: "a3f8c91d2e7b405689012345abcdef67890abcde12345678fedcba0987654321",
            proofSizeBytes: 1089,
            verificationMs: 2.4,
            queryId: "Q-2026-0212-7A3F",
        },
    },
    {
        profileId: "SIM-B4C92D11",
        nodeId: "BKA-DE",
        rawSimilarity: 0.9312,
        penalizedScore: 0.5819,
        activeLoci: 15,
        totalLoci: 24,
        completenessRatio: 0.625,
        qualityTier: "partial",
        missingLociQuery: ["D2S1338", "D19S433", "SE33"],
        missingLociTarget: ["D1S1656", "D10S1248", "D22S1045", "PENTA_D", "PENTA_E", "FGA"],
        zkpStatus: "verifying",
    },
    {
        profileId: "SIM-E8F0A367",
        nodeId: "NCA-UK",
        rawSimilarity: 0.8921,
        penalizedScore: 0.2973,
        activeLoci: 8,
        totalLoci: 24,
        completenessRatio: 0.3333,
        qualityTier: "degraded",
        missingLociQuery: ["D2S1338", "D19S433", "SE33"],
        missingLociTarget: [
            "D1S1656", "D2S441", "D5S818", "D7S820", "D10S1248",
            "D12S391", "D16S539", "D18S51", "D22S1045", "PENTA_D",
            "PENTA_E", "TH01", "TPOX",
        ],
        zkpStatus: "invalid",
    },
];

export function MatchResultCardDemo() {
    return (
        <div className="space-y-3 p-4 bg-[#0a0a0a]">
            <div className="flex items-center gap-2 mb-2">
                <Shield className="w-4 h-4 text-tactical-primary" />
                <h2 className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text">
                    Search Results — Completeness-Aware Ranking
                </h2>
            </div>
            {DEMO_MATCHES.map((m) => (
                <MatchResultCard key={m.profileId} match={m} />
            ))}
        </div>
    );
}
