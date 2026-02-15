"use client";

import { motion } from "framer-motion";
import {
    Globe,
    ShieldCheck,
    BarChart3,
    ExternalLink,
    TrendingUp,
} from "lucide-react";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface GeoProbability {
    region: string;
    lat: number;
    lng: number;
    probability: number;
    color: string;
}

interface AncestryDataPanelProps {
    data: GeoProbability[];
    reliabilityScore: number;
    txHash?: string;
    selectedRegion?: string | null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// PROBABILITY BAR
// ═══════════════════════════════════════════════════════════════════════════════

function ProbabilityBar({
    region,
    probability,
    color,
    rank,
    isSelected,
}: {
    region: string;
    probability: number;
    color: string;
    rank: number;
    isSelected?: boolean;
}) {
    const pct = (probability * 100).toFixed(1);
    return (
        <motion.div
            initial={{ opacity: 0, x: -8 }}
            animate={{ opacity: 1, x: 0, scale: isSelected ? 1.02 : 1 }}
            transition={{ delay: 0.1 + rank * 0.06 }}
            className={`group p-1.5 rounded transition-all ${isSelected ? 'bg-tactical-primary/10 border border-tactical-primary/30' : 'border border-transparent'}`}
        >
            <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                    <span
                        className="w-2 h-2 rounded-full shrink-0"
                        style={{ backgroundColor: color }}
                    />
                    <span className="font-mono text-[9px] text-zinc-400 tracking-wide truncate max-w-[120px]">
                        {region}
                    </span>
                </div>
                <span className="font-mono text-[9px] font-bold text-tactical-text tabular-nums">
                    {pct}%
                </span>
            </div>
            <div className="h-1 rounded-full bg-zinc-800/80 overflow-hidden">
                <motion.div
                    initial={{ width: 0 }}
                    animate={{ width: `${probability * 100}%` }}
                    transition={{ duration: 0.8, delay: 0.2 + rank * 0.06, ease: "easeOut" }}
                    className="h-full rounded-full"
                    style={{ backgroundColor: color }}
                />
            </div>
        </motion.div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// RELIABILITY GAUGE
// ═══════════════════════════════════════════════════════════════════════════════

function ReliabilityGauge({ score }: { score: number }) {
    const pct = Math.round(score * 100);
    const circumference = 2 * Math.PI * 20;
    const dashOffset = circumference * (1 - score);

    const color =
        score >= 0.8
            ? "#22C55E"
            : score >= 0.5
                ? "#F59E0B"
                : "#EF4444";

    return (
        <div className="flex items-center gap-3">
            <div className="relative w-12 h-12">
                <svg viewBox="0 0 48 48" className="w-full h-full -rotate-90">
                    <circle
                        cx="24"
                        cy="24"
                        r="20"
                        fill="none"
                        stroke="#27272A"
                        strokeWidth="3"
                    />
                    <motion.circle
                        cx="24"
                        cy="24"
                        r="20"
                        fill="none"
                        stroke={color}
                        strokeWidth="3"
                        strokeLinecap="round"
                        strokeDasharray={circumference}
                        initial={{ strokeDashoffset: circumference }}
                        animate={{ strokeDashoffset: dashOffset }}
                        transition={{ duration: 1, delay: 0.4, ease: "easeOut" }}
                    />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                    <span className="font-mono text-[10px] font-bold" style={{ color }}>
                        {pct}%
                    </span>
                </div>
            </div>
            <div>
                <div className="font-mono text-[9px] font-bold text-tactical-text tracking-[0.12em] uppercase leading-none">
                    Reliability Score
                </div>
                <div className="font-mono text-[8px] text-zinc-500 mt-0.5">
                    {score >= 0.8 ? "HIGH" : score >= 0.5 ? "MODERATE" : "LOW"} CONFIDENCE
                </div>
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN PANEL
// ═══════════════════════════════════════════════════════════════════════════════

export default function AncestryDataPanel({
    data,
    reliabilityScore,
    txHash,
    selectedRegion,
}: AncestryDataPanelProps) {
    const truncatedHash = txHash
        ? `${txHash.slice(0, 8)}...${txHash.slice(-6)}`
        : null;

    return (
        <div className="h-full flex flex-col gap-2 px-4 py-3 overflow-hidden">
            {/* Section 1: Ancestry Breakdown */}
            <div>
                <div className="flex items-center gap-1.5 mb-2">
                    <BarChart3 className="w-3.5 h-3.5 text-tactical-primary" />
                    <span className="font-mono text-[9px] font-bold text-tactical-text tracking-[0.15em] uppercase">
                        Ancestry_Breakdown
                    </span>
                </div>
                <div className="space-y-1">
                    {data.map((region, idx) => (
                        <ProbabilityBar
                            key={region.region}
                            region={region.region}
                            probability={region.probability}
                            color={region.color}
                            rank={idx}
                            isSelected={selectedRegion === region.region}
                        />
                    ))}
                </div>
            </div>

            {/* Divider */}
            <div className="border-t border-tactical-border/50" />

            {/* Section 2 + 3: Reliability & Chain of Custody */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <ReliabilityGauge score={reliabilityScore} />

                {/* Chain of Custody Badge */}
                {truncatedHash ? (
                    <motion.div
                        initial={{ opacity: 0, scale: 0.95 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.6 }}
                        className="flex items-center gap-2 px-3 py-2 rounded bg-tactical-primary/5 border border-tactical-primary/20 hover:border-tactical-primary/40 transition-colors cursor-pointer group"
                    >
                        <ShieldCheck className="w-3.5 h-3.5 text-tactical-primary" />
                        <div>
                            <div className="font-mono text-[8px] text-zinc-500 tracking-wider uppercase">
                                On-Chain Verification
                            </div>
                            <div className="font-mono text-[10px] text-tactical-primary font-bold flex items-center gap-1">
                                {truncatedHash}
                                <ExternalLink className="w-2.5 h-2.5 opacity-50 group-hover:opacity-100 transition-opacity" />
                            </div>
                        </div>
                    </motion.div>
                ) : (
                    <div className="flex items-center gap-2 px-3 py-2 rounded bg-zinc-800/30 border border-zinc-700/30">
                        <ShieldCheck className="w-3.5 h-3.5 text-zinc-600" />
                        <div className="font-mono text-[8px] text-zinc-600 tracking-wider uppercase">
                            Awaiting Ledger Sync
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
