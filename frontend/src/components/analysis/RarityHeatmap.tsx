"use client";

import { motion } from "framer-motion";
import { Dna, Activity, AlertTriangle, Info } from "lucide-react";
import { useState, useMemo } from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface LocusDetail {
    marker: string;
    alleles: number[];
    is_homozygous: boolean;
    frequencies: number[];
    genotype_probability: number;
    individual_lr: number;
    log10_lr: number;
    rarity_score: number;
}

interface RarityHeatmapProps {
    perLocusDetails: LocusDetail[];
    population: string;
    onPopulationChange?: (population: string) => void;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const POPULATIONS = ["European", "African", "East_Asian"] as const;

const POPULATION_LABELS: Record<string, string> = {
    "European": "EUR",
    "African": "AFR",
    "East_Asian": "EAS",
};

// ─── Rarity color interpolation ──────────────────────────────────────────────

function rarityToColor(score: number): { bg: string; border: string; text: string; glow: string } {
    if (score >= 0.8) {
        // Extremely rare → hot red/orange
        return {
            bg: `rgba(239, 68, 68, ${0.15 + score * 0.25})`,
            border: `rgba(239, 68, 68, ${0.3 + score * 0.4})`,
            text: "text-red-400",
            glow: `0 0 ${8 + score * 12}px rgba(239, 68, 68, ${score * 0.5})`,
        };
    }
    if (score >= 0.5) {
        // Rare → amber/orange
        const t = (score - 0.5) / 0.3;
        return {
            bg: `rgba(245, 158, 11, ${0.1 + t * 0.15})`,
            border: `rgba(245, 158, 11, ${0.2 + t * 0.3})`,
            text: "text-amber-400",
            glow: `0 0 ${4 + t * 8}px rgba(245, 158, 11, ${t * 0.3})`,
        };
    }
    if (score >= 0.2) {
        // Moderate → cyan
        return {
            bg: "rgba(6, 182, 212, 0.08)",
            border: "rgba(6, 182, 212, 0.15)",
            text: "text-cyan-500",
            glow: "none",
        };
    }
    // Common → dim zinc
    return {
        bg: "rgba(161, 161, 170, 0.04)",
        border: "rgba(161, 161, 170, 0.08)",
        text: "text-zinc-600",
        glow: "none",
    };
}

function formatLR(lr: number): string {
    if (lr >= 1e9) return lr.toExponential(2);
    if (lr >= 1e6) return `${(lr / 1e6).toFixed(1)}M`;
    if (lr >= 1e3) return `${(lr / 1e3).toFixed(1)}K`;
    return lr.toFixed(1);
}

// ─── Locus Cell ──────────────────────────────────────────────────────────────

function LocusCell({ detail, index }: { detail: LocusDetail; index: number }) {
    const [hovered, setHovered] = useState(false);
    const colors = rarityToColor(detail.rarity_score);

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: index * 0.03, duration: 0.3 }}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
            className="relative group cursor-default"
        >
            <motion.div
                animate={{
                    scale: hovered ? 1.08 : 1,
                }}
                transition={{ type: "spring", stiffness: 400, damping: 25 }}
                className="flex flex-col items-center justify-center rounded border py-2 px-1"
                style={{
                    backgroundColor: colors.bg,
                    borderColor: colors.border,
                    boxShadow: colors.glow,
                    minHeight: 56,
                }}
            >
                <span className={`font-data text-[8px] font-bold tracking-wider ${colors.text}`}>
                    {detail.marker}
                </span>
                <span className={`font-data text-[10px] font-bold tabular-nums mt-0.5 ${colors.text}`}>
                    {detail.alleles[0]}/{detail.alleles[1]}
                </span>
                {detail.is_homozygous && (
                    <span className="font-data text-[6px] text-amber-500/60 mt-0.5">HOM</span>
                )}
            </motion.div>

            {/* ── Hover tooltip ── */}
            {hovered && (
                <motion.div
                    initial={{ opacity: 0, y: 4, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-48 p-2.5 rounded bg-zinc-900/95 border border-zinc-700/50 backdrop-blur-sm shadow-xl"
                >
                    <div className="space-y-1.5">
                        <div className="flex items-center justify-between">
                            <span className="font-data text-[9px] font-bold text-zinc-300">
                                {detail.marker}
                            </span>
                            <span className={`font-data text-[8px] font-bold ${colors.text}`}>
                                RARITY: {(detail.rarity_score * 100).toFixed(0)}%
                            </span>
                        </div>
                        <div className="h-px bg-zinc-800" />
                        <div className="grid grid-cols-2 gap-1 text-[8px] font-data">
                            <div>
                                <span className="text-zinc-600 block">Alleles</span>
                                <span className="text-zinc-300">{detail.alleles[0]} / {detail.alleles[1]}</span>
                            </div>
                            <div>
                                <span className="text-zinc-600 block">Frequencies</span>
                                <span className="text-zinc-300">
                                    {detail.frequencies[0].toFixed(4)} / {detail.frequencies[1].toFixed(4)}
                                </span>
                            </div>
                            <div>
                                <span className="text-zinc-600 block">Geno. Prob.</span>
                                <span className="text-cyan-400">{detail.genotype_probability.toExponential(2)}</span>
                            </div>
                            <div>
                                <span className="text-zinc-600 block">Locus LR</span>
                                <span className="text-tactical-primary font-bold">{formatLR(detail.individual_lr)}</span>
                            </div>
                        </div>
                    </div>
                    {/* Tooltip arrow */}
                    <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-2 h-2 rotate-45 bg-zinc-900/95 border-r border-b border-zinc-700/50" />
                </motion.div>
            )}
        </motion.div>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function RarityHeatmap({ perLocusDetails, population, onPopulationChange }: RarityHeatmapProps) {
    const sortedDetails = useMemo(
        () => [...perLocusDetails].sort((a, b) => b.rarity_score - a.rarity_score),
        [perLocusDetails]
    );

    const avgRarity = useMemo(() => {
        if (!perLocusDetails.length) return 0;
        return perLocusDetails.reduce((sum, d) => sum + d.rarity_score, 0) / perLocusDetails.length;
    }, [perLocusDetails]);

    const rareCount = useMemo(
        () => perLocusDetails.filter((d) => d.rarity_score >= 0.5).length,
        [perLocusDetails]
    );

    if (!perLocusDetails.length) return null;

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="border border-zinc-800/60 rounded bg-tactical-bg overflow-hidden"
        >
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/40">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 text-tactical-primary" />
                    <h3 className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text">
                        Allele Rarity Heatmap
                    </h3>
                    <span className="font-data text-[8px] text-zinc-600">
                        {perLocusDetails.length} loci
                    </span>
                </div>

                {/* Population selector */}
                {onPopulationChange && (
                    <div className="flex items-center gap-1">
                        {POPULATIONS.map((pop) => (
                            <button
                                key={pop}
                                onClick={() => onPopulationChange(pop)}
                                className={`px-2 py-0.5 rounded font-data text-[8px] font-bold tracking-wider uppercase transition-all ${population === pop
                                        ? "bg-tactical-primary/20 text-tactical-primary border border-tactical-primary/30"
                                        : "bg-zinc-900 text-zinc-600 border border-zinc-800/50 hover:text-zinc-400 hover:border-zinc-700"
                                    }`}
                            >
                                {POPULATION_LABELS[pop]}
                            </button>
                        ))}
                    </div>
                )}
            </div>

            {/* ── Summary bar ── */}
            <div className="flex items-center gap-4 px-4 py-2 bg-zinc-900/30 border-b border-zinc-800/30">
                <div className="flex items-center gap-1.5">
                    <Dna className="w-3 h-3 text-zinc-600" />
                    <span className="font-data text-[8px] text-zinc-500">Avg. Rarity:</span>
                    <span className={`font-data text-[9px] font-bold ${avgRarity >= 0.5 ? "text-amber-400" : avgRarity >= 0.2 ? "text-cyan-400" : "text-zinc-500"
                        }`}>
                        {(avgRarity * 100).toFixed(1)}%
                    </span>
                </div>
                {rareCount > 0 && (
                    <div className="flex items-center gap-1.5">
                        <AlertTriangle className="w-3 h-3 text-red-500/60" />
                        <span className="font-data text-[8px] text-red-400/80">
                            {rareCount} rare marker{rareCount > 1 ? "s" : ""}
                        </span>
                    </div>
                )}
            </div>

            {/* ── Heatmap Grid ── */}
            <div className="p-3">
                <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-1.5">
                    {sortedDetails.map((detail, i) => (
                        <LocusCell key={detail.marker} detail={detail} index={i} />
                    ))}
                </div>
            </div>

            {/* ── Legend ── */}
            <div className="flex items-center justify-center gap-4 px-4 py-2 border-t border-zinc-800/30">
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-2 rounded-sm" style={{ backgroundColor: "rgba(161, 161, 170, 0.15)" }} />
                    <span className="font-data text-[7px] text-zinc-600">Common</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-2 rounded-sm" style={{ backgroundColor: "rgba(6, 182, 212, 0.2)" }} />
                    <span className="font-data text-[7px] text-zinc-600">Moderate</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div className="w-3 h-2 rounded-sm" style={{ backgroundColor: "rgba(245, 158, 11, 0.3)" }} />
                    <span className="font-data text-[7px] text-zinc-600">Rare</span>
                </div>
                <div className="flex items-center gap-1.5">
                    <div
                        className="w-3 h-2 rounded-sm"
                        style={{
                            backgroundColor: "rgba(239, 68, 68, 0.35)",
                            boxShadow: "0 0 6px rgba(239, 68, 68, 0.4)",
                        }}
                    />
                    <span className="font-data text-[7px] text-zinc-600">Extremely Rare</span>
                </div>
            </div>
        </motion.div>
    );
}
