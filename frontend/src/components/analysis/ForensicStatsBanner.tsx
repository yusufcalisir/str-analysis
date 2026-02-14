"use client";

import { motion } from "framer-motion";
import {
    Scale,
    Shield,
    Fingerprint,
    AlertTriangle,
    TrendingUp,
    CheckCircle,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ForensicStatsBannerProps {
    combinedLR: number;
    log10LR: number;
    randomMatchProbability: number;
    randomMatchProbabilityStr: string;
    verbalEquivalence: string;
    prosecutionProbability: number;
    matchClassification: string;
    recommendedAction: string;
    lociAnalyzed: number;
    highFrequencyWarning: boolean;
    warningMessage: string;
    population: string;
    totalAnalysisTimeMs: number;
}

// ─── Verbal equivalence styling ──────────────────────────────────────────────

const VERBAL_STYLES: Record<string, { color: string; bg: string; border: string; icon: typeof Shield }> = {
    IDENTIFICATION_PRACTICALLY_PROVEN: {
        color: "text-emerald-300",
        bg: "bg-emerald-500/10",
        border: "border-emerald-500/25",
        icon: CheckCircle,
    },
    EXTREMELY_STRONG_SUPPORT: {
        color: "text-emerald-400",
        bg: "bg-emerald-500/8",
        border: "border-emerald-500/20",
        icon: Shield,
    },
    VERY_STRONG_SUPPORT: {
        color: "text-cyan-400",
        bg: "bg-cyan-500/8",
        border: "border-cyan-500/20",
        icon: Shield,
    },
    STRONG_SUPPORT: {
        color: "text-blue-400",
        bg: "bg-blue-500/8",
        border: "border-blue-500/20",
        icon: TrendingUp,
    },
    MODERATELY_STRONG_SUPPORT: {
        color: "text-amber-400",
        bg: "bg-amber-500/8",
        border: "border-amber-500/20",
        icon: TrendingUp,
    },
    MODERATE_SUPPORT: {
        color: "text-amber-500",
        bg: "bg-amber-500/6",
        border: "border-amber-500/15",
        icon: Scale,
    },
    LIMITED_SUPPORT: {
        color: "text-orange-400",
        bg: "bg-orange-500/6",
        border: "border-orange-500/15",
        icon: AlertTriangle,
    },
    INCONCLUSIVE: {
        color: "text-zinc-500",
        bg: "bg-zinc-800/50",
        border: "border-zinc-700/30",
        icon: AlertTriangle,
    },
};

function formatVerbalEquivalence(s: string): string {
    return s.replace(/_/g, " ");
}

function formatCLR(lr: number): string {
    if (lr >= 1e15) return `${(lr / 1e15).toFixed(1)}×10¹⁵`;
    if (lr >= 1e12) return `${(lr / 1e12).toFixed(1)}×10¹²`;
    if (lr >= 1e9) return `${(lr / 1e9).toFixed(1)}×10⁹`;
    if (lr >= 1e6) return `${(lr / 1e6).toFixed(1)}×10⁶`;
    return lr.toExponential(2);
}

// ─── Stat Cell ───────────────────────────────────────────────────────────────

function StatCell({
    label,
    value,
    subtext,
    icon: Icon,
    color = "text-tactical-primary",
}: {
    label: string;
    value: string;
    subtext?: string;
    icon: typeof Shield;
    color?: string;
}) {
    return (
        <div className="flex flex-col items-center gap-1 px-3 py-2">
            <Icon className={`w-3.5 h-3.5 ${color} opacity-60`} />
            <span className="font-data text-[7px] uppercase tracking-[0.12em] text-zinc-600">
                {label}
            </span>
            <span className={`font-data text-[13px] font-bold tabular-nums ${color}`}>
                {value}
            </span>
            {subtext && (
                <span className="font-data text-[7px] text-zinc-600 text-center max-w-[100px]">
                    {subtext}
                </span>
            )}
        </div>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function ForensicStatsBanner(props: ForensicStatsBannerProps) {
    const verbal = props.verbalEquivalence || "INCONCLUSIVE";
    const style = VERBAL_STYLES[verbal] || VERBAL_STYLES.INCONCLUSIVE;
    const VerbalIcon = style.icon;

    return (
        <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="border border-zinc-800/60 rounded bg-tactical-bg overflow-hidden"
        >
            {/* ── Header Row ── */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-800/40">
                <div className="flex items-center gap-2">
                    <Scale className="w-4 h-4 text-tactical-primary" />
                    <h3 className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text">
                        Dynamic Forensic Analysis
                    </h3>
                </div>
                <div className="flex items-center gap-2">
                    <span className="font-data text-[8px] text-zinc-600">
                        {props.lociAnalyzed} loci · {props.population}
                    </span>
                    <span className="font-data text-[8px] text-zinc-700">
                        {props.totalAnalysisTimeMs.toFixed(1)}ms
                    </span>
                </div>
            </div>

            {/* ── Verbal Equivalence Badge ── */}
            <div className="flex items-center justify-center py-3 px-4 border-b border-zinc-800/30">
                <motion.div
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.15 }}
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-full border ${style.bg} ${style.border}`}
                >
                    <VerbalIcon className={`w-3.5 h-3.5 ${style.color}`} />
                    <span className={`font-data text-[9px] font-bold uppercase tracking-[0.15em] ${style.color}`}>
                        {formatVerbalEquivalence(verbal)}
                    </span>
                </motion.div>
            </div>

            {/* ── Stats Grid ── */}
            <div className="grid grid-cols-2 sm:grid-cols-4 divide-x divide-zinc-800/30">
                <StatCell
                    label="Combined LR"
                    value={formatCLR(props.combinedLR)}
                    subtext={`log₁₀ = ${props.log10LR.toFixed(1)}`}
                    icon={Scale}
                    color="text-tactical-primary"
                />
                <StatCell
                    label="RMP"
                    value={props.randomMatchProbabilityStr}
                    icon={Fingerprint}
                    color="text-cyan-400"
                />
                <StatCell
                    label="P(Hp|E)"
                    value={`${(props.prosecutionProbability * 100).toFixed(4)}%`}
                    subtext="Prosecution posterior"
                    icon={TrendingUp}
                    color="text-emerald-400"
                />
                <StatCell
                    label="Classification"
                    value={props.matchClassification.replace(/_/g, " ")}
                    subtext={props.recommendedAction.replace(/_/g, " ")}
                    icon={Shield}
                    color={
                        props.matchClassification === "DIRECT_IDENTITY"
                            ? "text-emerald-400"
                            : props.matchClassification === "FAMILIAL_LINK"
                                ? "text-amber-400"
                                : "text-zinc-500"
                    }
                />
            </div>

            {/* ── Warning bar ── */}
            {props.highFrequencyWarning && (
                <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    className="flex items-center gap-2 px-4 py-2 bg-amber-500/5 border-t border-amber-500/15"
                >
                    <AlertTriangle className="w-3 h-3 text-amber-500 flex-shrink-0" />
                    <span className="font-data text-[8px] text-amber-400/80">
                        {props.warningMessage}
                    </span>
                </motion.div>
            )}
        </motion.div>
    );
}
