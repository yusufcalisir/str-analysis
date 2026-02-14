"use client";

import { motion } from "framer-motion";
import { GitBranch, Users, AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";
import { useMemo } from "react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface IBDSummary {
    ibs0_proportion: number;
    ibs1_proportion: number;
    ibs2_proportion: number;
    ibs0_count: number;
    ibs1_count: number;
    ibs2_count: number;
}

interface KinshipData {
    relationship_type: string;
    confidence: number;
    kinship_index_parent_child: number;
    kinship_index_full_sibling: number;
    kinship_index_half_sibling: number;
    log10_ki_parent_child: number;
    log10_ki_full_sibling: number;
    log10_ki_half_sibling: number;
    exclusion_count: number;
    loci_analyzed: number;
    ibd_summary: IBDSummary;
    population_used: string;
    reasoning: string;
}

interface PedigreeTreeProps {
    kinshipData: KinshipData;
    profileAId: string;
    profileBId: string;
}

// ─── Styling ─────────────────────────────────────────────────────────────────

const REL_STYLES: Record<string, { color: string; label: string; icon: typeof Users }> = {
    PARENT_CHILD: { color: "#34d399", label: "Parent–Child", icon: Users },
    FULL_SIBLING: { color: "#fbbf24", label: "Full Sibling", icon: Users },
    HALF_SIBLING: { color: "#22d3ee", label: "Half Sibling", icon: Users },
    SELF: { color: "#a78bfa", label: "Self / Identical", icon: CheckCircle },
    UNRELATED: { color: "#71717a", label: "Unrelated", icon: HelpCircle },
    INCONCLUSIVE: { color: "#71717a", label: "Inconclusive", icon: AlertTriangle },
};

function formatKI(ki: number): string {
    if (ki === 0) return "EXCLUDED";
    if (ki >= 1e9) return ki.toExponential(2);
    if (ki >= 1e6) return `${(ki / 1e6).toFixed(1)}M`;
    if (ki >= 1e3) return `${(ki / 1e3).toFixed(1)}K`;
    return ki.toFixed(2);
}

// ─── Profile Node ────────────────────────────────────────────────────────────

function ProfileNode({
    label,
    id,
    x,
    y,
    color,
    isQuery,
}: {
    label: string;
    id: string;
    x: number;
    y: number;
    color: string;
    isQuery: boolean;
}) {
    return (
        <g>
            {/* Glow */}
            <circle cx={x} cy={y} r={32} fill={color} opacity={0.08} />
            <circle cx={x} cy={y} r={24} fill={color} opacity={0.12} />

            {/* Node */}
            <circle
                cx={x}
                cy={y}
                r={20}
                fill="rgba(24, 24, 27, 0.9)"
                stroke={color}
                strokeWidth={1.5}
                strokeDasharray={isQuery ? "none" : "4 2"}
            />

            {/* Icon (DNA double helix simplified) */}
            <text
                x={x}
                y={y + 1}
                textAnchor="middle"
                dominantBaseline="central"
                fill={color}
                fontSize={14}
                fontFamily="monospace"
            >
                ⬡
            </text>

            {/* Label */}
            <text
                x={x}
                y={y - 30}
                textAnchor="middle"
                fill="rgba(161, 161, 170, 0.8)"
                fontSize={7}
                fontFamily="monospace"
                letterSpacing="0.1em"
                style={{ textTransform: "uppercase" }}
            >
                {label}
            </text>
            <text
                x={x}
                y={y + 36}
                textAnchor="middle"
                fill={color}
                fontSize={8}
                fontFamily="monospace"
                fontWeight="bold"
                letterSpacing="0.05em"
            >
                {id.length > 16 ? id.slice(0, 14) + "…" : id}
            </text>
        </g>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function PedigreeTree({ kinshipData, profileAId, profileBId }: PedigreeTreeProps) {
    const style = REL_STYLES[kinshipData.relationship_type] || REL_STYLES.INCONCLUSIVE;
    const RelIcon = style.icon;

    const bestKI = useMemo(() => {
        switch (kinshipData.relationship_type) {
            case "PARENT_CHILD":
                return kinshipData.kinship_index_parent_child;
            case "FULL_SIBLING":
                return kinshipData.kinship_index_full_sibling;
            case "HALF_SIBLING":
                return kinshipData.kinship_index_half_sibling;
            default:
                return Math.max(
                    kinshipData.kinship_index_parent_child,
                    kinshipData.kinship_index_full_sibling,
                    kinshipData.kinship_index_half_sibling
                );
        }
    }, [kinshipData]);

    const svgWidth = 340;
    const svgHeight = 220;
    const nodeAx = svgWidth / 2;
    const nodeAy = 50;
    const nodeBx = svgWidth / 2;
    const nodeBy = 170;

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="border border-zinc-800/60 rounded bg-tactical-bg overflow-hidden"
        >
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-zinc-800/40">
                <div className="flex items-center gap-2">
                    <GitBranch className="w-4 h-4 text-tactical-primary" />
                    <h3 className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text">
                        Pedigree Analysis
                    </h3>
                </div>
                <div className="flex items-center gap-1.5">
                    <span
                        className="font-data text-[8px] font-bold px-2 py-0.5 rounded-full border uppercase tracking-wider"
                        style={{
                            color: style.color,
                            borderColor: `${style.color}40`,
                            backgroundColor: `${style.color}10`,
                        }}
                    >
                        {style.label}
                    </span>
                </div>
            </div>

            {/* SVG Pedigree Diagram */}
            <div className="flex items-center justify-center py-3 px-4">
                <svg width={svgWidth} height={svgHeight} viewBox={`0 0 ${svgWidth} ${svgHeight}`}>
                    {/* Connecting line — dotted */}
                    <motion.line
                        initial={{ pathLength: 0, opacity: 0 }}
                        animate={{ pathLength: 1, opacity: 1 }}
                        transition={{ delay: 0.3, duration: 0.6 }}
                        x1={nodeAx}
                        y1={nodeAy + 22}
                        x2={nodeBx}
                        y2={nodeBy - 22}
                        stroke={style.color}
                        strokeWidth={1.5}
                        strokeDasharray="6 4"
                        opacity={0.6}
                    />

                    {/* Midpoint label — relationship + KI */}
                    <rect
                        x={svgWidth / 2 - 60}
                        y={svgHeight / 2 - 16}
                        width={120}
                        height={32}
                        rx={4}
                        fill="rgba(24, 24, 27, 0.95)"
                        stroke={`${style.color}30`}
                        strokeWidth={1}
                    />
                    <text
                        x={svgWidth / 2}
                        y={svgHeight / 2 - 4}
                        textAnchor="middle"
                        fill={style.color}
                        fontSize={8}
                        fontFamily="monospace"
                        fontWeight="bold"
                        letterSpacing="0.08em"
                    >
                        KI: {formatKI(bestKI)}
                    </text>
                    <text
                        x={svgWidth / 2}
                        y={svgHeight / 2 + 8}
                        textAnchor="middle"
                        fill="rgba(161, 161, 170, 0.6)"
                        fontSize={6}
                        fontFamily="monospace"
                        letterSpacing="0.08em"
                    >
                        {(kinshipData.confidence * 100).toFixed(1)}% CONFIDENCE
                    </text>

                    {/* Profile Nodes */}
                    <ProfileNode
                        label="Query Profile"
                        id={profileAId}
                        x={nodeAx}
                        y={nodeAy}
                        color={style.color}
                        isQuery={true}
                    />
                    <ProfileNode
                        label="Database Hit"
                        id={profileBId}
                        x={nodeBx}
                        y={nodeBy}
                        color={style.color}
                        isQuery={false}
                    />
                </svg>
            </div>

            {/* IBD Summary Bar */}
            <div className="px-4 py-2 border-t border-zinc-800/30">
                <div className="flex items-center gap-2 mb-1.5">
                    <span className="font-data text-[7px] text-zinc-600 uppercase tracking-wider">
                        IBS Distribution
                    </span>
                    <span className="font-data text-[7px] text-zinc-700">
                        {kinshipData.loci_analyzed} loci
                    </span>
                </div>
                <div className="flex h-2 rounded-full overflow-hidden bg-zinc-900">
                    {/* IBS2 */}
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${kinshipData.ibd_summary.ibs2_proportion * 100}%` }}
                        transition={{ delay: 0.4, duration: 0.5 }}
                        className="h-full"
                        style={{ backgroundColor: "#34d399" }}
                        title={`IBS2: ${kinshipData.ibd_summary.ibs2_count}`}
                    />
                    {/* IBS1 */}
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${kinshipData.ibd_summary.ibs1_proportion * 100}%` }}
                        transition={{ delay: 0.5, duration: 0.5 }}
                        className="h-full"
                        style={{ backgroundColor: "#fbbf24" }}
                        title={`IBS1: ${kinshipData.ibd_summary.ibs1_count}`}
                    />
                    {/* IBS0 */}
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${kinshipData.ibd_summary.ibs0_proportion * 100}%` }}
                        transition={{ delay: 0.6, duration: 0.5 }}
                        className="h-full"
                        style={{ backgroundColor: "#ef4444" }}
                        title={`IBS0: ${kinshipData.ibd_summary.ibs0_count}`}
                    />
                </div>
                <div className="flex items-center justify-between mt-1">
                    <div className="flex items-center gap-3">
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-1.5 rounded-sm" style={{ backgroundColor: "#34d399" }} />
                            <span className="font-data text-[6px] text-zinc-600">
                                IBS2: {kinshipData.ibd_summary.ibs2_count}
                            </span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-1.5 rounded-sm" style={{ backgroundColor: "#fbbf24" }} />
                            <span className="font-data text-[6px] text-zinc-600">
                                IBS1: {kinshipData.ibd_summary.ibs1_count}
                            </span>
                        </div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-1.5 rounded-sm" style={{ backgroundColor: "#ef4444" }} />
                            <span className="font-data text-[6px] text-zinc-600">
                                IBS0: {kinshipData.ibd_summary.ibs0_count}
                            </span>
                        </div>
                    </div>
                    {kinshipData.exclusion_count > 0 && (
                        <span className="font-data text-[6px] text-amber-400/80">
                            {kinshipData.exclusion_count} exclusion{kinshipData.exclusion_count > 1 ? "s" : ""}
                        </span>
                    )}
                </div>
            </div>

            {/* KI Comparison Grid */}
            <div className="grid grid-cols-3 divide-x divide-zinc-800/30 border-t border-zinc-800/30">
                {[
                    { label: "Parent-Child", ki: kinshipData.kinship_index_parent_child, log: kinshipData.log10_ki_parent_child, type: "PARENT_CHILD" },
                    { label: "Full Sibling", ki: kinshipData.kinship_index_full_sibling, log: kinshipData.log10_ki_full_sibling, type: "FULL_SIBLING" },
                    { label: "Half Sibling", ki: kinshipData.kinship_index_half_sibling, log: kinshipData.log10_ki_half_sibling, type: "HALF_SIBLING" },
                ].map((item) => {
                    const isActive = kinshipData.relationship_type === item.type;
                    return (
                        <div
                            key={item.type}
                            className={`flex flex-col items-center py-2 ${isActive ? "bg-white/[0.02]" : ""}`}
                        >
                            <span className="font-data text-[6px] text-zinc-600 uppercase tracking-wider">
                                {item.label}
                            </span>
                            <span
                                className={`font-data text-[11px] font-bold tabular-nums mt-0.5 ${isActive ? "text-tactical-primary" : "text-zinc-600"
                                    }`}
                            >
                                {formatKI(item.ki)}
                            </span>
                            <span className="font-data text-[6px] text-zinc-700 mt-0.5">
                                log₁₀ = {item.log.toFixed(1)}
                            </span>
                        </div>
                    );
                })}
            </div>

            {/* Reasoning */}
            {kinshipData.reasoning && (
                <div className="px-4 py-2 border-t border-zinc-800/30">
                    <p className="font-data text-[8px] text-zinc-500 leading-relaxed">
                        {kinshipData.reasoning}
                    </p>
                </div>
            )}
        </motion.div>
    );
}
