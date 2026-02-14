"use client";

import React, { useMemo } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Shield, Zap } from "lucide-react";

interface SensitivityEntry {
    marker: string;
    log10_lr: number;
    contribution_pct: number;
    has_dropout: boolean;
    has_stutter: boolean;
}

interface LociSensitivityMapProps {
    sensitivityMap: SensitivityEntry[];
    degradationIndex?: number;
}

export default function LociSensitivityMap({
    sensitivityMap,
    degradationIndex = 0,
}: LociSensitivityMapProps) {
    const sorted = useMemo(
        () => [...sensitivityMap].sort((a, b) => b.contribution_pct - a.contribution_pct),
        [sensitivityMap]
    );

    const maxPct = useMemo(
        () => Math.max(...sorted.map((s) => Math.abs(s.contribution_pct)), 1),
        [sorted]
    );

    const getBarColor = (entry: SensitivityEntry) => {
        if (entry.has_dropout || entry.has_stutter) return "rgba(239, 68, 68, 0.7)";
        if (entry.contribution_pct > 8) return "rgba(34, 197, 94, 0.8)";
        if (entry.contribution_pct > 4) return "rgba(59, 130, 246, 0.7)";
        return "rgba(161, 161, 170, 0.5)";
    };

    const getGlowColor = (entry: SensitivityEntry) => {
        if (entry.has_dropout || entry.has_stutter) return "0 0 12px rgba(239, 68, 68, 0.3)";
        if (entry.contribution_pct > 8) return "0 0 12px rgba(34, 197, 94, 0.2)";
        return "none";
    };

    return (
        <div
            style={{
                background: "rgba(0, 0, 0, 0.3)",
                border: "1px solid rgba(59, 130, 246, 0.15)",
                borderRadius: 12,
                padding: "20px 24px",
                backdropFilter: "blur(12px)",
            }}
        >
            {/* Header */}
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                <Zap size={16} style={{ color: "rgba(59, 130, 246, 0.8)" }} />
                <span
                    style={{
                        fontSize: 11,
                        fontWeight: 600,
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        color: "rgba(161, 161, 170, 0.9)",
                    }}
                >
                    Loci Sensitivity Map
                </span>
                {degradationIndex > 0.3 && (
                    <span
                        style={{
                            fontSize: 10,
                            fontWeight: 500,
                            color: "rgba(239, 68, 68, 0.9)",
                            background: "rgba(239, 68, 68, 0.1)",
                            padding: "2px 8px",
                            borderRadius: 4,
                            marginLeft: "auto",
                        }}
                    >
                        ⚠ Degradation: {(degradationIndex * 100).toFixed(1)}%
                    </span>
                )}
            </div>

            {/* Legend */}
            <div
                style={{
                    display: "flex",
                    gap: 16,
                    marginBottom: 16,
                    fontSize: 10,
                    color: "rgba(161, 161, 170, 0.7)",
                }}
            >
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(34, 197, 94, 0.8)" }} />
                    High LR
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(59, 130, 246, 0.7)" }} />
                    Moderate
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(161, 161, 170, 0.5)" }} />
                    Low
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    <span style={{ width: 8, height: 8, borderRadius: 2, background: "rgba(239, 68, 68, 0.7)" }} />
                    Error Flagged
                </span>
            </div>

            {/* Bars */}
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                {sorted.map((entry, i) => (
                    <motion.div
                        key={entry.marker}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.3 }}
                        style={{ display: "flex", alignItems: "center", gap: 8 }}
                    >
                        {/* Marker label */}
                        <span
                            style={{
                                width: 85,
                                fontSize: 10,
                                fontFamily: "monospace",
                                color: entry.has_dropout || entry.has_stutter
                                    ? "rgba(239, 68, 68, 0.9)"
                                    : "rgba(161, 161, 170, 0.9)",
                                textAlign: "right",
                                flexShrink: 0,
                            }}
                        >
                            {entry.marker}
                        </span>

                        {/* Bar */}
                        <div
                            style={{
                                flex: 1,
                                height: 18,
                                background: "rgba(255, 255, 255, 0.03)",
                                borderRadius: 3,
                                overflow: "hidden",
                                position: "relative",
                            }}
                        >
                            <motion.div
                                initial={{ width: 0 }}
                                animate={{
                                    width: `${Math.max(2, (Math.abs(entry.contribution_pct) / maxPct) * 100)}%`,
                                }}
                                transition={{ delay: i * 0.04 + 0.2, duration: 0.5, ease: "easeOut" }}
                                style={{
                                    height: "100%",
                                    background: getBarColor(entry),
                                    borderRadius: 3,
                                    boxShadow: getGlowColor(entry),
                                    position: "relative",
                                }}
                            />
                        </div>

                        {/* Value */}
                        <span
                            style={{
                                width: 50,
                                fontSize: 10,
                                fontFamily: "monospace",
                                color: "rgba(161, 161, 170, 0.7)",
                                textAlign: "right",
                                flexShrink: 0,
                            }}
                        >
                            {entry.contribution_pct.toFixed(1)}%
                        </span>

                        {/* Warning badges */}
                        <div style={{ width: 32, display: "flex", gap: 2, flexShrink: 0 }}>
                            {entry.has_dropout && (
                                <AlertTriangle size={12} style={{ color: "rgba(239, 68, 68, 0.8)" }} />
                            )}
                            {entry.has_stutter && (
                                <Shield size={12} style={{ color: "rgba(251, 191, 36, 0.8)" }} />
                            )}
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* Summary footer */}
            <div
                style={{
                    marginTop: 16,
                    paddingTop: 12,
                    borderTop: "1px solid rgba(255, 255, 255, 0.06)",
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 10,
                    color: "rgba(161, 161, 170, 0.6)",
                }}
            >
                <span>{sensitivityMap.length} loci analyzed</span>
                <span>
                    {sensitivityMap.filter((s) => s.has_dropout).length} dropout ·{" "}
                    {sensitivityMap.filter((s) => s.has_stutter).length} stutter
                </span>
            </div>
        </div>
    );
}
