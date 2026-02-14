"use client";

import React, { useMemo } from "react";
import { motion } from "framer-motion";
import { TrendingUp } from "lucide-react";

interface PerLocusDetail {
    marker: string;
    adjusted_lr?: number;
    individual_lr: number;
    dropout_probability?: number;
    stutter_probability?: number;
}

interface BayesianShiftChartProps {
    perLocusDetails: PerLocusDetail[];
    priorHp: number;
    posteriorHp: number;
    ciLower?: number;
    ciUpper?: number;
}

export default function BayesianShiftChart({
    perLocusDetails,
    priorHp,
    posteriorHp,
    ciLower = 0,
    ciUpper = 1,
}: BayesianShiftChartProps) {
    // Compute cumulative posterior as each locus is validated
    const steps = useMemo(() => {
        const result: { marker: string; posterior: number; lr: number }[] = [];
        let cumulativeLR = 1.0;
        const poolSize = priorHp > 0 ? 1.0 / priorHp : 1_000_000;

        for (const detail of perLocusDetails) {
            const lr = detail.adjusted_lr ?? detail.individual_lr;
            cumulativeLR *= lr;
            const p = (cumulativeLR * priorHp) / (cumulativeLR * priorHp + (1 - priorHp));
            result.push({
                marker: detail.marker,
                posterior: Math.min(1, Math.max(0, p)),
                lr,
            });
        }
        return result;
    }, [perLocusDetails, priorHp]);

    // Chart dimensions
    const W = 700;
    const H = 280;
    const PAD_L = 50;
    const PAD_R = 20;
    const PAD_T = 30;
    const PAD_B = 60;
    const chartW = W - PAD_L - PAD_R;
    const chartH = H - PAD_T - PAD_B;

    const n = steps.length;
    const xStep = n > 1 ? chartW / (n - 1) : chartW;

    // Map posterior to log scale for better visualization of extreme values
    const toY = (p: number) => {
        // Use ln(-ln(1-p)) transform for extreme probabilities near 1
        const clamped = Math.max(1e-10, Math.min(1 - 1e-10, p));
        // Simple linear scale 0..1 → chartH..0
        return PAD_T + chartH * (1 - clamped);
    };

    // Build SVG path for the stepped line
    const pathD = useMemo(() => {
        if (steps.length === 0) return "";
        const points = steps.map((s, i) => ({
            x: PAD_L + i * xStep,
            y: toY(s.posterior),
        }));

        let d = `M ${PAD_L} ${toY(priorHp)}`;
        for (const pt of points) {
            d += ` L ${pt.x} ${pt.y}`;
        }
        return d;
    }, [steps, xStep, priorHp]);

    // Gradient area path
    const areaD = useMemo(() => {
        if (steps.length === 0) return "";
        const baseline = PAD_T + chartH;
        let d = `M ${PAD_L} ${baseline} L ${PAD_L} ${toY(priorHp)}`;
        for (let i = 0; i < steps.length; i++) {
            const x = PAD_L + i * xStep;
            const y = toY(steps[i].posterior);
            d += ` L ${x} ${y}`;
        }
        d += ` L ${PAD_L + (steps.length - 1) * xStep} ${baseline} Z`;
        return d;
    }, [steps, xStep, priorHp]);

    // Y-axis ticks
    const yTicks = [0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0];

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
                <TrendingUp size={16} style={{ color: "rgba(34, 197, 94, 0.8)" }} />
                <span
                    style={{
                        fontSize: 11,
                        fontWeight: 600,
                        letterSpacing: "0.08em",
                        textTransform: "uppercase",
                        color: "rgba(161, 161, 170, 0.9)",
                    }}
                >
                    Bayesian Shift Chart
                </span>
                <span
                    style={{
                        fontSize: 10,
                        color: "rgba(34, 197, 94, 0.8)",
                        background: "rgba(34, 197, 94, 0.1)",
                        padding: "2px 8px",
                        borderRadius: 4,
                        marginLeft: "auto",
                        fontFamily: "monospace",
                    }}
                >
                    P(Hp|E) = {posteriorHp.toFixed(6)}
                </span>
            </div>

            {/* SVG Chart */}
            <svg
                viewBox={`0 0 ${W} ${H}`}
                style={{ width: "100%", height: "auto" }}
                preserveAspectRatio="xMidYMid meet"
            >
                <defs>
                    <linearGradient id="bayesAreaGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="rgba(59, 130, 246, 0.25)" />
                        <stop offset="100%" stopColor="rgba(59, 130, 246, 0.02)" />
                    </linearGradient>
                    <filter id="glow">
                        <feGaussianBlur stdDeviation="3" result="blur" />
                        <feMerge>
                            <feMergeNode in="blur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Grid lines */}
                {yTicks.map((tick) => (
                    <g key={tick}>
                        <line
                            x1={PAD_L}
                            y1={toY(tick)}
                            x2={W - PAD_R}
                            y2={toY(tick)}
                            stroke="rgba(255,255,255,0.06)"
                            strokeDasharray="3,3"
                        />
                        <text
                            x={PAD_L - 8}
                            y={toY(tick) + 3}
                            textAnchor="end"
                            fill="rgba(161,161,170,0.5)"
                            fontSize={9}
                            fontFamily="monospace"
                        >
                            {tick < 0.01 ? tick.toExponential(0) : tick.toFixed(2)}
                        </text>
                    </g>
                ))}

                {/* Area fill */}
                {areaD && (
                    <motion.path
                        d={areaD}
                        fill="url(#bayesAreaGrad)"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ duration: 1 }}
                    />
                )}

                {/* CI band */}
                {ciLower > 0 && ciUpper > 0 && (
                    <rect
                        x={PAD_L}
                        y={toY(ciUpper)}
                        width={chartW}
                        height={Math.max(1, toY(ciLower) - toY(ciUpper))}
                        fill="rgba(59, 130, 246, 0.08)"
                        rx={2}
                    />
                )}

                {/* Prior marker line */}
                <line
                    x1={PAD_L}
                    y1={toY(priorHp)}
                    x2={W - PAD_R}
                    y2={toY(priorHp)}
                    stroke="rgba(251, 191, 36, 0.4)"
                    strokeDasharray="6,4"
                    strokeWidth={1}
                />
                <text
                    x={W - PAD_R + 2}
                    y={toY(priorHp) + 3}
                    fill="rgba(251, 191, 36, 0.6)"
                    fontSize={8}
                    fontFamily="monospace"
                >
                    Prior
                </text>

                {/* Stepped line */}
                {pathD && (
                    <motion.path
                        d={pathD}
                        fill="none"
                        stroke="rgba(59, 130, 246, 0.9)"
                        strokeWidth={2}
                        strokeLinejoin="round"
                        filter="url(#glow)"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{ duration: 1.5, ease: "easeInOut" }}
                    />
                )}

                {/* Data points */}
                {steps.map((step, i) => {
                    const cx = PAD_L + i * xStep;
                    const cy = toY(step.posterior);
                    const hasError =
                        (perLocusDetails[i]?.dropout_probability ?? 0) > 0.05 ||
                        (perLocusDetails[i]?.stutter_probability ?? 0) > 0.15;

                    return (
                        <g key={step.marker}>
                            <motion.circle
                                cx={cx}
                                cy={cy}
                                r={hasError ? 4 : 3}
                                fill={hasError ? "rgba(239, 68, 68, 0.9)" : "rgba(59, 130, 246, 0.9)"}
                                stroke={hasError ? "rgba(239, 68, 68, 0.3)" : "rgba(59, 130, 246, 0.3)"}
                                strokeWidth={hasError ? 3 : 2}
                                initial={{ scale: 0 }}
                                animate={{ scale: 1 }}
                                transition={{ delay: 0.5 + i * 0.05 }}
                            />
                            {/* X-axis label (marker name) */}
                            <text
                                x={cx}
                                y={H - PAD_B + 14}
                                textAnchor="end"
                                fill={hasError ? "rgba(239, 68, 68, 0.7)" : "rgba(161,161,170,0.5)"}
                                fontSize={7}
                                fontFamily="monospace"
                                transform={`rotate(-45, ${cx}, ${H - PAD_B + 14})`}
                            >
                                {step.marker}
                            </text>
                        </g>
                    );
                })}

                {/* Final value highlight */}
                {steps.length > 0 && (
                    <g>
                        <motion.circle
                            cx={PAD_L + (steps.length - 1) * xStep}
                            cy={toY(steps[steps.length - 1].posterior)}
                            r={6}
                            fill="rgba(34, 197, 94, 0.9)"
                            filter="url(#glow)"
                            initial={{ scale: 0 }}
                            animate={{ scale: [0, 1.3, 1] }}
                            transition={{ delay: 1.5, duration: 0.6 }}
                        />
                        <text
                            x={PAD_L + (steps.length - 1) * xStep}
                            y={toY(steps[steps.length - 1].posterior) - 12}
                            textAnchor="middle"
                            fill="rgba(34, 197, 94, 0.95)"
                            fontSize={10}
                            fontWeight={600}
                            fontFamily="monospace"
                        >
                            {steps[steps.length - 1].posterior >= 0.9999
                                ? "≈ 1.0"
                                : steps[steps.length - 1].posterior.toFixed(4)}
                        </text>
                    </g>
                )}

                {/* Axes labels */}
                <text
                    x={PAD_L - 5}
                    y={PAD_T - 10}
                    fill="rgba(161,161,170,0.5)"
                    fontSize={9}
                    fontFamily="monospace"
                >
                    P(Hp|E)
                </text>
                <text
                    x={W / 2}
                    y={H - 4}
                    textAnchor="middle"
                    fill="rgba(161,161,170,0.4)"
                    fontSize={9}
                    fontFamily="monospace"
                >
                    Loci Validated →
                </text>
            </svg>

            {/* CI footer */}
            <div
                style={{
                    marginTop: 8,
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 10,
                    color: "rgba(161, 161, 170, 0.5)",
                    fontFamily: "monospace",
                }}
            >
                <span>95% HPD: [{ciLower.toFixed(6)}, {ciUpper.toFixed(6)}]</span>
                <span>Prior: {priorHp.toExponential(2)}</span>
            </div>
        </div>
    );
}
