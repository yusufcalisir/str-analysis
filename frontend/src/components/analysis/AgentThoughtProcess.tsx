"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Terminal,
    BrainCircuit,
    Dna,
    Scale,
    Search,
    FileCheck,
    AlertTriangle,
    CheckCircle,
    Clock,
    ChevronRight,
    Loader2,
    Shield,
    Activity,
    BarChart3,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ThoughtStep {
    stepNumber: number;
    phase: string;
    content: string;
    durationMs: number;
    confidence: number;
    timestamp: number;
}

interface InvestigationState {
    queryId: string;
    status: "idle" | "running" | "complete" | "error";
    currentPhase: string;
    steps: ThoughtStep[];
    classification: string;
    recommendedAction: string;
    verbalEquivalence: string;
    combinedLR: string;
    totalTimeMs: number;
}

// ─── Phase Config ────────────────────────────────────────────────────────────

const PHASE_CONFIG: Record<string, { icon: typeof Terminal; label: string; color: string }> = {
    SAMPLE_ASSESSMENT: { icon: Dna, label: "Sample Quality Assessment", color: "text-cyan-400" },
    SCORE_EVALUATION: { icon: BarChart3, label: "Global Score Evaluation", color: "text-blue-400" },
    LIKELIHOOD_RATIO: { icon: Scale, label: "Likelihood Ratio Computation", color: "text-violet-400" },
    CLASSIFICATION: { icon: Shield, label: "Match Classification", color: "text-amber-400" },
    HYPOTHESIS_GENERATION: { icon: BrainCircuit, label: "Forensic Hypothesis", color: "text-emerald-400" },
    CERTAINTY_REPORT: { icon: FileCheck, label: "Certainty Report", color: "text-tactical-primary" },
    ADAPTIVE_REQUERY: { icon: Search, label: "Adaptive Re-Query", color: "text-orange-400" },
    REQUERY_ERROR: { icon: AlertTriangle, label: "Re-Query Error", color: "text-red-400" },
};

// ─── Mock Investigation Data ─────────────────────────────────────────────────

const MOCK_STEPS: ThoughtStep[] = [
    {
        stepNumber: 1,
        phase: "SAMPLE_ASSESSMENT",
        content: "DNA sample assessed as moderate. Evidence type: blood stain from crime scene. CODIS-20 panel detected with 18/20 loci amplified. Minor degradation at D2S1338 and D19S433. Proceeding with adjusted tolerance for partially degraded markers.",
        durationMs: 45.2,
        confidence: 0.88,
        timestamp: Date.now(),
    },
    {
        stepNumber: 2,
        phase: "SCORE_EVALUATION",
        content: "Analyzed 12 results from global broadcast. Top-10 extracted. High-confidence (≥0.95): 1 hit — Node EUROPOL-NL. Familial range (0.85–0.95): 2 hits — Node BKA-DE (0.912), Node NCA-UK (0.887). Ambiguous (0.78–0.85): 3 hits below actionable threshold.",
        durationMs: 12.8,
        confidence: 0.95,
        timestamp: Date.now(),
    },
    {
        stepNumber: 3,
        phase: "LIKELIHOOD_RATIO",
        content: "Combined LR = 4.72e+14 (log10 = 14.67). Verbal: EXTREMELY_STRONG_SUPPORT. P(Hp|E) = 0.9999. RMP = 2.12e-15. Per-locus LR computed across 18 informative loci with Hardy-Weinberg equilibrium and Balding-Nichols θ=0.01 correction.",
        durationMs: 28.6,
        confidence: 0.9999,
        timestamp: Date.now(),
    },
    {
        stepNumber: 4,
        phase: "CLASSIFICATION",
        content: "Match classified as DIRECT_IDENTITY. Recommended action: CONFIRM_MATCH. Based on 1 direct hit with score 0.9847 from EUROPOL-NL, supported by Combined LR of 4.72e+14 exceeding the ISFG threshold for extremely strong evidential support.",
        durationMs: 8.4,
        confidence: 0.9999,
        timestamp: Date.now(),
    },
    {
        stepNumber: 5,
        phase: "HYPOTHESIS_GENERATION",
        content: "The highest-scoring match was obtained from node EUROPOL-NL with similarity 0.9847 (reference: LRT-7a3f19...). The statistical evidence provides extremely strong support for the proposition that the evidential and reference profiles originate from the same individual. 2 additional familial-range hits from BKA-DE and NCA-UK suggest extended family members may also be indexed.",
        durationMs: 52.1,
        confidence: 0.9999,
        timestamp: Date.now(),
    },
    {
        stepNumber: 6,
        phase: "CERTAINTY_REPORT",
        content: "Certainty report generated. Verbal equivalence: EXTREMELY STRONG SUPPORT. ISO 17025 compliant report prepared for legal proceedings. RMP: 1 in 472 trillion. Confidence confirmed at 99.99%.",
        durationMs: 15.3,
        confidence: 0.9999,
        timestamp: Date.now(),
    },
];

// ─── Confidence Bar ──────────────────────────────────────────────────────────

function ConfidenceBar({ value, size = "sm" }: { value: number; size?: "sm" | "lg" }) {
    const pct = Math.round(value * 100);
    const color =
        pct >= 95 ? "bg-emerald-500"
            : pct >= 80 ? "bg-cyan-500"
                : pct >= 60 ? "bg-amber-500"
                    : "bg-red-500";

    const h = size === "lg" ? "h-2" : "h-1";

    return (
        <div className="flex items-center gap-2 w-full">
            <div className={`flex-1 ${h} bg-zinc-800 rounded-full overflow-hidden`}>
                <motion.div
                    className={`${h} ${color} rounded-full`}
                    initial={{ width: 0 }}
                    animate={{ width: `${pct}%` }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                />
            </div>
            <span className="font-data text-[9px] text-zinc-500 tabular-nums w-8 text-right">
                {pct}%
            </span>
        </div>
    );
}

// ─── Typewriter Effect ───────────────────────────────────────────────────────

function TypewriterText({ text, speed = 12 }: { text: string; speed?: number }) {
    const [displayed, setDisplayed] = useState("");
    const [done, setDone] = useState(false);

    useEffect(() => {
        setDisplayed("");
        setDone(false);
        let i = 0;
        const interval = setInterval(() => {
            i += 1;
            if (i >= text.length) {
                setDisplayed(text);
                setDone(true);
                clearInterval(interval);
            } else {
                setDisplayed(text.slice(0, i));
            }
        }, speed);
        return () => clearInterval(interval);
    }, [text, speed]);

    return (
        <span>
            {displayed}
            {!done && (
                <span className="inline-block w-[6px] h-[12px] bg-tactical-primary animate-pulse ml-0.5 align-middle" />
            )}
        </span>
    );
}

// ─── Thought Step Component ──────────────────────────────────────────────────

function ThoughtStepCard({
    step,
    isActive,
    isLast,
}: {
    step: ThoughtStep;
    isActive: boolean;
    isLast: boolean;
}) {
    const config = PHASE_CONFIG[step.phase] ?? {
        icon: Terminal,
        label: step.phase,
        color: "text-zinc-400",
    };
    const Icon = config.icon;

    return (
        <motion.div
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="relative group"
        >
            {/* Timeline connector */}
            {!isLast && (
                <div className="absolute left-[11px] top-[28px] bottom-[-4px] w-px bg-zinc-800" />
            )}

            <div className={`flex gap-3 ${isActive ? "" : "opacity-80"}`}>
                {/* Timeline dot */}
                <div className="flex-shrink-0 mt-1">
                    <div
                        className={`w-[22px] h-[22px] rounded-full flex items-center justify-center border ${isActive
                            ? "border-tactical-primary bg-tactical-primary/10"
                            : "border-zinc-700 bg-zinc-900"
                            }`}
                    >
                        {isActive ? (
                            <Loader2 className="w-3 h-3 text-tactical-primary animate-spin" />
                        ) : (
                            <Icon className={`w-3 h-3 ${config.color}`} />
                        )}
                    </div>
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0 pb-4">
                    <div className="flex items-center gap-2 mb-1">
                        <span className={`font-data text-[9px] font-bold uppercase tracking-[0.15em] ${config.color}`}>
                            Step {step.stepNumber}: {config.label}
                        </span>
                        <span className="font-data text-[8px] text-zinc-600 tabular-nums">
                            {step.durationMs.toFixed(1)}ms
                        </span>
                    </div>

                    <div className="font-mono text-[11px] leading-relaxed text-zinc-300 bg-zinc-900/50 border border-zinc-800/50 rounded px-3 py-2">
                        {isActive ? (
                            <TypewriterText text={step.content} speed={8} />
                        ) : (
                            step.content
                        )}
                    </div>

                    <div className="mt-1.5">
                        <ConfidenceBar value={step.confidence} />
                    </div>
                </div>
            </div>
        </motion.div>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function AgentThoughtProcess() {
    const [investigation, setInvestigation] = useState<InvestigationState>({
        queryId: "Q-2026-0212-ALPHA",
        status: "idle",
        currentPhase: "",
        steps: [],
        classification: "",
        recommendedAction: "",
        verbalEquivalence: "",
        combinedLR: "",
        totalTimeMs: 0,
    });
    const [activeStepIdx, setActiveStepIdx] = useState(-1);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Simulate investigation run
    const runInvestigation = useCallback(() => {
        setInvestigation((prev) => ({
            ...prev,
            status: "running",
            steps: [],
            currentPhase: "",
            classification: "",
            recommendedAction: "",
            verbalEquivalence: "",
            combinedLR: "",
            totalTimeMs: 0,
        }));
        setActiveStepIdx(-1);

        let stepIndex = 0;

        const addStep = () => {
            if (stepIndex >= MOCK_STEPS.length) {
                setInvestigation((prev) => ({
                    ...prev,
                    status: "complete",
                    currentPhase: "COMPLETE",
                    classification: "DIRECT_IDENTITY",
                    recommendedAction: "CONFIRM_MATCH",
                    verbalEquivalence: "EXTREMELY STRONG SUPPORT",
                    combinedLR: "4.72e+14",
                    totalTimeMs: MOCK_STEPS.reduce((s, st) => s + st.durationMs, 0),
                }));
                setActiveStepIdx(-1);
                return;
            }

            const step = MOCK_STEPS[stepIndex];
            setActiveStepIdx(stepIndex);
            setInvestigation((prev) => ({
                ...prev,
                currentPhase: step.phase,
                steps: [...prev.steps, { ...step, timestamp: Date.now() }],
            }));

            // Scroll to bottom
            setTimeout(() => {
                scrollRef.current?.scrollTo({
                    top: scrollRef.current.scrollHeight,
                    behavior: "smooth",
                });
            }, 100);

            stepIndex += 1;

            // Advance to next step after content "types out"
            const typeTime = step.content.length * 8 + 400;
            setTimeout(() => {
                setActiveStepIdx(-1);
                setTimeout(addStep, 300);
            }, typeTime);
        };

        setTimeout(addStep, 500);
    }, []);

    // Auto-run on mount
    useEffect(() => {
        const timer = setTimeout(runInvestigation, 1500);
        return () => clearTimeout(timer);
    }, [runInvestigation]);

    const isRunning = investigation.status === "running";
    const isComplete = investigation.status === "complete";

    return (
        <div className="flex flex-col gap-0 border border-tactical-border rounded overflow-hidden bg-tactical-bg">
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-4 py-3 bg-tactical-surface border-b border-tactical-border">
                <div className="flex items-center gap-3">
                    <div className="relative">
                        <BrainCircuit className="w-4 h-4 text-tactical-primary" />
                        {isRunning && (
                            <span className="absolute -top-0.5 -right-0.5 w-2 h-2 rounded-full bg-tactical-primary animate-ping" />
                        )}
                    </div>
                    <div>
                        <h3 className="font-data text-[10px] font-bold tracking-[0.15em] uppercase text-tactical-text">
                            Forensic Investigator Agent
                        </h3>
                        <p className="font-data text-[8px] text-zinc-600">
                            DSPy Chain-of-Thought • ISO 17025
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    {/* Query ID */}
                    <span className="font-data text-[8px] text-zinc-600 bg-zinc-900 border border-zinc-800 rounded px-2 py-0.5">
                        {investigation.queryId}
                    </span>
                    {/* Status indicator */}
                    <div className="flex items-center gap-1.5">
                        {isRunning && <Loader2 className="w-3 h-3 text-tactical-primary animate-spin" />}
                        {isComplete && <CheckCircle className="w-3 h-3 text-emerald-400" />}
                        {investigation.status === "idle" && <Clock className="w-3 h-3 text-zinc-600" />}
                        <span className={`font-data text-[8px] font-bold uppercase tracking-wider ${isRunning ? "text-tactical-primary" : isComplete ? "text-emerald-400" : "text-zinc-600"
                            }`}>
                            {investigation.status}
                        </span>
                    </div>
                </div>
            </div>

            {/* ── Terminal Feed ── */}
            <div
                ref={scrollRef}
                className="flex-1 overflow-y-auto px-4 py-3 min-h-[300px] max-h-[520px] bg-[#0a0a0a]"
            >
                {/* Prompt line */}
                <div className="font-mono text-[10px] text-zinc-600 mb-3 flex items-center gap-1.5">
                    <ChevronRight className="w-3 h-3 text-tactical-primary" />
                    <span className="text-tactical-primary">vantage-agent</span>
                    <span className="text-zinc-700">$</span>
                    <span className="text-zinc-400">
                        investigate --query={investigation.queryId} --mode=forensic --iso17025
                    </span>
                </div>

                {investigation.status === "idle" && (
                    <div className="font-mono text-[11px] text-zinc-600 animate-pulse">
                        Awaiting investigation trigger...
                    </div>
                )}

                <AnimatePresence mode="sync">
                    {investigation.steps.map((step, i) => (
                        <ThoughtStepCard
                            key={`${step.stepNumber}-${step.phase}`}
                            step={step}
                            isActive={i === activeStepIdx}
                            isLast={i === investigation.steps.length - 1 && !isRunning}
                        />
                    ))}
                </AnimatePresence>

                {/* Completion summary */}
                <AnimatePresence>
                    {isComplete && (
                        <motion.div
                            initial={{ opacity: 0, y: 8 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.3 }}
                            className="mt-4 border border-emerald-500/20 bg-emerald-500/5 rounded p-3"
                        >
                            <div className="flex items-center gap-2 mb-2">
                                <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
                                <span className="font-data text-[10px] font-bold tracking-[0.12em] uppercase text-emerald-400">
                                    Investigation Complete
                                </span>
                            </div>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                                {[
                                    { label: "Classification", value: investigation.classification, color: "text-amber-400" },
                                    { label: "Action", value: investigation.recommendedAction, color: "text-cyan-400" },
                                    { label: "Verbal Eq.", value: investigation.verbalEquivalence, color: "text-emerald-400" },
                                    { label: "Combined LR", value: investigation.combinedLR, color: "text-violet-400" },
                                ].map((item) => (
                                    <div key={item.label} className="flex flex-col gap-0.5">
                                        <span className="font-data text-[7px] uppercase tracking-wider text-zinc-600">
                                            {item.label}
                                        </span>
                                        <span className={`font-data text-[10px] font-bold ${item.color} truncate`}>
                                            {item.value}
                                        </span>
                                    </div>
                                ))}
                            </div>
                            <div className="mt-2 flex items-center gap-2">
                                <Activity className="w-3 h-3 text-zinc-600" />
                                <span className="font-data text-[8px] text-zinc-600">
                                    {investigation.steps.length} steps • {investigation.totalTimeMs.toFixed(1)}ms total
                                </span>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* ── Footer ── */}
            <div className="flex items-center justify-between px-4 py-2 border-t border-tactical-border bg-tactical-surface">
                <div className="flex items-center gap-1.5">
                    <Terminal className="w-3 h-3 text-zinc-600" />
                    <span className="font-data text-[8px] text-zinc-600">
                        {isRunning
                            ? `Processing: ${PHASE_CONFIG[investigation.currentPhase]?.label ?? investigation.currentPhase}`
                            : isComplete
                                ? "Analysis pipeline complete"
                                : "Ready"}
                    </span>
                </div>
                <button
                    onClick={runInvestigation}
                    disabled={isRunning}
                    className={`font-data text-[8px] font-bold uppercase tracking-wider px-3 py-1 rounded border transition-all ${isRunning
                        ? "text-zinc-600 border-zinc-800 cursor-not-allowed"
                        : "text-tactical-primary border-tactical-primary/30 hover:bg-tactical-primary/10 cursor-pointer"
                        }`}
                >
                    {isRunning ? "Running..." : "Re-run Analysis"}
                </button>
            </div>
        </div>
    );
}
