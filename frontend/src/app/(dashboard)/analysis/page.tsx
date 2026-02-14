"use client";

import { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { FlaskConical, Dna, Radio, Loader2, RefreshCw, BarChart3, GitBranch, Brain, Trash2 } from "lucide-react";
import AgentThoughtProcess from "@/components/analysis/AgentThoughtProcess";
import { MatchResultCardDemo } from "@/components/analysis/MatchResultCard";
import SuspectVisualizer from "@/components/analysis/SuspectVisualizer";
import ForensicStatsBanner from "@/components/analysis/ForensicStatsBanner";
import RarityHeatmap from "@/components/analysis/RarityHeatmap";
import PedigreeTree from "@/components/analysis/PedigreeTree";
import LociSensitivityMap from "@/components/analysis/LociSensitivityMap";
import BayesianShiftChart from "@/components/analysis/BayesianShiftChart";
import { useIngestStore } from "@/store/ingestStore";

// ─── Types ───────────────────────────────────────────────────────────────────

interface AnalysisData {
    profile_id: string;
    population: string;
    combined_lr: number;
    log10_lr: number;
    random_match_probability: number;
    random_match_probability_str: string;
    verbal_equivalence: string;
    prosecution_probability: number;
    defense_probability: number;
    loci_analyzed: number;
    per_locus_details: any[];
    high_frequency_warning: boolean;
    warning_message: string;
    match_classification: string;
    recommended_action: string;
    forensic_hypothesis: string;
    certainty_report: string;
    thought_chain: any[];
    total_analysis_time_ms: number;
    kinship_result: any | null;
    familial_hit_detected: boolean;
    // Phase 3.7 — Bayesian
    bayesian_posterior: number;
    prior_hp: number;
    bayesian_ci_lower: number;
    bayesian_ci_upper: number;
    degradation_index: number;
    dropout_warnings: string[];
    stutter_warnings: string[];
    iso17025_verbal: string;
    sensitivity_map: any[];
}

type TabId = "statistical" | "relationship" | "bayesian";

// ─── API ─────────────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function fetchAnalysis(profileId: string, population: string): Promise<AnalysisData> {
    const res = await fetch(`${API_BASE}/profile/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_id: profileId, population }),
    });
    if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
    return res.json();
}

async function fetchKinship(profileAId: string, profileBId: string, population: string) {
    const res = await fetch(`${API_BASE}/profile/kinship`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_a_id: profileAId, profile_b_id: profileBId, population }),
    });
    if (!res.ok) return null;
    return res.json();
}

// ─── Tab Button ──────────────────────────────────────────────────────────────

function TabButton({
    id,
    label,
    icon: Icon,
    activeTab,
    onClick,
    badge,
}: {
    id: TabId;
    label: string;
    icon: typeof BarChart3;
    activeTab: TabId;
    onClick: (id: TabId) => void;
    badge?: string;
}) {
    const isActive = activeTab === id;
    return (
        <button
            onClick={() => onClick(id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-data text-[8px] font-bold tracking-wider uppercase transition-all ${isActive
                ? "bg-tactical-primary/15 text-tactical-primary border border-tactical-primary/25"
                : "bg-zinc-900/50 text-zinc-500 border border-zinc-800/50 hover:text-zinc-300 hover:border-zinc-700"
                }`}
        >
            <Icon className="w-3 h-3" />
            {label}
            {badge && (
                <span className="px-1 py-0.5 rounded text-[6px] bg-amber-500/20 text-amber-400 border border-amber-500/30">
                    {badge}
                </span>
            )}
        </button>
    );
}

// ─── Main Component ──────────────────────────────────────────────────────────

export default function AnalysisPage() {
    const { lastIngestedProfileId, lastIngestedNodeId, isValid, setLastIngested } = useIngestStore();

    const [population, setPopulation] = useState("European");
    const [analysis, setAnalysis] = useState<AnalysisData | null>(null);
    const [kinship, setKinship] = useState<any | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<TabId>("statistical");

    const activeProfileId = lastIngestedProfileId || "test-profile-eu";

    const runAnalysis = useCallback(async (profileId: string, pop: string) => {
        setLoading(true);
        setError(null);
        let shouldStopLoading = true;

        try {
            // 1. Critical Phase: Main Profile Analysis
            const data = await fetchAnalysis(profileId, pop);
            setAnalysis(data);

            // 2. Secondary Phase: Kinship Analytics (Non-blocking)
            try {
                if (data.familial_hit_detected && data.kinship_result) {
                    setKinship(data.kinship_result);
                } else {
                    // Fallback demo between test profiles if no hit
                    const kinshipData = await fetchKinship("test-profile-eu", "test-profile-af", pop);
                    setKinship(kinshipData);
                }
            } catch (kinshipError) {
                console.warn("[ANALYSIS] Kinship data unavailable (non-critical):", kinshipError);
                // We do NOT fail the whole page for this
                setKinship(null);
            }

        } catch (e: any) {
            console.error("[ANALYSIS] Critical failure:", e);
            let msg = e.message || "Unknown analysis error";

            // specific 422 handling: Profile not found (likely backend restart cleared memory)
            if (msg.includes("422") && profileId !== "test-profile-eu") {
                console.warn("[ANALYSIS] Profile not found (422). Auto-falling back to test-profile-eu.");

                // Sync global state so the UI header updates to match the data being shown
                // useIngestStore.getState().setLastIngested(...) could work too, but we have the setter.
                setLastIngested("test-profile-eu", "DEMO-NODE", 24);

                shouldStopLoading = false; // Don't stop loading, we are recurring
                runAnalysis("test-profile-eu", pop);
                return;
            }

            // User-friendly network error
            if (msg.includes("Failed to fetch")) {
                msg = `Network Error: Cannot reach backend at ${API_BASE}. Ensure server is running.`;
            }

            setError(msg);
            setAnalysis(null); // Clear stale data on error to avoid confusion
        } finally {
            if (shouldStopLoading) {
                setLoading(false);
            }
        }
    }, []);

    useEffect(() => {
        runAnalysis(activeProfileId, population);
    }, [activeProfileId, population, runAnalysis]);

    const handlePopulationChange = useCallback((pop: string) => {
        setPopulation(pop);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="space-y-5"
        >
            {/* ── Header ── */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 min-w-0">
                    <FlaskConical className="w-4 h-4 text-tactical-primary shrink-0" />
                    <h1 className="font-data text-[10px] lg:text-xs font-bold tracking-[0.1em] lg:tracking-[0.2em] text-tactical-text uppercase truncate">
                        Forensic_Analysis_Engine
                    </h1>
                </div>
                <div className="flex items-center gap-3">
                    {loading && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            className="flex items-center gap-1.5"
                        >
                            <Loader2 className="w-3 h-3 text-tactical-primary animate-spin" />
                            <span className="font-data text-[8px] text-tactical-primary uppercase tracking-wider">
                                Computing...
                            </span>
                        </motion.div>
                    )}
                    <motion.div
                        initial={{ opacity: 0, x: 12 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="flex items-center gap-2 rounded-full border border-tactical-primary/20 bg-tactical-primary/5 px-2.5 py-1 lg:px-3 lg:py-1.5"
                    >
                        <div className="relative flex h-1.5 w-1.5 lg:h-2 lg:w-2">
                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-tactical-primary opacity-60" />
                            <span className="relative inline-flex h-1.5 w-1.5 lg:h-2 lg:w-2 rounded-full bg-tactical-primary" />
                        </div>
                        <span className="font-data text-[8px] lg:text-[9px] text-tactical-primary font-semibold tracking-wider uppercase truncate max-w-[100px] lg:max-w-none">
                            {activeProfileId}
                        </span>
                        {lastIngestedNodeId && (
                            <span className="font-data text-[7px] lg:text-[8px] text-zinc-500 hidden sm:inline">
                                from {lastIngestedNodeId}
                            </span>
                        )}
                    </motion.div>
                </div>
            </div>

            {/* ── Context Banner ── */}
            <div className="flex flex-col sm:flex-row sm:items-center gap-3 px-4 py-3 rounded-lg border border-tactical-border bg-tactical-surface">
                <div className="flex items-center gap-3 min-w-0">
                    <Dna className="w-4 h-4 text-cyan-400 flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                        <p className="font-data text-[9px] lg:text-[10px] font-bold text-tactical-text uppercase tracking-[0.12em] truncate">
                            Dynamic LR Engine + Kinship Analytics
                        </p>
                        <p className="font-data text-[7px] lg:text-[8px] text-zinc-600 mt-0.5 truncate">
                            Balding-Nichols NRC II • IBD-based Kinship Index
                        </p>
                    </div>
                </div>
                <div className="flex items-center justify-between sm:justify-end gap-2 border-t border-tactical-border sm:border-t-0 pt-2 sm:pt-0">
                    <button
                        onClick={useIngestStore((s) => s.clear)}
                        className="flex items-center gap-1 px-2 py-1 rounded border border-tactical-danger/30 hover:border-tactical-danger text-tactical-danger/70 hover:text-tactical-danger transition-all"
                    >
                        <Trash2 className="w-3 h-3" />
                        <span className="font-data text-[7px] uppercase tracking-wider">Clear</span>
                    </button>
                    <button
                        onClick={() => runAnalysis(activeProfileId, population)}
                        disabled={loading || !isValid}
                        className="flex items-center gap-1 px-2 py-1 rounded border border-zinc-800 hover:border-zinc-700 text-zinc-500 hover:text-zinc-300 transition-all disabled:opacity-30 disabled:cursor-not-allowed group"
                        title={!isValid ? "Ingest at least 13 markers to run analysis" : ""}
                    >
                        <RefreshCw className={`w-3 h-3 ${loading ? "animate-spin" : ""}`} />
                        <span className="font-data text-[7px] uppercase tracking-wider group-disabled:text-zinc-700">
                            {!isValid ? "Awaiting Data" : "Run Analysis"}
                        </span>
                    </button>
                    <div className="flex items-center gap-1.5">
                        <Radio className={`w-3 h-3 ${isValid ? "text-tactical-primary animate-pulse" : "text-zinc-700"}`} />
                        <span className={`font-data text-[8px] font-bold uppercase tracking-wider ${isValid ? "text-tactical-primary" : "text-zinc-700"}`}>
                            {isValid ? "Live" : "Standby"}
                        </span>
                    </div>
                </div>
            </div>

            {/* ── Error State ── */}
            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="px-4 py-3 rounded border border-red-500/20 bg-red-500/5"
                    >
                        <p className="font-data text-[9px] text-red-400">
                            Analysis Error: {error}. Backend may be offline.
                        </p>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* ── Forensic Stats Banner ── */}
            {analysis && (
                <ForensicStatsBanner
                    combinedLR={analysis.combined_lr}
                    log10LR={analysis.log10_lr}
                    randomMatchProbability={analysis.random_match_probability}
                    randomMatchProbabilityStr={analysis.random_match_probability_str}
                    verbalEquivalence={analysis.verbal_equivalence}
                    prosecutionProbability={analysis.prosecution_probability}
                    matchClassification={analysis.match_classification}
                    recommendedAction={analysis.recommended_action}
                    lociAnalyzed={analysis.loci_analyzed}
                    highFrequencyWarning={analysis.high_frequency_warning}
                    warningMessage={analysis.warning_message}
                    population={analysis.population}
                    totalAnalysisTimeMs={analysis.total_analysis_time_ms}
                />
            )}

            {/* ── Tab Switcher ── */}
            <div className="flex items-center gap-2 overflow-x-auto pb-1 no-scrollbar sm:overflow-visible sm:pb-0">
                <TabButton
                    id="statistical"
                    label="Statistical Analysis"
                    icon={BarChart3}
                    activeTab={activeTab}
                    onClick={setActiveTab}
                />
                <TabButton
                    id="relationship"
                    label="Relationship Mapping"
                    icon={GitBranch}
                    activeTab={activeTab}
                    onClick={setActiveTab}
                    badge={kinship && kinship.relationship_type !== "UNRELATED" ? "HIT" : undefined}
                />
                <TabButton
                    id="bayesian"
                    label="Bayesian Inference"
                    icon={Brain}
                    activeTab={activeTab}
                    onClick={setActiveTab}
                    badge={analysis && analysis.degradation_index > 0.3 ? "⚠" : undefined}
                />
            </div>

            {/* ── Tab Content ── */}
            <AnimatePresence mode="wait">
                {activeTab === "statistical" && (
                    <motion.div
                        key="statistical"
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.2 }}
                    >
                        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5">
                            {/* Left Column */}
                            <div className="space-y-5 min-w-0">
                                {analysis && (
                                    <section>
                                        <RarityHeatmap
                                            perLocusDetails={analysis.per_locus_details}
                                            population={population}
                                            onPopulationChange={handlePopulationChange}
                                        />
                                    </section>
                                )}
                                <section>
                                    <AgentThoughtProcess />
                                </section>
                                <section>
                                    <div className="flex items-center gap-2 mb-3">
                                        <div className="w-1 h-4 rounded-full bg-tactical-primary" />
                                        <h2 className="font-data text-[10px] font-bold tracking-[0.15em] text-tactical-text uppercase">
                                            Global Verification Results
                                        </h2>
                                        <span className="font-data text-[8px] text-zinc-600">
                                            — Completeness-Aware Ranking
                                        </span>
                                    </div>
                                    <MatchResultCardDemo />
                                </section>
                            </div>

                            {/* Right Column */}
                            <aside className="lg:sticky lg:top-4 lg:self-start">
                                <SuspectVisualizer profileId={lastIngestedProfileId || undefined} />
                            </aside>
                        </div>
                    </motion.div>
                )}

                {activeTab === "relationship" && (
                    <motion.div
                        key="relationship"
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.2 }}
                    >
                        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5">
                            {/* Left Column — Pedigree Analysis */}
                            <div className="space-y-5 min-w-0">
                                {kinship ? (
                                    <PedigreeTree
                                        kinshipData={kinship}
                                        profileAId={activeProfileId}
                                        profileBId="test-profile-af"
                                    />
                                ) : (
                                    <div className="flex items-center justify-center h-48 border border-zinc-800/40 rounded bg-zinc-900/30">
                                        <p className="font-data text-[9px] text-zinc-600 uppercase tracking-wider">
                                            No kinship data available — ingest a second profile to compare
                                        </p>
                                    </div>
                                )}

                                {/* Hypothesis text */}
                                {analysis && (
                                    <div className="border border-zinc-800/40 rounded bg-tactical-bg p-4">
                                        <h3 className="font-data text-[9px] font-bold text-tactical-text uppercase tracking-wider mb-2">
                                            Agent Forensic Hypothesis
                                        </h3>
                                        <pre className="font-data text-[8px] text-zinc-400 whitespace-pre-wrap leading-relaxed">
                                            {analysis.forensic_hypothesis}
                                        </pre>
                                    </div>
                                )}
                            </div>

                            {/* Right Column */}
                            <aside className="lg:sticky lg:top-4 lg:self-start">
                                <SuspectVisualizer profileId={lastIngestedProfileId || undefined} />
                            </aside>
                        </div>
                    </motion.div>
                )}

                {activeTab === "bayesian" && analysis && (
                    <motion.div
                        key="bayesian"
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -6 }}
                        transition={{ duration: 0.2 }}
                    >
                        {/* ISO 17025 Verdict Ribbon */}
                        <div className="flex items-center gap-3 px-4 py-3 rounded-lg mb-5 border border-tactical-border bg-tactical-surface">
                            <Brain className="w-4 h-4 text-indigo-400 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                                <p className="font-data text-[10px] font-bold text-tactical-text uppercase tracking-[0.12em]">
                                    Bayesian Forensic Inference — Phase 3.7
                                </p>
                                <p className="font-data text-[8px] text-zinc-600 mt-0.5">
                                    Error-adjusted LR • Dynamic Prior • 95% HPD Interval • ISO 17025 Verbal Scale
                                </p>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="font-data text-[8px] px-2 py-1 rounded border border-indigo-500/30 bg-indigo-500/10 text-indigo-400 font-bold uppercase tracking-wider">
                                    {analysis.iso17025_verbal?.replace(/_/g, ' ') || 'INCONCLUSIVE'}
                                </span>
                            </div>
                        </div>

                        {/* Stats Row */}
                        <div className="grid grid-cols-4 gap-3 mb-5">
                            {[
                                { label: "Posterior P(Hp|E)", value: analysis.bayesian_posterior?.toFixed(6) || "0" },
                                { label: "Prior P(Hp)", value: analysis.prior_hp ? analysis.prior_hp.toExponential(2) : "5.00e-1" },
                                { label: "95% HPD", value: `[${(analysis.bayesian_ci_lower || 0).toFixed(4)}, ${(analysis.bayesian_ci_upper || 0).toFixed(4)}]` },
                                { label: "Degradation Index", value: `${((analysis.degradation_index || 0) * 100).toFixed(1)}%` },
                            ].map((stat) => (
                                <div key={stat.label} className="px-3 py-2.5 rounded border border-zinc-800/40 bg-zinc-900/40">
                                    <p className="font-data text-[7px] text-zinc-600 uppercase tracking-wider">{stat.label}</p>
                                    <p className="font-data text-[11px] text-tactical-text font-bold mt-0.5 font-mono">{stat.value}</p>
                                </div>
                            ))}
                        </div>

                        <div className="grid grid-cols-1 lg:grid-cols-[1fr_380px] gap-5">
                            {/* Left Column — Charts */}
                            <div className="space-y-5 min-w-0">
                                {/* Bayesian Shift Chart */}
                                <BayesianShiftChart
                                    perLocusDetails={analysis.per_locus_details || []}
                                    priorHp={analysis.prior_hp || 0.000001}
                                    posteriorHp={analysis.bayesian_posterior || 0}
                                    ciLower={analysis.bayesian_ci_lower || 0}
                                    ciUpper={analysis.bayesian_ci_upper || 0}
                                />

                                {/* Loci Sensitivity Map */}
                                <LociSensitivityMap
                                    sensitivityMap={analysis.sensitivity_map || []}
                                    degradationIndex={analysis.degradation_index || 0}
                                />

                                {/* Warnings Panel */}
                                {((analysis.dropout_warnings?.length || 0) > 0 || (analysis.stutter_warnings?.length || 0) > 0) && (
                                    <div className="border border-amber-500/20 rounded-lg bg-amber-500/5 p-4">
                                        <h3 className="font-data text-[9px] font-bold text-amber-400 uppercase tracking-wider mb-2">
                                            ⚠ Stochastic Error Warnings
                                        </h3>
                                        {analysis.dropout_warnings?.length > 0 && (
                                            <div className="mb-2">
                                                <p className="font-data text-[8px] text-amber-300/80 font-semibold mb-1">Allele Dropout</p>
                                                {analysis.dropout_warnings.map((w: string, i: number) => (
                                                    <p key={`do-${i}`} className="font-data text-[7px] text-zinc-500 pl-2">
                                                        • {w}
                                                    </p>
                                                ))}
                                            </div>
                                        )}
                                        {analysis.stutter_warnings?.length > 0 && (
                                            <div>
                                                <p className="font-data text-[8px] text-amber-300/80 font-semibold mb-1">Stutter Artifacts</p>
                                                {analysis.stutter_warnings.map((w: string, i: number) => (
                                                    <p key={`st-${i}`} className="font-data text-[7px] text-zinc-500 pl-2">
                                                        • {w}
                                                    </p>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* Right Column */}
                            <aside className="lg:sticky lg:top-4 lg:self-start">
                                <SuspectVisualizer profileId={lastIngestedProfileId || undefined} />
                            </aside>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </motion.div>
    );
}
