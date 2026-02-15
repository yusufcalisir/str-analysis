"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    ScanFace,
    Download,
    Loader2,
    Dna,
    Eye,
    Palette,
    User,
    Globe,
    RefreshCw,
    ChevronRight,
    AlertTriangle,
    CheckCircle2,
} from "lucide-react";
import { toPng } from "html-to-image";

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface TraitSummary {
    [key: string]: string;
}

interface ReconstructionData {
    profile_id: string;
    image_url?: string;
    seed?: number;
    prompt_hash?: string;
    generation_time_ms?: number;
    model_id?: string;
    trait_summary?: TraitSummary;
    positive_prompt?: string;
    negative_prompt?: string;
    coherence_score?: number;
    coherence_status?: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// MOCK DATA (Mirrors backend test-profile-eu response)
// ═══════════════════════════════════════════════════════════════════════════════

const MOCK_RECONSTRUCTION: ReconstructionData = {
    profile_id: "test-profile-eu",
    image_url: "https://randomuser.me/api/portraits/men/42.jpg", // Stable mock placeholder for offline mode
    seed: 1847293650,
    prompt_hash: "a3f2c1b8e9d4",
    generation_time_ms: 142.7,
    model_id: "mock-sdxl-dev",
    trait_summary: {
        "Eye Color": "Blue (85%)",
        "Hair Color": "Blond (42%)",
        "Skin Tone": "Light (78%)",
        "Ancestry": "European (65%)",
        "Sex": "Male",
    },
    positive_prompt:
        "((adult male portrait)), single person, ((European facial morphology, moderate brow ridge, narrow nasal bridge, defined cheekbones, medium lip volume, angular jawline)), ((piercing ice-blue eyes, light iris with limbal ring)), (light blond hair, natural golden tones, straight texture), (light skin tone, subtle warm undertones, Type II-III Fitzpatrick)",
    negative_prompt:
        "cartoon, anime, illustration, painting, drawing, sketch...",
};

// ═══════════════════════════════════════════════════════════════════════════════
// TRAIT ICON MAPPING
// ═══════════════════════════════════════════════════════════════════════════════

const TRAIT_ICONS: Record<string, typeof Eye> = {
    "Eye Color": Eye,
    "Hair Color": Palette,
    "Skin Tone": User,
    "Ancestry": Globe,
    "Sex": User,
};

// ═══════════════════════════════════════════════════════════════════════════════
// FORENSIC OVERLAY GRID SVG
// ═══════════════════════════════════════════════════════════════════════════════

function ForensicOverlay({ active }: { active: boolean }) {
    return (
        <svg
            className={`absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-700 ${active ? "opacity-40" : "opacity-0"}`}
            viewBox="0 0 400 560"
            preserveAspectRatio="none"
        >
            {/* Horizontal grid lines */}
            {Array.from({ length: 12 }, (_, i) => (
                <line
                    key={`h-${i}`}
                    x1="0"
                    y1={i * 48 + 20}
                    x2="400"
                    y2={i * 48 + 20}
                    stroke="#22C55E"
                    strokeWidth="0.5"
                    strokeDasharray="4 8"
                    opacity="0.3"
                />
            ))}
            {/* Vertical grid lines */}
            {Array.from({ length: 8 }, (_, i) => (
                <line
                    key={`v-${i}`}
                    x1={i * 52 + 20}
                    y1="0"
                    x2={i * 52 + 20}
                    y2="560"
                    stroke="#22C55E"
                    strokeWidth="0.5"
                    strokeDasharray="4 8"
                    opacity="0.3"
                />
            ))}
            {/* Center crosshair */}
            <line x1="180" y1="200" x2="220" y2="200" stroke="#22C55E" strokeWidth="1" opacity="0.6" />
            <line x1="200" y1="180" x2="200" y2="220" stroke="#22C55E" strokeWidth="1" opacity="0.6" />
            <circle cx="200" cy="200" r="60" fill="none" stroke="#22C55E" strokeWidth="0.5" strokeDasharray="3 6" opacity="0.3" />
            <circle cx="200" cy="200" r="120" fill="none" stroke="#22C55E" strokeWidth="0.5" strokeDasharray="3 6" opacity="0.2" />

            {/* Trait labels on overlay */}
            <text x="30" y="155" fill="#22C55E" fontSize="8" fontFamily="JetBrains Mono, monospace" opacity="0.7">IRIS_SCAN</text>
            <line x1="30" y1="158" x2="130" y2="200" stroke="#22C55E" strokeWidth="0.5" opacity="0.4" />

            <text x="280" y="120" fill="#22C55E" fontSize="8" fontFamily="JetBrains Mono, monospace" opacity="0.7">CRANIAL_STRUCT</text>
            <line x1="280" y1="123" x2="250" y2="165" stroke="#22C55E" strokeWidth="0.5" opacity="0.4" />

            <text x="290" y="280" fill="#22C55E" fontSize="8" fontFamily="JetBrains Mono, monospace" opacity="0.7">JAW_MORPH</text>
            <line x1="290" y1="275" x2="260" y2="310" stroke="#22C55E" strokeWidth="0.5" opacity="0.4" />

            <text x="20" y="310" fill="#22C55E" fontSize="8" fontFamily="JetBrains Mono, monospace" opacity="0.7">PIGMENT_IDX</text>
            <line x1="20" y1="305" x2="130" y2="250" stroke="#22C55E" strokeWidth="0.5" opacity="0.4" />

            {/* Corner brackets */}
            <path d="M10 10 L10 30 M10 10 L30 10" stroke="#22C55E" strokeWidth="1.5" fill="none" opacity="0.6" />
            <path d="M390 10 L390 30 M390 10 L370 10" stroke="#22C55E" strokeWidth="1.5" fill="none" opacity="0.6" />
            <path d="M10 550 L10 530 M10 550 L30 550" stroke="#22C55E" strokeWidth="1.5" fill="none" opacity="0.6" />
            <path d="M390 550 L390 530 M390 550 L370 550" stroke="#22C55E" strokeWidth="1.5" fill="none" opacity="0.6" />
        </svg>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCANNING ANIMATION
// ═══════════════════════════════════════════════════════════════════════════════

function ScanBeam({ active }: { active: boolean }) {
    if (!active) return null;
    return (
        <motion.div
            className="absolute left-0 right-0 h-[2px] z-20 pointer-events-none"
            style={{
                background: "linear-gradient(90deg, transparent, #22C55E, transparent)",
                boxShadow: "0 0 20px 4px rgba(34, 197, 94, 0.3)",
            }}
            animate={{ top: ["0%", "100%", "0%"] }}
            transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
        />
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// FORENSIC PLACEHOLDER (when no real image)
// ═══════════════════════════════════════════════════════════════════════════════

function ForensicPlaceholder() {
    return (
        <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-tactical-surface via-[#0d0d10] to-tactical-surface">
            <svg viewBox="0 0 200 280" className="w-48 h-auto opacity-20">
                {/* Head outline */}
                <ellipse cx="100" cy="95" rx="60" ry="75" fill="none" stroke="#22C55E" strokeWidth="1" />
                {/* Eyes */}
                <ellipse cx="78" cy="85" rx="12" ry="6" fill="none" stroke="#22C55E" strokeWidth="0.8" />
                <ellipse cx="122" cy="85" rx="12" ry="6" fill="none" stroke="#22C55E" strokeWidth="0.8" />
                <circle cx="78" cy="85" r="3" fill="#22C55E" opacity="0.5" />
                <circle cx="122" cy="85" r="3" fill="#22C55E" opacity="0.5" />
                {/* Nose */}
                <path d="M100 92 L95 115 L105 115 Z" fill="none" stroke="#22C55E" strokeWidth="0.8" />
                {/* Mouth */}
                <path d="M85 130 Q100 140 115 130" fill="none" stroke="#22C55E" strokeWidth="0.8" />
                {/* Neck */}
                <line x1="85" y1="165" x2="85" y2="200" stroke="#22C55E" strokeWidth="0.8" />
                <line x1="115" y1="165" x2="115" y2="200" stroke="#22C55E" strokeWidth="0.8" />
                {/* Shoulders */}
                <path d="M85 200 Q50 210 30 240" fill="none" stroke="#22C55E" strokeWidth="0.8" />
                <path d="M115 200 Q150 210 170 240" fill="none" stroke="#22C55E" strokeWidth="0.8" />
                {/* Measurement lines */}
                <line x1="30" y1="20" x2="30" y2="170" stroke="#22C55E" strokeWidth="0.3" strokeDasharray="2 4" />
                <line x1="170" y1="20" x2="170" y2="170" stroke="#22C55E" strokeWidth="0.3" strokeDasharray="2 4" />
                <text x="100" y="270" fill="#22C55E" fontSize="7" fontFamily="JetBrains Mono, monospace" textAnchor="middle" opacity="0.6">AWAITING_GENAI_RENDER</text>
            </svg>
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENERATION LOG LINE
// ═══════════════════════════════════════════════════════════════════════════════

const LOG_STEPS = [
    { msg: "Fetching SNP genotype data...", icon: Dna, delay: 0 },
    { msg: "Running HIrisPlex-S predictor...", icon: ScanFace, delay: 400 },
    { msg: "Composing SDXL prompt tokens...", icon: Palette, delay: 800 },
    { msg: "Generating facial reconstruction...", icon: RefreshCw, delay: 1200 },
    { msg: "Applying forensic overlay...", icon: CheckCircle2, delay: 1600 },
];

function GenerationLog({ isGenerating }: { isGenerating: boolean }) {
    const [visibleSteps, setVisibleSteps] = useState(0);

    useEffect(() => {
        if (!isGenerating) {
            setVisibleSteps(0);
            return;
        }
        const timers: NodeJS.Timeout[] = [];
        LOG_STEPS.forEach((step, idx) => {
            timers.push(setTimeout(() => setVisibleSteps(idx + 1), step.delay));
        });
        return () => timers.forEach(clearTimeout);
    }, [isGenerating]);

    if (!isGenerating) return null;

    return (
        <div className="space-y-1 min-h-[80px]">
            <AnimatePresence>
                {LOG_STEPS.slice(0, visibleSteps).map((step, idx) => {
                    const Icon = step.icon;
                    const isLast = idx === visibleSteps - 1;
                    return (
                        <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -8 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0 }}
                            className="flex items-center gap-2"
                        >
                            {isLast && isGenerating ? (
                                <Loader2 className="w-3 h-3 text-tactical-primary animate-spin" />
                            ) : (
                                <Icon className="w-3 h-3 text-tactical-primary" />
                            )}
                            <span className="font-mono text-[9px] text-zinc-500 tracking-wide">
                                {step.msg}
                            </span>
                            {!isLast && (
                                <CheckCircle2 className="w-2.5 h-2.5 text-tactical-primary/50 ml-auto" />
                            )}
                        </motion.div>
                    );
                })}
            </AnimatePresence>
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

interface ForensicIdentityCardProps {
    profileId?: string;
    hoveredRegion?: string | null;
    phenotypeReport?: {
        traits?: Record<string, string>;
        reliability_score?: number;
        coherence_score?: number;
        coherence_status?: string;
        snps_analyzed?: string[];
    } | null;
    coherenceScore?: number;
    txHash?: string;
    ancestryRegion?: string;
}

export default function SuspectVisualizer({
    profileId,
    hoveredRegion,
    phenotypeReport,
    coherenceScore,
    txHash,
    ancestryRegion
}: ForensicIdentityCardProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    const data = phenotypeReport;
    const isLoading = !data && !!profileId;

    // Safety check specific to phenotype data availability
    const hasData = data && data.traits && Object.keys(data.traits).length > 0;

    // Reliability formatting and color logic
    const reliabilityValue = coherenceScore ? (coherenceScore * 100).toFixed(1) : "0.0";
    const reliabilityColor = coherenceScore
        ? coherenceScore > 0.8 ? "text-emerald-500"
            : coherenceScore > 0.6 ? "text-amber-500"
                : "text-red-500"
        : "text-zinc-500";
    const reliabilityLabel = coherenceScore ? `${reliabilityValue}%` : "CALCULATING...";

    // Helper: Determine if trait matches hovered region
    const isTraitRelevant = (trait: string, value: string) => {
        if (!hoveredRegion) return false;
        const region = hoveredRegion.toLowerCase();
        const val = value.toLowerCase();

        if (region.includes("africa")) return (val.includes("dark") || val.includes("black") || val.includes("curly"));
        if (region.includes("europe")) return (val.includes("blue") || val.includes("light") || val.includes("blond"));
        if (region.includes("asia")) return (val.includes("dark") || val.includes("straight"));
        return false;
    };

    // Mapping backend traits to requested display labels
    const displayTraits = [
        {
            label: "BIOLOGICAL_EYE_COLOR",
            value: data?.traits?.["Ocular Pigmentation"] || "Unknown",
            key: "Ocular Pigmentation"
        },
        {
            label: "DERMAL_PIGMENTATION",
            value: data?.traits?.["Dermal Classification"] || "Unknown",
            key: "Dermal Classification"
        },
        {
            label: "HAIR_STRUCTURE",
            value: data?.traits?.["Hair Morphology"] || "Unknown",
            key: "Hair Morphology"
        },
        {
            label: "GENETIC_ANCESTRY_KEY",
            value: ancestryRegion || "Unknown",
            key: "Ancestry"
        }
    ];

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-lg border border-tactical-border bg-slate-950 overflow-hidden h-fit flex flex-col font-mono shadow-lg relative"
            ref={containerRef}
        >
            {/* ── Scanning Overlay Animation ── */}
            {isLoading && (
                <div className="absolute inset-0 z-20 pointer-events-none overflow-hidden">
                    <motion.div
                        className="w-full h-[2px] bg-cyan-400/80 shadow-[0_0_15px_rgba(34,211,238,0.8)]"
                        animate={{ top: ["0%", "100%", "0%"] }}
                        transition={{ duration: 3, ease: "linear", repeat: Infinity }}
                    />
                    <div className="absolute inset-0 bg-cyan-500/5 mix-blend-overlay" />
                </div>
            )}

            {/* ── Header ── */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800/50 shrink-0 bg-[#070709] relative z-10">
                <div className="flex items-center gap-2">
                    <User className="w-4 h-4 text-emerald-500" />
                    <h3 className="font-mono text-[10px] font-bold tracking-[0.2em] text-emerald-500 uppercase">
                        Forensic_Identity_Panel
                    </h3>
                </div>

                {hasData && (
                    <div className={`flex items-center gap-2 px-2 py-1 rounded-full border ${(coherenceScore || 0) > 0.85
                            ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400'
                            : 'bg-amber-500/10 border-amber-500/20 text-amber-400'
                        }`}>
                        {(coherenceScore || 0) > 0.85 ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                        <span className="font-mono text-[8px] font-bold tracking-tighter uppercase whitespace-nowrap">
                            {(coherenceScore || 0) > 0.85 ? "VERIFIED" : "LOW SYNC"}
                        </span>
                    </div>
                )}
            </div>

            {/* ── Content Area ── */}
            <div className="flex-1 p-4 bg-slate-950 relative overflow-hidden flex flex-col">
                <div
                    className="absolute inset-0 opacity-10 pointer-events-none"
                    style={{
                        backgroundImage: `linear-gradient(#10B981 1px, transparent 1px), linear-gradient(90deg, #10B981 1px, transparent 1px)`,
                        backgroundSize: '30px 30px',
                    }}
                />

                {!hasData && !isLoading ? (
                    <div className="flex-1 flex flex-col items-center justify-center space-y-3 py-10 opacity-70">
                        <AlertTriangle className="w-8 h-8 text-red-500/80" />
                        <p className="font-mono text-[9px] text-red-500 tracking-[0.2em] uppercase font-bold">
                            INSUFFICIENT GENETIC MARKERS
                        </p>
                    </div>
                ) : !hasData && isLoading ? (
                    <div className="flex-1 flex flex-col items-center justify-center space-y-3 py-10 opacity-60">
                        <div className="space-y-1 text-center">
                            <p className="font-mono text-[10px] text-cyan-400 tracking-[0.2em] uppercase animate-pulse">
                                ANALYZING PHENOTYPE...
                            </p>
                            <p className="font-mono text-[8px] text-zinc-500">
                                Constructing Forensic Profile
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="flex-1 flex flex-col gap-5 relative z-10 animate-in fade-in duration-500">
                        {/* Summary Header */}
                        <div className="grid grid-cols-2 gap-4 pb-4 border-b border-zinc-900">
                            <div>
                                <p className="font-mono text-[7px] text-zinc-500 uppercase tracking-widest mb-1">Subject_Reference</p>
                                <p className="font-mono text-xs font-bold text-white truncate">{profileId}</p>
                            </div>
                            <div className="text-right">
                                <p className="font-mono text-[7px] text-zinc-500 uppercase tracking-widest mb-1">Reliability_Index</p>
                                <p className={`font-mono text-xs font-bold ${reliabilityColor}`}>
                                    {reliabilityLabel}
                                </p>
                            </div>
                        </div>

                        {/* High-Fidelity Grid */}
                        <div className="grid grid-cols-1 gap-3">
                            {displayTraits.map((trait) => {
                                const highlight = isTraitRelevant(trait.key, trait.value);
                                return (
                                    <div
                                        key={trait.label}
                                        className={`relative group p-3 rounded bg-zinc-900/40 border transition-all duration-300 ${highlight
                                                ? "border-emerald-500/40 bg-emerald-500/5 shadow-[0_0_10px_rgba(16,185,129,0.1)]"
                                                : "border-zinc-800 hover:border-zinc-700"
                                            }`}
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <span className={`font-mono text-[8px] uppercase tracking-[0.15em] ${highlight ? 'text-emerald-400 font-bold' : 'text-zinc-500'}`}>
                                                {trait.label}
                                            </span>
                                            {highlight && (
                                                <div className="flex items-center gap-1">
                                                    <div className="w-1 h-1 bg-emerald-500 animate-pulse" />
                                                    <span className="font-mono text-[7px] text-emerald-500 max-[280px]:hidden">MATCH</span>
                                                </div>
                                            )}
                                        </div>
                                        <div className={`font-mono text-sm ${highlight ? 'text-white font-bold' : 'text-zinc-300'}`}>
                                            {trait.value}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>

                        {/* Footer Section with On-Chain Proof Button */}
                        <div className="mt-auto pt-4 border-t border-zinc-900 flex flex-col gap-3">
                            {txHash ? (
                                <a
                                    href={`https://sepolia.etherscan.io/tx/${txHash}`}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="group flex items-center justify-center gap-2 w-full bg-emerald-500/10 hover:bg-emerald-500/20 border border-emerald-500/30 hover:border-emerald-500/50 text-emerald-400 py-2.5 rounded transition-all active:scale-[0.98]"
                                >
                                    <CheckCircle2 className="w-3.5 h-3.5" />
                                    <span className="font-mono text-[9px] font-bold tracking-wider uppercase">
                                        View On-Chain Proof
                                    </span>
                                    <ChevronRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                                </a>
                            ) : (
                                <div className="flex items-center justify-center gap-2 w-full bg-zinc-900/50 border border-zinc-800 border-dashed text-zinc-600 py-2.5 rounded cursor-not-allowed">
                                    <AlertTriangle className="w-3.5 h-3.5" />
                                    <span className="font-mono text-[9px] tracking-wider uppercase">
                                        Proof Not Finalized
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                )}
            </div>
        </motion.div>
    );
}
