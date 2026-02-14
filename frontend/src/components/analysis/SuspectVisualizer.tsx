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

export default function SuspectVisualizer({ profileId }: { profileId?: string }) {
    const [data, setData] = useState<ReconstructionData | null>(null);
    const [isGenerating, setIsGenerating] = useState(false);
    const [overlayActive, setOverlayActive] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [isImageLoading, setIsImageLoading] = useState(false);
    const [imageLoadError, setImageLoadError] = useState(false);
    const containerRef = useRef<HTMLDivElement>(null);

    const handleGenerate = useCallback(async () => {
        setIsGenerating(true);
        setError(null);

        const targetProfileId = profileId || "test-profile-eu";

        try {
            // Attempt real API call to the phenotype endpoint (which now includes GenAI)
            const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(
                `${API_BASE}/profile/phenotype/${targetProfileId}`
            );
            if (res.ok) {
                const result = await res.json();

                // Map PhenotypeReport to ReconstructionData
                const mappedData: ReconstructionData = {
                    profile_id: result.profile_id,
                    image_url: result.image_url,
                    seed: result.seed,
                    prompt_hash: "N/A", // Not in PhenotypeReport yet
                    generation_time_ms: 0, // Not in PhenotypeReport yet
                    model_id: result.genai_model_id || "mock-sdxl-dev",
                    trait_summary: result.trait_summary || {},
                    positive_prompt: result.positive_prompt || "",
                    negative_prompt: result.negative_prompt || "",
                };

                // Simulate processing time for UX if it was too fast (cached)
                await new Promise((r) => setTimeout(r, 1000));

                if (mappedData.image_url) {
                    setIsImageLoading(true);
                    setImageLoadError(false);
                    console.log(`[UI_RENDER] Attempting to load suspect image from: ${mappedData.image_url}`);
                }

                setData(mappedData);
            } else {
                console.warn("API Error - Visualizer awaiting valid profile.");
                setData(null);
            }
        } catch (e) {
            console.error("Fetch Error:", e);
            setError("BACKEND CONNECTION FAILED");
            setData(null);
        } finally {
            setIsGenerating(false);
        }
    }, [profileId]);

    // Auto-generate on mount or profileId change
    useEffect(() => {
        handleGenerate();
    }, [handleGenerate]);

    const handleDownloadPoster = useCallback(async () => {
        if (!data || !containerRef.current) return;

        try {
            // Temporarily hide the download button and other UI controls for the capture
            const controls = containerRef.current.querySelectorAll("button, .capture-hide");
            controls.forEach((el) => ((el as HTMLElement).style.opacity = "0"));

            const dataUrl = await toPng(containerRef.current, {
                cacheBust: true,
                backgroundColor: "#0A0A0B", // Match --color-tactical-bg
                pixelRatio: 2,
                style: {
                    borderRadius: "0", // Clean capture
                }
            });

            // Restore visibility
            controls.forEach((el) => ((el as HTMLElement).style.opacity = "1"));

            const a = document.createElement("a");
            a.href = dataUrl;
            a.download = `VANTAGE_FORENSIC_POSTER_${data.profile_id}.png`;
            a.click();
        } catch (err) {
            console.error("Poster generation failed:", err);
            // Re-show controls on error just in case
            if (containerRef.current) {
                const controls = containerRef.current.querySelectorAll("button, .capture-hide");
                controls.forEach((el) => ((el as HTMLElement).style.opacity = "1"));
            }
        }
    }, [data]);

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-lg border border-tactical-border bg-tactical-surface overflow-hidden"
            ref={containerRef}
        >
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-tactical-border">
                <div className="flex items-center gap-2">
                    <ScanFace className="w-4 h-4 text-tactical-primary" />
                    <h3 className="font-mono text-[10px] font-bold tracking-[0.2em] text-tactical-text uppercase">
                        Suspect_Visual_Reconstruction
                    </h3>
                </div>
                <div className="flex items-center gap-2">
                    {data && (
                        <button
                            onClick={() => setOverlayActive(!overlayActive)}
                            className="font-mono text-[8px] text-zinc-600 hover:text-tactical-primary transition-colors px-2 py-1 rounded border border-tactical-border/50 hover:border-tactical-primary/30"
                        >
                            {overlayActive ? "OVERLAY:ON" : "OVERLAY:OFF"}
                        </button>
                    )}
                    <button
                        onClick={handleGenerate}
                        disabled={isGenerating}
                        className="font-mono text-[8px] text-zinc-600 hover:text-tactical-primary transition-colors px-2 py-1 rounded border border-tactical-border/50 hover:border-tactical-primary/30 disabled:opacity-30"
                    >
                        <RefreshCw className={`w-3 h-3 ${isGenerating ? "animate-spin" : ""}`} />
                    </button>
                </div>
            </div>

            {/* ── Image Panel ── */}
            <div className="relative aspect-[3/4] bg-[#070709] overflow-hidden">
                {/* Background gradient */}
                <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/60 z-10" />

                {/* Forensic placeholder or real image */}
                {/* Forensic placeholder or real image */}
                {data?.image_url && !imageLoadError ? (
                    <>
                        <img
                            src={data.image_url}
                            alt="Forensic facial reconstruction"
                            className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ${isImageLoading ? "opacity-0" : "opacity-100"
                                }`}
                            onLoad={() => setIsImageLoading(false)}
                            onError={() => {
                                console.error("[UI_RENDER] Image failed to load:", data.image_url);
                                setImageLoadError(true);
                                setIsImageLoading(false);
                            }}
                        />
                        {isImageLoading && (
                            <div className="absolute inset-0 flex items-center justify-center bg-black/80 z-20">
                                <div className="text-cyan-500 font-mono text-xs animate-pulse tracking-widest">LOADING VISUAL...</div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className="relative w-full h-full flex items-center justify-center">
                        {!data ? (
                            <div className="text-center space-y-2 opacity-40">
                                <ScanFace className="w-12 h-12 text-zinc-600 mx-auto" />
                                <p className="font-mono text-[9px] text-zinc-500 uppercase tracking-widest">
                                    SYSTEM STANDBY<br />AWAITING DNA INPUT
                                </p>
                            </div>
                        ) : (
                            <>
                                <ForensicPlaceholder />
                                {imageLoadError && (
                                    <div className="absolute bottom-4 left-0 right-0 text-center">
                                        <span className="bg-red-900/80 text-white text-[10px] px-2 py-1 rounded border border-red-500/50">
                                            IMAGE LOAD FAILED - CHECK URL
                                        </span>
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                )}

                {/* Overlay grid */}
                <ForensicOverlay active={overlayActive} />

                {/* Scanning beam */}
                <ScanBeam active={isGenerating} />

                {/* Bottom status bar */}
                <div className="absolute bottom-0 left-0 right-0 z-20 px-3 py-2 bg-gradient-to-t from-black/90 to-transparent">
                    <div className="flex items-center justify-between">
                        <span className="font-mono text-[8px] text-zinc-500">
                            {data ? `SEED:${data.seed} • ${data.prompt_hash}` : "NO_DATA"}
                        </span>
                        {data && (
                            <span className="font-mono text-[8px] text-tactical-primary">
                                {data.generation_time_ms}ms
                            </span>
                        )}
                    </div>
                </div>

                {/* Generating overlay */}
                <AnimatePresence>
                    {isGenerating && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 z-30 flex items-center justify-center bg-black/70 backdrop-blur-sm"
                        >
                            <div className="text-center space-y-3">
                                <motion.div
                                    animate={{ rotate: 360 }}
                                    transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                >
                                    <ScanFace className="w-8 h-8 text-tactical-primary mx-auto" />
                                </motion.div>
                                <p className="font-mono text-[9px] text-tactical-primary tracking-[0.3em] uppercase">
                                    Reconstructing
                                </p>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>

            {/* ── Trait Panel ── */}
            <div className="px-4 py-3 space-y-3 border-t border-tactical-border">
                {/* Generation Log */}
                <GenerationLog isGenerating={isGenerating} />

                {/* Trait Summary */}
                {data && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 }}
                        className="space-y-1.5"
                    >
                        <div className="flex items-center gap-1.5 mb-2">
                            <div className="w-0.5 h-3 rounded-full bg-tactical-primary" />
                            <span className="font-mono text-[9px] font-bold text-tactical-text tracking-[0.15em] uppercase">
                                Predicted_Traits
                            </span>
                        </div>

                        {Object.entries(data.trait_summary || {}).map(([trait, value], idx) => {
                            const Icon = TRAIT_ICONS[trait] || User;
                            return (
                                <motion.div
                                    key={trait}
                                    initial={{ opacity: 0, x: -6 }}
                                    animate={{ opacity: 1, x: 0 }}
                                    transition={{ delay: 0.4 + idx * 0.08 }}
                                    className="flex items-center justify-between py-1 px-2 rounded bg-tactical-surface-elevated/50 group hover:bg-tactical-primary/5 transition-colors"
                                >
                                    <div className="flex items-center gap-2">
                                        <Icon className="w-3 h-3 text-zinc-600 group-hover:text-tactical-primary transition-colors" />
                                        <span className="font-mono text-[9px] text-zinc-500 tracking-wide">
                                            {trait}
                                        </span>
                                    </div>
                                    <div className="flex items-center gap-1">
                                        <span className="font-mono text-[9px] text-tactical-text font-semibold">
                                            {value}
                                        </span>
                                        <ChevronRight className="w-2.5 h-2.5 text-zinc-700" />
                                    </div>
                                </motion.div>
                            );
                        })}
                    </motion.div>
                )}

                {/* Model Info */}
                {data && (
                    <div className="flex items-center justify-between pt-2 border-t border-tactical-border/50">
                        <span className="font-mono text-[8px] text-zinc-600">
                            MODEL: {data.model_id || "N/A"}
                        </span>
                        <span className="font-mono text-[8px] text-zinc-600">
                            PROFILE: {data.profile_id}
                        </span>
                    </div>
                )}

                {/* Error state */}
                {error && (
                    <div className="flex items-center gap-2 px-3 py-2 rounded bg-red-500/10 border border-red-500/20">
                        <AlertTriangle className="w-3 h-3 text-red-400" />
                        <span className="font-mono text-[9px] text-red-400">{error}</span>
                    </div>
                )}

                {/* Download Button */}
                {data && (
                    <motion.button
                        initial={{ opacity: 0, y: 6 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.8 }}
                        onClick={handleDownloadPoster}
                        className="w-full flex items-center justify-center gap-2 py-2.5 rounded-md
                            bg-tactical-primary/10 border border-tactical-primary/20
                            hover:bg-tactical-primary/20 hover:border-tactical-primary/40
                            transition-all duration-200 group"
                    >
                        <Download className="w-3.5 h-3.5 text-tactical-primary group-hover:scale-110 transition-transform" />
                        <span className="font-mono text-[9px] text-tactical-primary font-bold tracking-[0.2em] uppercase">
                            Download Forensic Poster
                        </span>
                    </motion.button>
                )}
            </div>
        </motion.div>
    );
}
