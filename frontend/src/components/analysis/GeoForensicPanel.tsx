"use client";

import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Globe, Loader2, Crosshair, Radio } from "lucide-react";
import dynamic from "next/dynamic";
import AncestryDataPanel from "./AncestryDataPanel";
import type { ScanPhase } from "./ForensicMap";

// ═══════════════════════════════════════════════════════════════════════════════
// DYNAMIC IMPORT — Leaflet must be loaded client-side only (no SSR)
// ═══════════════════════════════════════════════════════════════════════════════

const ForensicMap = dynamic(() => import("./ForensicMap"), {
    ssr: false,
    loading: () => (
        <div className="w-full h-full flex items-center justify-center bg-tactical-surface">
            <div className="text-center space-y-2">
                <Loader2 className="w-6 h-6 text-tactical-primary animate-spin mx-auto" />
                <p className="font-mono text-[9px] text-zinc-500 tracking-[0.2em] uppercase">
                    Initializing GIS Engine
                </p>
            </div>
        </div>
    ),
});

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

interface GeoProbability {
    region: string;
    lat: number;
    lng: number;
    probability: number;
    color: string;
    initial_radius_km?: number;
    final_radius_km?: number;
}

interface GeoForensicPanelProps {
    geoResults: GeoProbability[] | null;
    reliabilityScore: number;
    kinshipMatches?: any[];
    txHash?: string;
    isLoading?: boolean;
    selectedRegion?: string | null;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCANNING LOADER (New Component)
// ═══════════════════════════════════════════════════════════════════════════════

function ScanningMapLoader() {
    return (
        <div className="flex-1 flex flex-col items-center justify-center bg-[#070709] relative overflow-hidden min-h-[400px]">
            {/* Grid Background */}
            <div
                className="absolute inset-0 z-0 opacity-10 pointer-events-none"
                style={{
                    backgroundImage: `linear-gradient(#334155 1px, transparent 1px), linear-gradient(90deg, #334155 1px, transparent 1px)`,
                    backgroundSize: '40px 40px'
                }}
            />

            {/* Scanning Radar */}
            <div className="relative z-10 w-48 h-48 sm:w-64 sm:h-64 rounded-full flex items-center justify-center">
                {/* Rings */}
                <div className="absolute inset-0 rounded-full border border-zinc-800/60" />
                <div className="absolute inset-8 rounded-full border border-zinc-800/40" />
                <div className="absolute inset-16 rounded-full border border-zinc-800/20" />

                {/* Pulsing Ring */}
                <motion.div
                    className="absolute inset-0 rounded-full border border-tactical-primary/30"
                    animate={{ scale: [0.9, 1.1, 0.9], opacity: [0.3, 0.1, 0.3] }}
                    transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
                />

                {/* Expanding Echo */}
                <motion.div
                    className="absolute inset-0 rounded-full border border-tactical-primary/20"
                    animate={{ scale: [1, 1.4], opacity: [0.4, 0] }}
                    transition={{ duration: 2, repeat: Infinity, ease: "easeOut" }}
                />

                {/* Radar Sweep Line */}
                <motion.div
                    className="absolute inset-0 rounded-full"
                    style={{
                        background: `conic-gradient(from 0deg, transparent 0deg, transparent 270deg, rgba(34, 197, 94, 0.1) 360deg)`
                    }}
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2.5, repeat: Infinity, ease: "linear" }}
                />

                {/* Center Icon */}
                <div className="relative z-20 bg-[#070709] p-3 rounded-full border border-tactical-primary/20">
                    <Globe className="w-8 h-8 text-tactical-primary opacity-80" />
                </div>
            </div>

            {/* Value Tickers */}
            <div className="absolute top-1/2 left-8 -translate-y-1/2 hidden md:block space-y-1">
                {[1, 2, 3].map((i) => (
                    <motion.div
                        key={i}
                        className="flex items-center gap-2"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.2 }}
                    >
                        <div className="w-1 h-1 bg-tactical-primary/50" />
                        <span className="font-mono text-[7px] text-zinc-600">
                            COORD_X: {(Math.random() * 100).toFixed(4)}
                        </span>
                    </motion.div>
                ))}
            </div>

            {/* Text Feedback */}
            <div className="mt-8 text-center space-y-2 z-10">
                <div className="flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 text-tactical-primary animate-spin" />
                    <p className="font-mono text-[10px] font-bold text-tactical-primary tracking-[0.2em] uppercase">
                        Scanning Geo-Spatial Index
                    </p>
                </div>
                <p className="font-mono text-[8px] text-zinc-500 tracking-wider">
                    Calculating Allele Frequencies...
                </p>
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// SCAN STATUS TICKER
// ═══════════════════════════════════════════════════════════════════════════════

const SCAN_MESSAGES: Record<ScanPhase, { text: string; color: string }> = {
    idle: { text: "Awaiting Analysis Data...", color: "text-zinc-600" },
    scanning: { text: "Scanning Global Databases...", color: "text-blue-400" },
    calculating: { text: "Calculating Population Covariance...", color: "text-amber-400" },
    locked: { text: "95% Confidence Zone Locked.", color: "text-tactical-primary" },
};

function ScanStatusTicker({ phase, region }: { phase: ScanPhase; region?: GeoProbability }) {
    const msg = SCAN_MESSAGES[phase];
    const isLocked = phase === "locked";

    return (
        <div className="flex items-center gap-2 px-3 py-1 rounded bg-black/70 backdrop-blur-sm border border-tactical-border/50">
            {isLocked ? (
                <Crosshair className="w-3 h-3 text-tactical-primary" />
            ) : phase === "idle" ? (
                <Radio className="w-3 h-3 text-zinc-600" />
            ) : (
                <Loader2 className="w-3 h-3 animate-spin text-blue-400" />
            )}
            <span className={`font-mono text-[8px] tracking-wider uppercase ${msg.color}`}>
                {msg.text}
            </span>
            {isLocked && region && (
                <span className="font-mono text-[8px] font-bold text-tactical-primary">
                    — {region.region} ({(region.probability * 100).toFixed(1)}%)
                </span>
            )}
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// EMPTY STATE
// ═══════════════════════════════════════════════════════════════════════════════

function EmptyState() {
    return (
        <div className="flex-1 flex items-center justify-center">
            <div className="text-center space-y-3 opacity-40">
                <Globe className="w-12 h-12 text-zinc-600 mx-auto" />
                <p className="font-mono text-[9px] text-zinc-500 uppercase tracking-widest">
                    NO GEO-FORENSIC DATA
                    <br />
                    Run analysis to generate ancestry map
                </p>
            </div>
        </div>
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// MAIN COMPONENT
// ═══════════════════════════════════════════════════════════════════════════════

export default function GeoForensicPanel({
    geoResults,
    reliabilityScore,
    kinshipMatches,
    txHash,
    isLoading = false,
    onRegionHover,
    selectedRegion,
}: GeoForensicPanelProps & { onRegionHover?: (region: string | null) => void }) {
    // ── 1. Unified Data Source & Sorting ──
    // Sort by probability descending to align Map and List
    const sortedData = geoResults
        ? [...geoResults].sort((a, b) => b.probability - a.probability).filter(d => d.probability > 0)
        : [];

    // ── 2. Dynamic Reliability Calculation ──
    // Score = (Top_Prob - Second_Prob) * Multiplier
    // If top two are close, reliability is low.
    let calculatedReliability = reliabilityScore;
    if (sortedData.length >= 2) {
        const margin = sortedData[0].probability - sortedData[1].probability;
        // Multiplier 2.5 means a 40% margin gives 100% confidence
        calculatedReliability = Math.min(Math.max(margin * 2.5, 0.1), 0.99);
    } else if (sortedData.length === 1) {
        calculatedReliability = 0.95; // Single match is high confidence
    }

    const hasData = sortedData.length > 0;
    const [scanPhase, setScanPhase] = useState<ScanPhase>("idle");

    // Reset scan phase when data changes (new profile)
    // using a ref or effect to detect data change if needed, 
    // but ForensicMap handles flyTo on data change.

    const handleScanPhaseChange = useCallback((phase: ScanPhase) => {
        setScanPhase(phase);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-lg border border-tactical-border bg-tactical-surface overflow-hidden flex flex-col h-fit"
        >
            {/* ── Header ── */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-tactical-border shrink-0">
                <div className="flex items-center gap-2">
                    <Globe className="w-4 h-4 text-tactical-primary" />
                    <h3 className="font-mono text-[10px] font-bold tracking-[0.2em] text-tactical-text uppercase">
                        Geo-Forensic_Intelligence
                    </h3>
                </div>
                {hasData && (
                    <div className="flex items-center gap-1.5">
                        <div className={`w-1.5 h-1.5 rounded-full ${scanPhase === "locked"
                            ? "bg-tactical-primary"
                            : scanPhase === "idle"
                                ? "bg-zinc-600"
                                : "bg-blue-400 animate-pulse"
                            }`} />
                        <span className="font-mono text-[8px] text-zinc-500 tracking-wider uppercase">
                            {scanPhase === "locked"
                                ? `${sortedData.length} REGIONS · LOCKED`
                                : scanPhase === "idle"
                                    ? `${sortedData.length} regions`
                                    : "SCANNING..."
                            }
                        </span>
                    </div>
                )}
            </div>

            {!hasData && !isLoading ? (
                <EmptyState />
            ) : isLoading ? (
                <ScanningMapLoader />
            ) : (
                <>
                    {/* ── Map Section — Fixed Height for Stability ── */}
                    <div className="relative h-[300px] md:h-[350px] bg-[#070709]">
                        {/* Corner brackets */}
                        <div className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-tactical-primary/40 z-[1000] pointer-events-none" />
                        <div className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-tactical-primary/40 z-[1000] pointer-events-none" />
                        <div className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-tactical-primary/40 z-[1000] pointer-events-none" />
                        <div className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-tactical-primary/40 z-[1000] pointer-events-none" />

                        {/* Status bar with scan ticker */}
                        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-[1000] pointer-events-none">
                            <ScanStatusTicker phase={scanPhase} region={sortedData[0]} />
                        </div>

                        <ForensicMap
                            data={sortedData}
                            kinshipMatches={kinshipMatches}
                            onScanPhaseChange={handleScanPhaseChange}
                            onRegionHover={onRegionHover}
                        />
                    </div>

                    {/* ── Data Panel — Auto Height ── */}
                    <div className="h-fit border-t border-tactical-border">
                        <AncestryDataPanel
                            data={sortedData}
                            reliabilityScore={calculatedReliability}
                            txHash={txHash}
                            selectedRegion={selectedRegion}
                        />
                    </div>
                </>
            )}
        </motion.div>
    );
}
