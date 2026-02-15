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
    txHash?: string;
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
    txHash,
}: GeoForensicPanelProps) {
    const hasData = geoResults && geoResults.length > 0;
    const [scanPhase, setScanPhase] = useState<ScanPhase>("idle");

    const handleScanPhaseChange = useCallback((phase: ScanPhase) => {
        setScanPhase(phase);
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="rounded-lg border border-tactical-border bg-tactical-surface overflow-hidden flex flex-col h-full"
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
                                ? `${geoResults.length} REGIONS · LOCKED`
                                : scanPhase === "idle"
                                    ? `${geoResults.length} regions`
                                    : "SCANNING..."
                            }
                        </span>
                    </div>
                )}
            </div>

            {!hasData ? (
                <EmptyState />
            ) : (
                <>
                    {/* ── Map Section — 50% ── */}
                    <div className="relative h-[50%] min-h-[200px] md:min-h-[250px] bg-[#070709]">
                        {/* Corner brackets */}
                        <div className="absolute top-2 left-2 w-4 h-4 border-l-2 border-t-2 border-tactical-primary/40 z-[1000] pointer-events-none" />
                        <div className="absolute top-2 right-2 w-4 h-4 border-r-2 border-t-2 border-tactical-primary/40 z-[1000] pointer-events-none" />
                        <div className="absolute bottom-2 left-2 w-4 h-4 border-l-2 border-b-2 border-tactical-primary/40 z-[1000] pointer-events-none" />
                        <div className="absolute bottom-2 right-2 w-4 h-4 border-r-2 border-b-2 border-tactical-primary/40 z-[1000] pointer-events-none" />

                        {/* Status bar with scan ticker */}
                        <div className="absolute top-2 left-1/2 -translate-x-1/2 z-[1000] pointer-events-none">
                            <ScanStatusTicker phase={scanPhase} region={geoResults[0]} />
                        </div>

                        <ForensicMap
                            data={geoResults}
                            onScanPhaseChange={handleScanPhaseChange}
                        />
                    </div>

                    {/* ── Data Panel — 50% ── */}
                    <div className="h-[50%] min-h-[240px] border-t border-tactical-border">
                        <AncestryDataPanel
                            data={geoResults}
                            reliabilityScore={reliabilityScore}
                            txHash={txHash}
                        />
                    </div>
                </>
            )}
        </motion.div>
    );
}
