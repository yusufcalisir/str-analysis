"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldAlert, Lock, Search, Upload, CheckCircle, Brain, RefreshCw } from "lucide-react";
import { formatDistanceToNow } from "date-fns";
import { useIngestStore } from "@/store/ingestStore";

import SystemPulse from "@/components/investigation/SystemPulse";
import CryptographicShield from "@/components/investigation/CryptographicShield";
import EmbeddedAuditLog from "@/components/investigation/EmbeddedAuditLog";
import InvestigatorSidebar from "@/components/investigation/InvestigatorSidebar";
import { MatchResultCardDemo } from "@/components/analysis/MatchResultCard";
import GeoForensicPanel from "@/components/analysis/GeoForensicPanel";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

async function fetchAnalysis(profileId: string, population: string) {
    const res = await fetch(`${API_BASE}/profile/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile_id: profileId, population }),
    });
    if (!res.ok) throw new Error(`Analysis failed: ${res.status}`);
    return res.json();
}

export default function InvestigationDashboard() {
    // Session State
    const [panicMode, setPanicMode] = useState(false);

    // Analysis State
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [shieldActive, setShieldActive] = useState(false);
    const [zkpStatus, setZkpStatus] = useState<'idle' | 'generating' | 'verified' | 'failed'>('idle');
    const [analysisResult, setAnalysisResult] = useState<any | null>(null);
    const [activeProfileId, setActiveProfileId] = useState("test-profile-eu");

    const { setLastIngested } = useIngestStore();

    // ─── HANDLERS ─────────────────────────────────────────────────────────────

    const handlePanic = () => {
        setPanicMode(true);
        // In a real app, this would call /auth/logout and purge local storage
        console.log("PANIC: Session Revoked, Token Purged, WebSocket Closed.");
    };

    const runInvestigation = useCallback(async () => {
        if (isAnalyzing) return;
        setIsAnalyzing(true);
        setShieldActive(true); // Trigger ZKP Animation
        setZkpStatus('generating');

        try {
            // 1. Trigger ZKP Worker (Mocked for Demo if artifacts missing)
            // Real implementation would look like AnalysisPage.tsx worker logic
            // For dashboard flow, we simulate the cryptographic delay

            await new Promise(resolve => setTimeout(resolve, 2500));

            setZkpStatus('verified');
            setShieldActive(false); // Hide shield once verified

            // 2. Fetch Analysis Data
            const data = await fetchAnalysis(activeProfileId, "European");
            setAnalysisResult(data);

            // 3. Update Global Store
            setLastIngested(activeProfileId, "VANTAGE-NODE-01", 24);

        } catch (error) {
            console.error("Investigation Failed:", error);
            setZkpStatus('failed');
            setShieldActive(false);
        } finally {
            setIsAnalyzing(false);
        }
    }, [isAnalyzing, activeProfileId, setLastIngested]);

    const resetInvestigation = () => {
        setAnalysisResult(null);
        setZkpStatus('idle');
    };

    // ─── RENDER ───────────────────────────────────────────────────────────────

    if (panicMode) {
        return (
            <div className="flex flex-col items-center justify-center h-[80vh] text-red-500 space-y-4 bg-zinc-950">
                <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    className="p-6 bg-red-500/10 rounded-full border border-red-500/20"
                >
                    <Lock className="w-16 h-16 animate-pulse" />
                </motion.div>
                <h1 className="text-2xl font-bold tracking-[0.2em] uppercase">Session Terminated</h1>
                <div className="flex flex-col items-center space-y-1 text-zinc-500 font-mono text-sm">
                    <p>Blockchain Access Token Revoked</p>
                    <p>Local Key Material Shredded</p>
                    <p>Audit Log: <span className="text-red-400">EMERGENCY_EXIT_0x9F2A</span></p>
                </div>
            </div>
        );
    }

    return (
        <div className="flex h-[calc(100vh-4rem)] overflow-hidden bg-zinc-950">
            {/* Main Content Area */}
            <div className="flex-1 flex flex-col p-6 overflow-y-auto space-y-6 relative">

                {/* ZKP Shield Overlay */}
                <AnimatePresence>
                    {shieldActive && <CryptographicShield active={shieldActive} />}
                </AnimatePresence>

                {/* Header & Panic */}
                <div className="flex justify-between items-center mb-2">
                    <div className="flex items-center gap-3">
                        <div className="w-1 h-8 bg-tactical-primary rounded-full" />
                        <div>
                            <h1 className="text-xl font-bold tracking-[0.2em] text-tactical-text uppercase">
                                Investigation Command Center
                            </h1>
                            <p className="text-[10px] text-zinc-500 font-mono tracking-wider uppercase">
                                VANTAGE-STR // LEVEL 4 CLEARANCE
                            </p>
                        </div>
                    </div>
                    <button
                        onClick={handlePanic}
                        className="group flex items-center gap-2 px-4 py-2 bg-red-950/30 hover:bg-red-900/40 text-red-500 border border-red-900/50 rounded transition-all font-mono text-xs font-bold uppercase tracking-wider"
                    >
                        <ShieldAlert className="w-4 h-4 group-hover:animate-pulse" />
                        Panic: Revoke Access
                    </button>
                </div>

                {/* Module 1: System Pulse */}
                <SystemPulse />

                {/* Module 2: The Forensic Vault (Search vs Result) */}
                <div className="flex-1 min-h-[500px] flex flex-col relative w-full">
                    <AnimatePresence mode="wait">
                        {!analysisResult ? (
                            <motion.div
                                key="search-mode"
                                initial={{ opacity: 0, y: 20 }}
                                animate={{ opacity: 1, y: 0 }}
                                exit={{ opacity: 0, y: -20 }}
                                className="flex-1 bg-zinc-900/20 border border-zinc-800 rounded-lg p-8 flex flex-col items-center justify-center relative overflow-hidden group"
                            >
                                <div className="absolute inset-0 bg-grid-zinc-900/50 [mask-image:linear-gradient(0deg,white,rgba(255,255,255,0.6))] pointer-events-none" />

                                <div className="relative z-10 text-center space-y-8 max-w-lg">
                                    <div className="mx-auto w-20 h-20 bg-zinc-800 rounded-full flex items-center justify-center border border-zinc-700 shadow-xl group-hover:border-emerald-500/50 transition-colors">
                                        <Search className="w-10 h-10 text-zinc-500 group-hover:text-emerald-400 transition-colors" />
                                    </div>

                                    <div className="space-y-2">
                                        <h2 className="text-3xl font-bold text-zinc-200 tracking-tight">Forensic Vault Search</h2>
                                        <p className="text-zinc-500 leading-relaxed">
                                            Upload raw `.fsa` files or enter STR profile manually.
                                            <span className="text-emerald-500 block mt-2 font-mono text-xs border border-emerald-500/20 bg-emerald-500/5 py-1 px-2 rounded inline-block">
                                                <Lock className="w-3 h-3 inline mr-1 mb-0.5" />
                                                Zero-Knowledge Proof Enabled
                                            </span>
                                        </p>
                                    </div>

                                    <div className="flex gap-4 justify-center">
                                        <button
                                            onClick={runInvestigation}
                                            disabled={isAnalyzing}
                                            className="px-8 py-3 bg-zinc-100 hover:bg-white text-zinc-900 rounded font-bold transition-all flex items-center gap-2 shadow-[0_0_20px_rgba(255,255,255,0.1)] hover:shadow-[0_0_25px_rgba(255,255,255,0.2)]"
                                        >
                                            {isAnalyzing ? (
                                                <>
                                                    <RefreshCw className="w-4 h-4 animate-spin" />
                                                    Processing Cryptography...
                                                </>
                                            ) : (
                                                <>
                                                    <Upload className="w-4 h-4" />
                                                    Upload DNA Profile
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>
                            </motion.div>
                        ) : (
                            <motion.div
                                key="result-mode"
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="flex-1 flex flex-col md:flex-row gap-6 h-full"
                            >
                                {/* Left: Match Stats */}
                                <div className="flex-1 space-y-6">
                                    <div className="flex justify-between items-center bg-zinc-900/40 p-3 rounded border border-zinc-800">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 rounded bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
                                                <CheckCircle className="w-5 h-5 text-emerald-400" />
                                            </div>
                                            <div>
                                                <h3 className="text-sm font-bold text-zinc-200">Match Verified</h3>
                                                <p className="text-[10px] text-zinc-500 font-mono">ZKP Hash: 0x9a7...3b2</p>
                                            </div>
                                        </div>
                                        <button
                                            onClick={resetInvestigation}
                                            className="text-xs text-zinc-500 hover:text-zinc-300 font-mono underline"
                                        >
                                            New Search
                                        </button>
                                    </div>

                                    <div className="grid grid-cols-1 gap-6">
                                        <MatchResultCardDemo />
                                    </div>
                                </div>

                                {/* Right: Geo-Forensic Intelligence */}
                                <div className="w-full md:flex-1 h-[720px] min-h-[500px]">
                                    <GeoForensicPanel
                                        geoResults={analysisResult?.geo_analysis_results || null}
                                        reliabilityScore={analysisResult?.geo_reliability_score || 0}
                                    />
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>

                {/* Module 3: The Live Ledger */}
                <div className="h-64 shrink-0">
                    <EmbeddedAuditLog />
                </div>
            </div>

            {/* Sidebar: Agentic Intelligence */}
            <div className="w-96 border-l border-zinc-800 bg-zinc-950/30 hidden xl:block backdrop-blur-sm">
                <InvestigatorSidebar />
            </div>
        </div>
    );
}
