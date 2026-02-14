"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Plus, Trash2, Send, Dna, AlertTriangle, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";

import { generateUUID } from "@/lib/utils";
import { useIngestStore } from "@/store/ingestStore";

/* ── Standard CODIS markers for quick-add ── */
const CODIS_MARKERS = [
    "D3S1358", "TH01", "D21S11", "D18S51", "D5S818",
    "D13S317", "D7S820", "D16S539", "CSF1PO", "FGA",
    "TPOX", "VWA", "D8S1179", "D2S1338", "D19S433",
    "D1S1656", "D12S391", "SE33", "D2S441", "D10S1248",
    "D22S1045", "AMEL",
] as const;

interface LocusRow {
    id: string;
    marker: string;
    allele1: string;
    allele2: string;
}

function createEmptyRow(id?: string): LocusRow {
    return {
        id: id || generateUUID(),
        marker: "",
        allele1: "",
        allele2: "",
    };
}

export default function DNAIngestForm() {
    const router = useRouter();
    const setLastIngested = useIngestStore((s) => s.setLastIngested);

    const [nodeId, setNodeId] = useState("");
    const [rows, setRows] = useState<LocusRow[]>([
        createEmptyRow("initial-row-1"),
        createEmptyRow("initial-row-2"),
        createEmptyRow("initial-row-3"),
    ]);
    const [submitting, setSubmitting] = useState(false);
    const [result, setResult] = useState<string | null>(null);
    const [redirectCountdown, setRedirectCountdown] = useState<number | null>(null);

    const addRow = useCallback(() => {
        setRows((prev) => [...prev, createEmptyRow()]);
    }, []);

    const removeRow = useCallback((id: string) => {
        setRows((prev) => (prev.length > 1 ? prev.filter((r) => r.id !== id) : prev));
    }, []);

    const updateRow = useCallback(
        (id: string, field: keyof Omit<LocusRow, "id">, value: string) => {
            setRows((prev) =>
                prev.map((r) => (r.id === id ? { ...r, [field]: value } : r))
            );
        },
        []
    );

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setSubmitting(true);
        setResult(null);
        setRedirectCountdown(null);

        // Filter valid rows
        const validRows = rows.filter(r => r.marker && r.allele1 && r.allele2);

        // Build the payload
        const markers: Record<string, { allele_1: number; allele_2: number }> = {};
        for (const row of validRows) {
            markers[row.marker] = {
                allele_1: parseFloat(row.allele1),
                allele_2: parseFloat(row.allele2),
            };
        }

        const originNode = nodeId || "LOCAL-DEBUG";
        // Generate a standard UUID v4 to satisfy backend validation
        const tempProfileId = generateUUID();

        try {
            // Real API Call
            // Real API Call
            // Use environment variable or default to localhost:8000
            const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
            const res = await fetch(`${API_BASE}/profile/ingest`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    profile_id: tempProfileId,
                    node_id: originNode,
                    str_markers: markers,
                    qual_metrics: {
                        avg_peak_height: 1200, // Default good quality
                        stutter_ratio: 0.05
                    }
                })
            });

            if (!res.ok) {
                const errorData = await res.json();
                let errorMessage = "Ingest failed";

                if (errorData.detail) {
                    if (typeof errorData.detail === 'string') {
                        errorMessage = errorData.detail;
                    } else if (Array.isArray(errorData.detail)) {
                        // Handle Pydantic validation errors
                        errorMessage = errorData.detail.map((e: any) => e.msg).join(", ");
                    } else if (typeof errorData.detail === 'object') {
                        errorMessage = JSON.stringify(errorData.detail);
                    }
                }

                throw new Error(errorMessage);
            }

            const data = await res.json();

            // Update Global Store
            setLastIngested(data.profile_id, originNode, validRows.length);

            setResult(
                `✓ Profile accepted — ${validRows.length} markers validated from node "${originNode}" [${data.profile_id}]`
            );
            setSubmitting(false);
            setRedirectCountdown(3);

        } catch (err: any) {
            console.error("Ingest Error:", err);
            setResult(`❌ Ingest Failed: ${err.message || String(err)}`);
            setSubmitting(false);
        }
    };

    // Auto-redirect countdown after successful ingest
    useEffect(() => {
        if (redirectCountdown === null || redirectCountdown <= 0) return;

        const timer = setTimeout(() => {
            if (redirectCountdown <= 1) {
                router.push("/analysis");
            } else {
                setRedirectCountdown(redirectCountdown - 1);
            }
        }, 1000);

        return () => clearTimeout(timer);
    }, [redirectCountdown, router]);

    const filledCount = rows.filter((r) => r.marker && r.allele1 && r.allele2).length;

    return (
        <form onSubmit={handleSubmit} className="space-y-5">
            {/* ── Header ── */}
            <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Dna className="h-4 w-4 text-tactical-primary" />
                    <h2 className="font-data text-xs font-bold tracking-[0.15em] text-tactical-text uppercase">
                        STR Profile Ingest
                    </h2>
                </div>
                <span className="font-data text-[10px] text-tactical-text-dim">
                    {filledCount} / {rows.length} loci defined
                </span>
            </div>

            {/* ── Node ID ── */}
            <div className="space-y-1.5">
                <label className="font-data text-[10px] font-medium tracking-wider text-tactical-text-muted uppercase">
                    Originating Node ID
                </label>
                <input
                    type="text"
                    value={nodeId}
                    onChange={(e) => setNodeId(e.target.value)}
                    placeholder="e.g. INTERPOL-EU-DE"
                    className="
            w-full rounded-md border border-tactical-border bg-tactical-bg px-3 py-2
            font-data text-xs text-tactical-text placeholder:text-tactical-text-dim
            outline-none transition-colors
            focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20
          "
                />
            </div>

            {/* ── Locus Rows ── */}
            <div className="space-y-2">
                <div className="grid grid-cols-[1fr_0.6fr_0.6fr_2rem] gap-2 px-1">
                    <span className="font-data text-[9px] font-semibold tracking-widest text-tactical-text-dim uppercase">
                        Marker
                    </span>
                    <span className="font-data text-[9px] font-semibold tracking-widest text-tactical-text-dim uppercase">
                        Allele 1
                    </span>
                    <span className="font-data text-[9px] font-semibold tracking-widest text-tactical-text-dim uppercase">
                        Allele 2
                    </span>
                    <span />
                </div>

                {rows.map((row) => (
                    <div
                        key={row.id}
                        className="grid grid-cols-[1fr_0.6fr_0.6fr_2rem] items-center gap-2"
                    >
                        {/* Marker select / input */}
                        <div className="relative">
                            <input
                                type="text"
                                list={`markers-${row.id}`}
                                value={row.marker}
                                onChange={(e) => updateRow(row.id, "marker", e.target.value)}
                                placeholder="D3S1358"
                                className="
                  w-full rounded-md border border-tactical-border bg-tactical-bg px-3 py-2
                  font-data text-xs text-tactical-text placeholder:text-tactical-text-dim
                  outline-none transition-colors
                  focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20
                "
                            />
                            <datalist id={`markers-${row.id}`}>
                                {CODIS_MARKERS.map((m) => (
                                    <option key={m} value={m} />
                                ))}
                            </datalist>
                        </div>

                        {/* Allele 1 */}
                        <input
                            type="number"
                            step="0.1"
                            min="0"
                            value={row.allele1}
                            onChange={(e) => updateRow(row.id, "allele1", e.target.value)}
                            placeholder="12"
                            className="
                w-full rounded-md border border-tactical-border bg-tactical-bg px-3 py-2
                font-data text-xs text-tactical-text placeholder:text-tactical-text-dim
                outline-none transition-colors text-center
                focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20
              "
                        />

                        {/* Allele 2 */}
                        <input
                            type="number"
                            step="0.1"
                            min="0"
                            value={row.allele2}
                            onChange={(e) => updateRow(row.id, "allele2", e.target.value)}
                            placeholder="14"
                            className="
                w-full rounded-md border border-tactical-border bg-tactical-bg px-3 py-2
                font-data text-xs text-tactical-text placeholder:text-tactical-text-dim
                outline-none transition-colors text-center
                focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20
              "
                        />

                        {/* Remove row */}
                        <button
                            type="button"
                            onClick={() => removeRow(row.id)}
                            className="
                flex h-8 w-8 items-center justify-center rounded-md
                text-tactical-text-dim transition-colors
                hover:bg-tactical-danger/10 hover:text-tactical-danger
              "
                        >
                            <Trash2 className="h-3.5 w-3.5" />
                        </button>
                    </div>
                ))}

                {/* Add row button */}
                <button
                    type="button"
                    onClick={addRow}
                    className="
            flex w-full items-center justify-center gap-1.5 rounded-md
            border border-dashed border-tactical-border py-2
            font-data text-[10px] font-medium tracking-wider text-tactical-text-dim
            transition-colors uppercase
            hover:border-tactical-primary/30 hover:text-tactical-primary
          "
                >
                    <Plus className="h-3 w-3" />
                    Add Locus
                </button>
            </div>

            {/* ── Submit ── */}
            <button
                type="submit"
                disabled={submitting || filledCount === 0}
                className="
          flex w-full items-center justify-center gap-2 rounded-md
          bg-tactical-primary py-2.5
          font-data text-xs font-semibold tracking-widest text-tactical-bg uppercase
          transition-all duration-200
          hover:bg-tactical-primary-dim hover:glow-primary-strong
          disabled:cursor-not-allowed disabled:opacity-40
        "
            >
                {submitting ? (
                    <>
                        <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-tactical-bg border-t-transparent" />
                        Validating...
                    </>
                ) : (
                    <>
                        <Send className="h-3.5 w-3.5" />
                        Ingest Profile
                    </>
                )}
            </button>

            {/* ── Result feedback ── */}
            {result && (
                <motion.div
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="space-y-2"
                >
                    <div className="flex items-start gap-2 rounded-md border border-tactical-primary/20 bg-tactical-primary/5 px-3 py-2.5">
                        <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-tactical-primary" />
                        <span className="font-data text-[11px] leading-relaxed text-tactical-primary">
                            {result}
                        </span>
                    </div>

                    {/* Redirect indicator */}
                    {redirectCountdown !== null && redirectCountdown > 0 && (
                        <div className="flex items-center justify-center gap-2 rounded-md border border-cyan-500/20 bg-cyan-500/5 px-3 py-2">
                            <ArrowRight className="h-3 w-3 text-cyan-400 animate-pulse" />
                            <span className="font-data text-[10px] text-cyan-400">
                                Redirecting to Analysis in {redirectCountdown}s...
                            </span>
                        </div>
                    )}
                </motion.div>
            )}
        </form>
    );
}
