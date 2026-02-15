"use client";

import { useState, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Plus, Trash2, Send, Dna, AlertTriangle, ArrowRight } from "lucide-react";
import { motion } from "framer-motion";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";


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

interface DNAIngestFormProps {
    selectedNodeId?: string;
    onNodeChange?: (id: string) => void;
}

export default function DNAIngestForm({ selectedNodeId, onNodeChange }: DNAIngestFormProps) {
    const router = useRouter();
    const setLastIngested = useIngestStore((s) => s.setLastIngested);

    const [nodeId, setNodeId] = useState("");

    // Sync with external selection
    useEffect(() => {
        if (selectedNodeId !== undefined) {
            setNodeId(selectedNodeId);
        }
    }, [selectedNodeId]);

    const handleNodeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const newValue = e.target.value;
        setNodeId(newValue);
        if (onNodeChange) {
            onNodeChange(newValue);
        }
    };
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



    // ── SNP State ──
    const [showSnps, setShowSnps] = useState(false);
    const [snpRows, setSnpRows] = useState<{ id: string; rsid: string; genotype: string }[]>([
        { id: "snp-1", rsid: "", genotype: "" },
        { id: "snp-2", rsid: "", genotype: "" },
    ]);

    const addSnpRow = () => setSnpRows(prev => [...prev, { id: generateUUID(), rsid: "", genotype: "" }]);
    const removeSnpRow = (id: string) => setSnpRows(prev => prev.filter(r => r.id !== id));
    const updateSnpRow = (id: string, field: "rsid" | "genotype", value: string) => {
        setSnpRows(prev => prev.map(r => r.id === id ? { ...r, [field]: value } : r));
    };

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

        // Build SNP payload
        const snpPayload: Record<string, string> = {};
        for (const row of snpRows) {
            if (row.rsid && row.genotype) {
                snpPayload[row.rsid] = row.genotype;
            }
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
                    },
                    snp_data: Object.keys(snpPayload).length > 0 ? snpPayload : undefined,
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
    const filledSnpCount = snpRows.filter((r) => r.rsid && r.genotype).length;

    // Relaxed validation: At least 1 valid STR OR 1 valid SNP
    const isFormValid = filledCount >= 1 || filledSnpCount >= 1;

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
                    {filledCount} STRs / {filledSnpCount} SNPs
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
                    onChange={handleNodeChange}
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



            {/* ── Advanced: Phenotype SNPs ── */}
            <div className="pt-2 border-t border-tactical-border/50">
                <button
                    type="button"
                    onClick={() => setShowSnps(!showSnps)}
                    className="flex items-center gap-2 text-[10px] font-data font-bold tracking-widest text-tactical-text-dim uppercase hover:text-tactical-primary transition-colors"
                >
                    <div className={`transition-transform duration-200 ${showSnps ? "rotate-90" : ""}`}>▶</div>
                    Advanced: Phenotype Markers (SNPs)
                </button>

                {showSnps && (
                    <motion.div
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        className="space-y-3 pt-3"
                    >
                        <div className="grid grid-cols-[1fr_1fr_2rem] gap-2 px-1">
                            <span className="font-data text-[9px] font-semibold tracking-widest text-tactical-text-dim uppercase">
                                rsID
                            </span>
                            <span className="font-data text-[9px] font-semibold tracking-widest text-tactical-text-dim uppercase">
                                Genotype
                            </span>
                            <span />
                        </div>

                        {snpRows.map((row) => (
                            <div key={row.id} className="grid grid-cols-[1fr_1fr_2rem] items-center gap-2">
                                {/* rsID Input */}
                                <input
                                    type="text"
                                    list="common-rsids"
                                    value={row.rsid}
                                    onChange={(e) => updateSnpRow(row.id, "rsid", e.target.value)}
                                    placeholder="rs12913832"
                                    className="w-full rounded-md border border-tactical-border bg-tactical-bg px-3 py-2 font-data text-xs text-tactical-text placeholder:text-tactical-text-dim outline-none transition-colors focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20"
                                />
                                <datalist id="common-rsids">
                                    {/* Eye Color */}
                                    <option value="rs12913832" label="HERC2 (Blue/Brown Eye)" />
                                    <option value="rs16891982" label="SLC45A2 (Dark/Light)" />
                                    <option value="rs1800407" label="OCA2 (Green/Hazel)" />
                                    <option value="rs12896399" label="SLC24A4" />
                                    <option value="rs12203592" label="IRF4" />
                                    <option value="rs1393350" label="TYR" />

                                    {/* Hair Color */}
                                    <option value="rs1805007" label="MC1R (Red Hair)" />
                                    <option value="rs1805008" label="MC1R (Red Hair)" />
                                    <option value="rs1805009" label="MC1R (Red Hair)" />
                                    <option value="rs11547464" label="MC1R (Red Hair)" />
                                    <option value="rs1805006" label="MC1R (Red Hair)" />

                                    {/* Skin Color */}
                                    <option value="rs1426654" label="SLC24A5 (Light Skin)" />
                                    <option value="rs1042602" label="TYR (Skin)" />
                                    <option value="rs6119471" label="ASIP (Dark Skin)" />
                                </datalist>

                                {/* Genotype Input */}
                                <input
                                    type="text"
                                    list="common-genotypes"
                                    maxLength={2}
                                    value={row.genotype}
                                    onChange={(e) => updateSnpRow(row.id, "genotype", e.target.value.toUpperCase())}
                                    placeholder="GG"
                                    className="w-full rounded-md border border-tactical-border bg-tactical-bg px-3 py-2 font-data text-xs text-tactical-text placeholder:text-tactical-text-dim outline-none transition-colors text-center uppercase focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20"
                                />
                                <datalist id="common-genotypes">
                                    <option value="AA" />
                                    <option value="AC" />
                                    <option value="AG" />
                                    <option value="AT" />
                                    <option value="CC" />
                                    <option value="CG" />
                                    <option value="CT" />
                                    <option value="GG" />
                                    <option value="GT" />
                                    <option value="TT" />
                                </datalist>

                                {/* Remove Row */}
                                <button
                                    type="button"
                                    onClick={() => removeSnpRow(row.id)}
                                    className="flex h-8 w-8 items-center justify-center rounded-md text-tactical-text-dim transition-colors hover:bg-tactical-danger/10 hover:text-tactical-danger"
                                >
                                    <Trash2 className="h-3.5 w-3.5" />
                                </button>
                            </div>
                        ))}

                        <button
                            type="button"
                            onClick={addSnpRow}
                            className="flex w-full items-center justify-center gap-1.5 rounded-md border border-dashed border-tactical-border py-2 font-data text-[10px] font-medium tracking-wider text-tactical-text-dim transition-colors uppercase hover:border-tactical-primary/30 hover:text-tactical-primary"
                        >
                            <Plus className="h-3 w-3" />
                            Add SNP Marker
                        </button>
                    </motion.div>
                )}
            </div>

            {/* ── Submit ── */}
            {/* ── Submit ── */}
            <div className="flex w-full justify-center pt-4">
                <TooltipProvider>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <button
                                type="submit"
                                disabled={submitting || !isFormValid}
                                className="
                                flex items-center justify-center gap-2 rounded-md
                                bg-tactical-primary px-8 py-2.5 min-w-[140px] shadow-[0_0_15px_rgba(16,185,129,0.15)]
                                font-data text-xs font-bold tracking-[0.15em] text-tactical-bg uppercase whitespace-nowrap
                                transition-all duration-300
                                hover:bg-tactical-primary-dim hover:shadow-[0_0_20px_rgba(16,185,129,0.4)]
                                hover:scale-[1.02] active:scale-[0.98]
                                disabled:cursor-not-allowed disabled:opacity-40 disabled:bg-tactical-surface-elevated disabled:text-tactical-text-dim disabled:shadow-none disabled:hover:scale-100
                                "
                            >
                                {submitting ? (
                                    <>
                                        {/* DNA Helix Animation */}
                                        <div className="flex items-center gap-[2px] h-3">
                                            {[0, 1, 2, 3, 4].map((i) => (
                                                <motion.div
                                                    key={i}
                                                    className="w-[2px] bg-tactical-bg rounded-full"
                                                    animate={{
                                                        height: ["4px", "12px", "4px"],
                                                        opacity: [0.5, 1, 0.5],
                                                        backgroundColor: ["#09090b", "#ffffff", "#09090b"]
                                                    }}
                                                    transition={{
                                                        duration: 0.8,
                                                        repeat: Infinity,
                                                        ease: "easeInOut",
                                                        delay: i * 0.1
                                                    }}
                                                />
                                            ))}
                                        </div>
                                        <span className="animate-pulse">Analyzing Sequence...</span>
                                    </>
                                ) : (
                                    <>
                                        <Send className="h-3.5 w-3.5" />
                                        Ingest Profile
                                    </>
                                )}
                            </button>
                        </TooltipTrigger>
                        {!isFormValid && (
                            <TooltipContent side="top" className="border-tactical-border bg-tactical-surface-elevated text-tactical-text shadow-xl">
                                <div className="flex items-center gap-2">
                                    <AlertTriangle className="h-3 w-3 text-amber-500" />
                                    <p className="font-data text-[10px] font-bold tracking-wider uppercase text-amber-500">
                                        At least 1 valid marker (STR or SNP) required
                                    </p>
                                </div>
                            </TooltipContent>
                        )}
                    </Tooltip>
                </TooltipProvider>
            </div>

            {/* ── Result feedback ── */}
            {
                result && (
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
                )
            }
        </form>
    );
}
