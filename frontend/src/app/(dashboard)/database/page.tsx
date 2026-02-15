"use client";

import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
    Database,
    Search,
    ChevronUp,
    ChevronDown,
    Dna,
    MapPin,
    Hash,
    Filter,
    Server,
    HardDrive,
} from "lucide-react";

// ─── Types ───────────────────────────────────────────────────────────────────

interface ProfileRecord {
    id: string;
    originNode: string;
    lociCount: number;
    insertedAt: string;
    quality: "complete" | "partial" | "degraded";
    vectorId: number;
}

// ─── Mock Data Generator ─────────────────────────────────────────────────────

const NODES = [
    "EUROPOL-NL", "BKA-DE", "NCA-UK", "FBI-US-DC", "NPA-JP",
    "SAPS-ZA", "PFA-AR", "CBI-IN", "KNPA-KR", "DGPN-FR",
    "RCMP-CA", "AFP-AU", "INTERPOL-EU",
];

function seededRandom(seed: number): () => number {
    let s = seed;
    return () => {
        s = (s * 16807 + 0) % 2147483647;
        return s / 2147483647;
    };
}

function generateProfiles(count: number): ProfileRecord[] {
    const rand = seededRandom(42);
    const profiles: ProfileRecord[] = [];
    for (let i = 0; i < count; i++) {
        const loci = Math.floor(rand() * 15) + 10;
        const quality: ProfileRecord["quality"] =
            loci >= 18 ? "complete" : loci >= 14 ? "partial" : "degraded";
        const month = String(Math.floor(rand() * 12) + 1).padStart(2, "0");
        const day = String(Math.floor(rand() * 28) + 1).padStart(2, "0");
        const hour = String(Math.floor(rand() * 24)).padStart(2, "0");
        const min = String(Math.floor(rand() * 60)).padStart(2, "0");

        profiles.push({
            id: `PRF-${String(i + 1).padStart(6, "0")}`,
            originNode: NODES[Math.floor(rand() * NODES.length)],
            lociCount: loci,
            insertedAt: `2026-${month}-${day} ${hour}:${min}`,
            quality,
            vectorId: 100000 + i,
        });
    }
    return profiles;
}

const ALL_PROFILES = generateProfiles(24_847);

// ─── Quality Badge ───────────────────────────────────────────────────────────

const QUALITY_CONFIG = {
    complete: { label: "COMPLETE", color: "text-emerald-400", bg: "bg-emerald-500/10" },
    partial: { label: "PARTIAL", color: "text-amber-400", bg: "bg-amber-500/10" },
    degraded: { label: "DEGRADED", color: "text-red-400", bg: "bg-red-500/10" },
} as const;

// ─── Sort Types ──────────────────────────────────────────────────────────────

type SortKey = "id" | "originNode" | "lociCount" | "insertedAt";
type SortDir = "asc" | "desc";

// ─── Page Component ──────────────────────────────────────────────────────────

export default function DatabasePage() {
    const [search, setSearch] = useState("");
    const [nodeFilter, setNodeFilter] = useState<string>("all");
    const [sortKey, setSortKey] = useState<SortKey>("id");
    const [sortDir, setSortDir] = useState<SortDir>("desc");
    const [page, setPage] = useState(0);
    const PAGE_SIZE = 15;

    const filtered = useMemo(() => {
        let data = ALL_PROFILES;
        if (search) {
            const q = search.toLowerCase();
            data = data.filter(
                (p) =>
                    p.id.toLowerCase().includes(q) ||
                    p.originNode.toLowerCase().includes(q)
            );
        }
        if (nodeFilter !== "all") {
            data = data.filter((p) => p.originNode === nodeFilter);
        }
        return data;
    }, [search, nodeFilter]);

    const sorted = useMemo(() => {
        const copy = [...filtered];
        copy.sort((a, b) => {
            const aVal = a[sortKey];
            const bVal = b[sortKey];
            if (typeof aVal === "number" && typeof bVal === "number") {
                return sortDir === "asc" ? aVal - bVal : bVal - aVal;
            }
            const aStr = String(aVal);
            const bStr = String(bVal);
            return sortDir === "asc"
                ? aStr.localeCompare(bStr)
                : bStr.localeCompare(aStr);
        });
        return copy;
    }, [filtered, sortKey, sortDir]);

    const totalPages = Math.ceil(sorted.length / PAGE_SIZE);
    const pageData = sorted.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

    const toggleSort = (key: SortKey) => {
        if (sortKey === key) {
            setSortDir((d) => (d === "asc" ? "desc" : "asc"));
        } else {
            setSortKey(key);
            setSortDir("desc");
        }
        setPage(0);
    };

    const SortIcon = ({ field }: { field: SortKey }) => {
        if (sortKey !== field) return <ChevronDown className="w-2.5 h-2.5 text-zinc-700" />;
        return sortDir === "asc"
            ? <ChevronUp className="w-2.5 h-2.5 text-tactical-primary" />
            : <ChevronDown className="w-2.5 h-2.5 text-tactical-primary" />;
    };

    // Stats
    const stats = useMemo(() => {
        const complete = ALL_PROFILES.filter((p) => p.quality === "complete").length;
        const partial = ALL_PROFILES.filter((p) => p.quality === "partial").length;
        const degraded = ALL_PROFILES.filter((p) => p.quality === "degraded").length;
        const uniqueNodes = new Set(ALL_PROFILES.map((p) => p.originNode)).size;
        return { total: ALL_PROFILES.length, complete, partial, degraded, uniqueNodes };
    }, []);

    return (
        <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6 py-2 sm:py-6 overflow-x-hidden"
        >
            {/* ── Header ── */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex flex-col lg:flex-row lg:items-center gap-2 lg:gap-3 text-center sm:text-left">
                    <div className="flex items-center justify-center sm:justify-start gap-2">
                        <Database className="w-4 h-4 text-tactical-primary" />
                        <h1 className="font-data text-xs sm:text-sm font-bold tracking-[0.1em] lg:tracking-[0.2em] text-tactical-text uppercase">
                            Milvus_Vector_Database
                        </h1>
                    </div>
                    <span className="font-data text-[7px] lg:text-[8px] text-zinc-600 tracking-wider uppercase">
                        Vector Engine v2.4 — Active Collection
                    </span>
                </div>
                <div className="flex items-center justify-center sm:justify-end gap-2">
                    <HardDrive className="w-3 h-3 text-zinc-600" />
                    <span className="font-data text-[8px] sm:text-[9px] text-zinc-600 uppercase tracking-wider">
                        Collection: str_profiles
                    </span>
                </div>
            </div>

            {/* ── Stats Strip ── */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-px bg-zinc-800/50 rounded-lg overflow-hidden border border-tactical-border">
                {[
                    { label: "Profiles", value: stats.total.toLocaleString(), color: "text-tactical-text" },
                    { label: "Complete", value: stats.complete.toLocaleString(), color: "text-emerald-400" },
                    { label: "Partial", value: stats.partial.toLocaleString(), color: "text-amber-400" },
                    { label: "Degraded", value: stats.degraded.toLocaleString(), color: "text-red-400" },
                    { label: "Nodes", value: stats.uniqueNodes.toString(), color: "text-cyan-400" },
                ].map((s, idx) => (
                    <div
                        key={s.label}
                        className={`text-center px-2 py-3 bg-tactical-surface ${idx === 4 ? "col-span-2 lg:col-span-1 border-t border-tactical-border lg:border-t-0" : ""}`}
                    >
                        <p className={`font-data text-base sm:text-lg font-bold tabular-nums ${s.color}`}>{s.value}</p>
                        <p className="font-data text-[7px] sm:text-[8px] uppercase tracking-[0.15em] text-zinc-600 mt-0.5">{s.label}</p>
                    </div>
                ))}
            </div>

            {/* ── Toolbar ── */}
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 lg:gap-3">
                <div className="flex-1 relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-zinc-600" />
                    <input
                        type="text"
                        value={search}
                        onChange={(e) => { setSearch(e.target.value); setPage(0); }}
                        placeholder="Search ID or Node..."
                        className="w-full pl-9 pr-3 py-2 rounded-md border border-tactical-border bg-tactical-bg
                            font-data text-[10px] lg:text-xs text-tactical-text placeholder:text-zinc-600
                            outline-none transition-colors
                            focus:border-tactical-primary/40 focus:ring-1 focus:ring-tactical-primary/20"
                    />
                </div>
                <div className="flex items-center gap-1.5 justify-end">
                    <Filter className="w-3 h-3 text-zinc-600" />
                    <select
                        value={nodeFilter}
                        onChange={(e) => { setNodeFilter(e.target.value); setPage(0); }}
                        className="bg-tactical-bg border border-tactical-border rounded-md px-2 py-1.5 lg:py-2
                            font-data text-[9px] lg:text-[10px] text-tactical-text outline-none
                            focus:border-tactical-primary/40 cursor-pointer min-w-[120px]"
                    >
                        <option value="all">All Nodes</option>
                        {NODES.map((n) => (
                            <option key={n} value={n}>{n}</option>
                        ))}
                    </select>
                </div>
            </div>

            {/* ── Table ── */}
            <div className="rounded-lg border border-tactical-border overflow-hidden">
                {/* Head */}
                <div className="grid grid-cols-[1fr_1fr_0.8fr] sm:grid-cols-[1.5fr_1.2fr_0.8fr_0.6fr_1fr_0.8fr] gap-0 bg-tactical-surface border-b border-tactical-border">
                    {[
                        { key: "id" as SortKey, label: "Profile ID", mobile: true },
                        { key: "originNode" as SortKey, label: "Node", mobile: true },
                        { key: "lociCount" as SortKey, label: "Loci", mobile: true },
                        { key: null, label: "Quality", mobile: false },
                        { key: "insertedAt" as SortKey, label: "Indexed", mobile: false },
                        { key: null, label: "Vector", mobile: false },
                    ].map((col) => (
                        <button
                            key={col.label}
                            onClick={() => col.key && toggleSort(col.key)}
                            disabled={!col.key}
                            className={`items-center gap-1 px-3 py-2.5 text-left
                                font-data text-[8px] font-bold uppercase tracking-[0.1em] text-zinc-500
                                ${col.key ? "hover:text-zinc-300 cursor-pointer" : "cursor-default"}
                                ${col.mobile ? "flex" : "hidden sm:flex"}`}
                        >
                            {col.label}
                            {col.key && <SortIcon field={col.key} />}
                        </button>
                    ))}
                </div>

                {/* Body */}
                <div>
                    {pageData.map((profile, i) => {
                        const q = QUALITY_CONFIG[profile.quality];
                        return (
                            <motion.div
                                key={profile.id}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: i * 0.005 }}
                                className="grid grid-cols-[1fr_1fr_0.8fr] sm:grid-cols-[1.5fr_1.2fr_0.8fr_0.6fr_1fr_0.8fr] gap-0 border-b border-tactical-border-subtle
                                    hover:bg-tactical-surface-elevated/50 transition-colors"
                            >
                                {/* Profile ID */}
                                <div className="flex items-center gap-1.5 px-3 py-2 min-w-0">
                                    <Dna className="w-2.5 h-2.5 text-zinc-700 flex-shrink-0" />
                                    <span className="font-data text-[9px] font-medium text-tactical-text truncate">
                                        {profile.id}
                                    </span>
                                </div>
                                {/* Origin Node */}
                                <div className="flex items-center gap-1 px-3 py-2 min-w-0">
                                    <MapPin className="w-2 h-2 text-zinc-700 flex-shrink-0" />
                                    <span className="font-data text-[9px] text-zinc-500 truncate">
                                        {profile.originNode}
                                    </span>
                                </div>
                                {/* Loci Count */}
                                <div className="flex items-center px-3 py-2">
                                    <span className="font-data text-[9px] font-bold tabular-nums text-tactical-text">
                                        {profile.lociCount}
                                    </span>
                                    <span className="font-data text-[7px] text-zinc-700 ml-0.5">/24</span>
                                </div>
                                {/* Quality */}
                                <div className="hidden sm:flex items-center px-3 py-2">
                                    <span className={`font-data text-[7px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded ${q.bg} ${q.color}`}>
                                        {q.label}
                                    </span>
                                </div>
                                {/* Indexed At */}
                                <div className="hidden sm:flex items-center px-3 py-2">
                                    <span className="font-data text-[8px] text-zinc-500 tabular-nums truncate">
                                        {profile.insertedAt}
                                    </span>
                                </div>
                                {/* Vector ID */}
                                <div className="hidden sm:flex items-center gap-1 px-3 py-2">
                                    <Hash className="w-2 h-2 text-zinc-700 flex-shrink-0" />
                                    <span className="font-data text-[8px] text-zinc-600 tabular-nums">
                                        {profile.vectorId}
                                    </span>
                                </div>
                            </motion.div>
                        );
                    })}
                </div>
            </div>

            {/* ── Pagination ── */}
            <div className="flex items-center justify-between">
                <span className="font-data text-[9px] text-zinc-600">
                    Showing {page * PAGE_SIZE + 1}–{Math.min((page + 1) * PAGE_SIZE, sorted.length)} of{" "}
                    {sorted.length.toLocaleString()} profiles
                </span>
                <div className="flex items-center gap-1">
                    <button
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={page === 0}
                        className="px-2.5 py-1 rounded border border-tactical-border font-data text-[9px] text-zinc-500
                            hover:text-tactical-text hover:border-tactical-primary/30 transition-colors
                            disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                        Prev
                    </button>
                    <span className="font-data text-[9px] text-zinc-500 px-2 tabular-nums">
                        {page + 1} / {totalPages}
                    </span>
                    <button
                        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                        disabled={page >= totalPages - 1}
                        className="px-2.5 py-1 rounded border border-tactical-border font-data text-[9px] text-zinc-500
                            hover:text-tactical-text hover:border-tactical-primary/30 transition-colors
                            disabled:opacity-30 disabled:cursor-not-allowed"
                    >
                        Next
                    </button>
                </div>
            </div>

            {/* ── Footer ── */}
            <div className="flex items-center gap-2 pt-2 border-t border-tactical-border-subtle">
                <Server className="w-3 h-3 text-zinc-700" />
                <span className="font-data text-[7px] text-zinc-700 uppercase tracking-wider">
                    Milvus v2.4 • IVF_FLAT Index • Dimension 384 • Distance: COSINE
                </span>
            </div>
        </motion.div>
    );
}
