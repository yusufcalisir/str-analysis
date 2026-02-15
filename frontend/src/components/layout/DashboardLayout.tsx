"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { useState } from "react";
import {
    Network,
    Database,
    FlaskConical,
    ShieldCheck,
    Activity,
    ChevronLeft,
    ChevronRight,
    Radio,
    Menu,
    X,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

/* ── Navigation Items ── */
const NAV_ITEMS = [
    { id: "nodes", label: "Nodes", href: "/nodes", icon: Network },
    { id: "database", label: "Database", href: "/database", icon: Database },
    { id: "analysis", label: "Analysis", href: "/analysis", icon: FlaskConical },
    { id: "audit", label: "Audit Log", href: "/audit", icon: ShieldCheck },
] as const;

/* ── Mock network data ── */
const MOCK_NODES_ONLINE = 7;
const MOCK_LATENCY = 12;

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    const pathname = usePathname();
    const [sidebarCollapsed, setSidebarCollapsed] = useState(true);
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    return (
        <div className="flex min-h-screen lg:h-screen lg:overflow-hidden bg-tactical-bg text-tactical-text">
            {/* ════════════════════════════════════════════════
          MOBILE SIDEBAR OVERLAY
         ════════════════════════════════════════════════ */}
            <AnimatePresence>
                {mobileMenuOpen && (
                    <>
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            onClick={() => setMobileMenuOpen(false)}
                            className="fixed inset-0 z-[60] bg-black/60 backdrop-blur-sm lg:hidden"
                        />
                        <motion.aside
                            initial={{ x: "-100%" }}
                            animate={{ x: 0 }}
                            exit={{ x: "-100%" }}
                            transition={{ type: "spring", damping: 25, stiffness: 200 }}
                            className="fixed inset-y-0 left-0 z-[70] w-64 bg-tactical-surface border-r border-tactical-border lg:hidden"
                        >
                            <div className="flex h-14 items-center justify-between px-4 border-b border-tactical-border">
                                <span className="font-data text-xs font-bold tracking-widest text-tactical-primary">VANTAGE-STR</span>
                                <button onClick={() => setMobileMenuOpen(false)} className="text-tactical-text-dim hover:text-white transition-colors">
                                    <X className="h-5 w-5" />
                                </button>
                            </div>
                            <nav className="p-4 space-y-2">
                                {NAV_ITEMS.map((item) => {
                                    const Icon = item.icon;
                                    const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
                                    return (
                                        <Link
                                            key={item.id}
                                            href={item.href}
                                            onClick={() => setMobileMenuOpen(false)}
                                            className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-all ${isActive
                                                ? "bg-tactical-primary/10 border-tactical-primary/30 text-tactical-primary"
                                                : "border-transparent text-tactical-text-dim hover:bg-tactical-surface-elevated hover:text-white"
                                                }`}
                                        >
                                            <Icon className="h-4 w-4" />
                                            <span className="font-data text-xs font-medium tracking-wide">{item.label}</span>
                                        </Link>
                                    );
                                })}
                            </nav>
                        </motion.aside>
                    </>
                )}
            </AnimatePresence>

            {/* ════════════════════════════════════════════════
          DESKTOP SIDEBAR
         ════════════════════════════════════════════════ */}
            <aside
                className={`
          relative hidden flex-col border-r border-tactical-border
          bg-tactical-surface transition-all duration-300 ease-out lg:flex
          ${sidebarCollapsed ? "w-16" : "w-52"}
        `}
            >
                {/* Logo */}
                <div className="flex h-14 items-center justify-center border-b border-tactical-border px-3">
                    <div className="flex items-center gap-2 overflow-hidden">
                        <div className="relative flex h-8 w-8 shrink-0 items-center justify-center">
                            <div className="absolute inset-0 rounded-md bg-tactical-primary/10" />
                            <ShieldCheck className="h-4 w-4 text-tactical-primary" />
                        </div>
                        {!sidebarCollapsed && (
                            <span className="font-data text-xs font-semibold tracking-widest text-tactical-primary whitespace-nowrap">
                                VNT-STR
                            </span>
                        )}
                    </div>
                </div>

                {/* Nav Links */}
                <nav className="flex-1 space-y-1 px-2 py-4">
                    {NAV_ITEMS.map((item) => {
                        const Icon = item.icon;
                        const isActive = pathname === item.href || pathname.startsWith(item.href + "/");
                        return (
                            <Link
                                key={item.id}
                                href={item.href}
                                title={item.label}
                                className={`
                  group relative flex w-full items-center gap-3 rounded-md px-3 py-2.5
                  text-sm transition-all duration-200
                  ${isActive
                                        ? "bg-tactical-primary/10 text-[#22C55E]"
                                        : "text-tactical-text-muted hover:bg-tactical-surface-elevated hover:text-tactical-text"
                                    }
                `}
                                style={isActive ? { textShadow: "0 0 12px rgba(34,197,94,0.5), 0 0 24px rgba(34,197,94,0.2)" } : undefined}
                            >
                                <Icon className={`h-4 w-4 shrink-0 transition-all duration-200 ${isActive ? "drop-shadow-[0_0_6px_rgba(34,197,94,0.6)]" : ""}`} />
                                {!sidebarCollapsed && (
                                    <span className="font-data text-xs font-medium tracking-wide whitespace-nowrap">
                                        {item.label}
                                    </span>
                                )}
                                {/* Active indicator bar */}
                                {isActive && (
                                    <div className="absolute left-0 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-r bg-tactical-primary shadow-[0_0_8px_rgba(34,197,94,0.6)]" />
                                )}
                                {/* Tooltip when collapsed */}
                                {sidebarCollapsed && (
                                    <div className="
                    pointer-events-none absolute left-full ml-2 rounded-md
                    bg-tactical-surface-elevated px-2.5 py-1.5 text-xs font-medium
                    text-tactical-text opacity-0 shadow-lg
                    transition-opacity group-hover:opacity-100
                    border border-tactical-border
                  ">
                                        {item.label}
                                    </div>
                                )}
                            </Link>
                        );
                    })}
                </nav>

                {/* Collapse toggle */}
                <button
                    onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                    className="
            flex h-10 items-center justify-center border-t border-tactical-border
            text-tactical-text-dim transition-colors hover:text-tactical-text
          "
                >
                    {sidebarCollapsed ? (
                        <ChevronRight className="h-4 w-4" />
                    ) : (
                        <ChevronLeft className="h-4 w-4" />
                    )}
                </button>
            </aside>

            {/* ════════════════════════════════════════════════
          MAIN CONTENT AREA
         ════════════════════════════════════════════════ */}
            <div className="flex flex-1 flex-col min-h-screen lg:h-full lg:overflow-hidden overflow-x-hidden">
                {/* ── Header ── */}
                <header className="flex h-14 shrink-0 items-center justify-between border-b border-tactical-border bg-tactical-surface px-3 sm:px-4 lg:px-6 overflow-hidden">
                    {/* Left: Branding & Mobile Toggle */}
                    <div className="flex items-center gap-2 sm:gap-3 min-w-0">
                        <button
                            onClick={() => setMobileMenuOpen(true)}
                            className="p-1 text-tactical-text-dim hover:text-tactical-text lg:hidden flex-shrink-0"
                        >
                            <Menu className="h-5 w-5" />
                        </button>
                        <h1 className="font-data text-[10px] sm:text-xs lg:text-sm font-bold tracking-[0.1em] sm:tracking-[0.15em] text-tactical-text uppercase truncate">
                            VANTAGE-STR
                        </h1>
                        <span className="text-tactical-border hidden md:inline flex-shrink-0">|</span>
                        <span className="font-data text-[10px] lg:text-[11px] tracking-wider text-tactical-text-muted hidden md:inline truncate">
                            TACTICAL FORENSIC NETWORK
                        </span>
                    </div>

                    {/* Right: Network status */}
                    <div className="flex items-center gap-1.5 sm:gap-2 lg:gap-5 flex-shrink-0">
                        {/* Global Network Status */}
                        <div className="flex items-center gap-1 lg:gap-2 rounded-full border border-tactical-primary/20 bg-tactical-primary/5 px-1.5 py-0.5 sm:px-2 sm:py-1 lg:px-3 lg:py-1.5">
                            <div className="relative flex h-1 w-1 sm:h-1.5 sm:w-1.5 lg:h-2 lg:w-2">
                                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-tactical-primary opacity-60" />
                                <span className="relative inline-flex h-1 w-1 sm:h-1.5 sm:w-1.5 lg:h-2 lg:w-2 rounded-full bg-tactical-primary" />
                            </div>
                            <span className="font-data text-[7px] sm:text-[8px] lg:text-[10px] font-semibold tracking-widest text-tactical-primary uppercase truncate max-w-[30px] sm:max-w-none">
                                <span className="hidden sm:inline">ONLINE</span>
                                <span className="sm:hidden">ON</span>
                            </span>
                        </div>

                        {/* Nodes count - Hidden on small mobile */}
                        <div className="hidden items-center gap-1.5 text-tactical-text-muted sm:flex">
                            <Radio className="h-3 w-3" />
                            <span className="font-data text-[10px] tracking-wider whitespace-nowrap">
                                {MOCK_NODES_ONLINE} NODES
                            </span>
                        </div>

                        {/* Latency - Small on mobile */}
                        <div className="flex items-center gap-1 text-tactical-text-dim lg:gap-1.5">
                            <Activity className="h-2.5 w-2.5 sm:h-3 sm:w-3 flex-shrink-0" />
                            <span className="font-data text-[8px] sm:text-[9px] lg:text-[10px] tracking-wider whitespace-nowrap">
                                {MOCK_LATENCY}ms
                            </span>
                        </div>
                    </div>
                </header>

                {/* ── Page Content ── */}
                <main className="flex-1 overflow-y-auto p-4 sm:p-6">
                    {children}
                </main>

                {/* ── Footer ── */}
                <footer className="flex h-7 shrink-0 items-center justify-center border-t border-tactical-border bg-tactical-surface/30 px-6 select-none">
                    <div className="relative overflow-hidden py-1 px-4 rounded-full border border-tactical-border/30 bg-black/20">
                        <motion.span
                            className="font-data text-[7px] tracking-[0.6em] uppercase relative z-10 bg-clip-text text-transparent bg-gradient-to-r from-zinc-500 via-white to-zinc-500 bg-[length:200%_100%]"
                            animate={{
                                backgroundPosition: ["200% 0", "-200% 0"]
                            }}
                            transition={{
                                duration: 5,
                                repeat: Infinity,
                                ease: "linear"
                            }}
                            style={{
                                backgroundImage: "linear-gradient(90deg, #525252 0%, #ffffff 50%, #525252 100%)",
                                WebkitBackgroundClip: "text",
                                color: "transparent"
                            }}
                        >
                            Developed by Yusuf & İrem
                        </motion.span>
                        {/* Subtle background glow sweep */}
                        <motion.div
                            className="absolute inset-0 z-0 bg-gradient-to-r from-transparent via-tactical-primary/10 to-transparent"
                            animate={{
                                x: ['-200%', '200%']
                            }}
                            transition={{
                                duration: 8,
                                repeat: Infinity,
                                ease: "linear"
                            }}
                        />
                    </div>
                </footer>
            </div>
        </div>
    );
}
