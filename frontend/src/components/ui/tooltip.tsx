"use client";

import * as React from "react";
import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils";

interface TooltipContextType {
    open: boolean;
    setOpen: (open: boolean) => void;
}

const TooltipContext = React.createContext<TooltipContextType | undefined>(undefined);

export function TooltipProvider({ children }: { children: React.ReactNode }) {
    return <>{children}</>;
}

export function Tooltip({ children }: { children: React.ReactNode }) {
    const [open, setOpen] = useState(false);
    return (
        <TooltipContext.Provider value={{ open, setOpen }}>
            <div
                className="relative inline-block"
                onMouseEnter={() => setOpen(true)}
                onMouseLeave={() => setOpen(false)}
                onFocus={() => setOpen(true)}
                onBlur={() => setOpen(false)}
            >
                {children}
            </div>
        </TooltipContext.Provider>
    );
}

export function TooltipTrigger({ children, asChild }: { children: React.ReactNode; asChild?: boolean }) {
    return <>{children}</>;
}

export function TooltipContent({
    children,
    className,
    side = "top"
}: {
    children: React.ReactNode;
    className?: string;
    side?: "top" | "bottom" | "left" | "right"
}) {
    const context = React.useContext(TooltipContext);
    if (!context?.open) return null;

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, scale: 0.95, y: 5 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95 }}
                transition={{ duration: 0.15 }}
                className={cn(
                    "absolute z-50 px-3 py-1.5 text-xs rounded-md border shadow-md",
                    "whitespace-nowrap pointer-events-none", // Prevent flickering on hover
                    // Positioning logic
                    side === "top" && "bottom-full left-1/2 -translate-x-1/2 mb-2",
                    side === "bottom" && "top-full left-1/2 -translate-x-1/2 mt-2",
                    className
                )}
            >
                {children}
                {/* Arrow */}
                <div className={cn(
                    "absolute w-2 h-2 rotate-45 border-r border-b bg-inherit",
                    side === "top" && "-bottom-1 left-1/2 -translate-x-1/2 border-t-0 border-l-0 border-tactical-border",
                    // Ensure background matches parent
                )} />
            </motion.div>
        </AnimatePresence>
    );
}
