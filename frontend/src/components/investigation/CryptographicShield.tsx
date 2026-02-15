"use client";

import { motion } from "framer-motion";
import { ShieldCheck, Lock, Binary } from "lucide-react";

export default function CryptographicShield({ active }: { active: boolean }) {
    if (!active) return null;

    return (
        <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="absolute inset-0 bg-zinc-950/80 backdrop-blur-sm flex flex-col items-center justify-center z-50 rounded-lg border border-emerald-500/30"
        >
            <div className="relative">
                {/* Rotating Rings */}
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    className="w-32 h-32 rounded-full border-2 border-emerald-500/30 border-t-emerald-400 absolute inset-0 -top-8 -left-8"
                />

                <motion.div
                    animate={{ rotate: -360 }}
                    transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
                    className="w-48 h-48 rounded-full border border-emerald-500/10 border-b-emerald-400 absolute -top-16 -left-16"
                />

                <ShieldCheck className="w-16 h-16 text-emerald-400 relative z-10" />
            </div>

            <motion.h3
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-8 text-xl font-bold tracking-widest text-emerald-400 uppercase"
            >
                Zero-Knowledge Proof Active
            </motion.h3>

            <p className="mt-2 text-zinc-400 font-mono text-xs max-w-md text-center">
                Generating mathematical proof of knowledge. Your DNA data is being hashed locally and remains private.
            </p>

            <div className="mt-6 flex items-center gap-2">
                <Lock className="w-3 h-3 text-zinc-500" />
                <span className="text-[10px] text-zinc-600 font-mono uppercase">Client-Side Encryption</span>
            </div>
        </motion.div>
    );
}
