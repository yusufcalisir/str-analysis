"use client";

import { motion } from "framer-motion";
import { Activity, ShieldCheck, Database, Server, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";

// Mock Hook for now - in real app, replace with useQuery
function useSystemMetrics() {
    const [metrics, setMetrics] = useState({
        blockHeight: 18459203,
        milvusLatency: 12, // ms
        zkpStatus: "Active",
        networkHealth: "Optimal"
    });

    useEffect(() => {
        const interval = setInterval(() => {
            setMetrics(prev => ({
                ...prev,
                blockHeight: prev.blockHeight + 1,
                milvusLatency: Math.floor(Math.random() * 5) + 8, // 8-13ms
            }));
        }, 3000);
        return () => clearInterval(interval);
    }, []);

    return metrics;
}

export default function SystemPulse() {
    const metrics = useSystemMetrics();

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {/* 1. Network Status */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-zinc-900/40 border border-zinc-800 rounded p-3 flex items-center justify-between"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-emerald-500/10 rounded-full">
                        <Activity className="w-4 h-4 text-emerald-400" />
                    </div>
                    <div>
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Network Status</p>
                        <p className="text-sm font-mono text-emerald-400 font-bold">{metrics.networkHealth}</p>
                    </div>
                </div>
                <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_rgba(16,185,129,0.5)]" />
            </motion.div>

            {/* 2. Block Height */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-zinc-900/40 border border-zinc-800 rounded p-3 flex items-center justify-between"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-blue-500/10 rounded-full">
                        <Database className="w-4 h-4 text-blue-400" />
                    </div>
                    <div>
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Block Height</p>
                        <p className="text-sm font-mono text-zinc-300">#{metrics.blockHeight.toLocaleString()}</p>
                    </div>
                </div>
            </motion.div>

            {/* 3. ZKP Engine */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="bg-zinc-900/40 border border-zinc-800 rounded p-3 flex items-center justify-between"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-purple-500/10 rounded-full">
                        <ShieldCheck className="w-4 h-4 text-purple-400" />
                    </div>
                    <div>
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">ZKP Engine</p>
                        <p className="text-sm font-mono text-purple-400 font-bold">{metrics.zkpStatus}</p>
                    </div>
                </div>
            </motion.div>

            {/* 4. Milvus Latency */}
            <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="bg-zinc-900/40 border border-zinc-800 rounded p-3 flex items-center justify-between"
            >
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-amber-500/10 rounded-full">
                        <Server className="w-4 h-4 text-amber-400" />
                    </div>
                    <div>
                        <p className="text-[10px] text-zinc-500 uppercase font-bold tracking-wider">Vector Search</p>
                        <div className="flex items-baseline gap-1">
                            <p className="text-sm font-mono text-zinc-300">{metrics.milvusLatency}ms</p>
                            <p className="text-[10px] text-zinc-600">latency</p>
                        </div>
                    </div>
                </div>
                {metrics.milvusLatency < 15 && (
                    <div className="px-1.5 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/20 text-[8px] text-emerald-400 font-bold uppercase">
                        Fast
                    </div>
                )}
            </motion.div>
        </div>
    );
}
