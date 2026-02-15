"use client";

import { useEffect, useState } from "react";
import { Brain, Bot, FileText, CheckCircle } from "lucide-react";

// Mock Agent Thoughts - in real app, subscribe to analysis context
const MOCK_THOUGHTS = [
    { step: 1, content: "Analyzing STR markers at D3S1358...", duration: 120 },
    { step: 2, content: "Calculating Likelihood Ratio (Bayesian)...", duration: 450 },
    { step: 3, content: "Scanning criminal database for partial hits...", duration: 800 },
    { step: 4, content: "Phenotype prediction: High confidence for European ancestry.", duration: 300 },
];

export default function InvestigatorSidebar() {
    const [thoughts, setThoughts] = useState<typeof MOCK_THOUGHTS>([]);

    useEffect(() => {
        // Simulate stream
        let i = 0;
        const interval = setInterval(() => {
            if (i < MOCK_THOUGHTS.length) {
                setThoughts(prev => [...prev, MOCK_THOUGHTS[i]]);
                i++;
            } else {
                clearInterval(interval);
            }
        }, 1500);
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="h-full flex flex-col bg-zinc-950/50 backdrop-blur-sm">
            <div className="p-4 border-b border-zinc-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Brain className="w-4 h-4 text-purple-400" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">Aura Logic</h3>
                </div>
                <span className="px-1.5 py-0.5 rounded bg-purple-500/10 border border-purple-500/20 text-[8px] text-purple-400 font-bold uppercase">Online</span>
            </div>

            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {/* Agent Introduction */}
                <div className="flex gap-3">
                    <div className="w-8 h-8 rounded-full bg-purple-500/10 border border-purple-500/20 flex items-center justify-center shrink-0">
                        <Bot className="w-4 h-4 text-purple-400" />
                    </div>
                    <div className="flex-1">
                        <div className="bg-zinc-900/50 border border-zinc-800 rounded-lg rounded-tl-none p-3">
                            <p className="text-xs text-zinc-400 leading-relaxed font-mono">
                                Awaiting forensic data input. I am ready to analyze STR profiles and verify chain integrity via ZKP.
                            </p>
                        </div>
                        <span className="text-[10px] text-zinc-600 mt-1 block">Just now</span>
                    </div>
                </div>

                {/* Analysis Stream */}
                {thoughts.map((thought, i) => (
                    <div key={thought.step} className="flex gap-3">
                        <div className="w-8 flex flex-col items-center pt-2">
                            <div className="w-1.5 h-1.5 rounded-full bg-purple-500 animate-pulse" />
                            {i < thoughts.length - 1 && <div className="w-px h-full bg-zinc-800 my-1" />}
                        </div>
                        <div className="flex-1 pb-4">
                            <p className="text-xs text-zinc-300 font-mono">{thought.content}</p>
                            <span className="text-[9px] text-zinc-600 font-mono mt-1 block">{thought.duration}ms</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* Input Area (Mock) */}
            <div className="p-4 border-t border-zinc-800">
                <div className="relative">
                    <input
                        type="text"
                        placeholder="Ask agent..."
                        disabled
                        className="w-full bg-zinc-900 border border-zinc-800 rounded px-3 py-2 text-xs text-zinc-400 focus:outline-none focus:border-purple-500/50 cursor-not-allowed"
                    />
                </div>
            </div>
        </div>
    );
}
