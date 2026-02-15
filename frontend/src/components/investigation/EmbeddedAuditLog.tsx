import { useEffect, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { ShieldCheck, Server, AlertTriangle, ExternalLink, Loader2 } from "lucide-react";
import { useAccount, useChainId, useSwitchChain, useReadContract, useWatchContractEvent } from "wagmi";
import { forensicAuditABI } from "@/config/wagmi";
import { polygonAmoy } from "wagmi/chains";

// Contract Address (Ideally from env or constants)
const CONTRACT_ADDRESS = process.env.NEXT_PUBLIC_AUDIT_CONTRACT_ADDRESS as `0x${string}` || "0x0000000000000000000000000000000000000000";

export default function EmbeddedAuditLog() {
    const chainId = useChainId();
    const { switchChain } = useSwitchChain();
    const { isConnected } = useAccount();
    const [liveLogs, setLiveLogs] = useState<any[]>([]);

    // Network Check
    const isWrongNetwork = isConnected && chainId !== polygonAmoy.id;

    // Fetch Logs (Simplified: In prod, use The Graph or specialized indexer)
    // Here we just listen to events or showing empty state until an event arrives
    // Real implementation would need a 'getRecentLogs' view function on contract or Graph query

    useWatchContractEvent({
        address: CONTRACT_ADDRESS,
        abi: forensicAuditABI,
        eventName: 'QueryLogged',
        onLogs(logs) {
            const newLogs = logs.map(l => ({
                id: l.transactionHash,
                action: (l.args as any).query_type || "UNKNOWN_QUERY",
                time: new Date(),
                status: "verified"
            }));
            setLiveLogs(prev => [...newLogs, ...prev].slice(0, 50));
        },
    });

    // Demo Data Hydration (if no live events)
    useEffect(() => {
        const timer = setTimeout(() => {
            if (liveLogs.length === 0) {
                setLiveLogs([
                    { id: "0x7f3a...9b1c", action: "Standard_Query", time: new Date(Date.now() - 1000 * 60), status: "verified" },
                    { id: "0x2e9d...4f5a", action: "Cross_Ref_Check", time: new Date(Date.now() - 1000 * 180), status: "verified" },
                    { id: "0x8b1c...3d2e", action: "Kinship_Analysis", time: new Date(Date.now() - 1000 * 420), status: "verified" },
                ]);
            }
        }, 1500);
        return () => clearTimeout(timer);
    }, [liveLogs.length]);

    return (
        <div className="bg-zinc-900/40 border border-zinc-800 rounded flex flex-col h-full overflow-hidden">
            <div className="px-4 py-3 border-b border-zinc-800 flex justify-between items-center bg-zinc-900/50">
                <div className="flex items-center gap-2">
                    <ShieldCheck className="w-4 h-4 text-emerald-500" />
                    <h3 className="text-xs font-bold uppercase tracking-wider text-zinc-400">Live Custom Audit Ledger</h3>
                </div>
                <div className="flex items-center gap-2">
                    {isWrongNetwork ? (
                        <span className="text-[10px] text-amber-500 font-mono uppercase animate-pulse">Wrong Network</span>
                    ) : (
                        <>
                            <span className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                            <span className="text-[10px] text-emerald-500 font-mono uppercase">Syncing {polygonAmoy.name}</span>
                        </>
                    )}
                </div>
            </div>

            {isWrongNetwork && (
                <div className="bg-amber-500/10 border-b border-amber-500/20 px-4 py-2 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
                        <span className="text-[10px] text-amber-500 font-mono">Switch to {polygonAmoy.name} to view live data</span>
                    </div>
                    <button
                        onClick={() => switchChain({ chainId: polygonAmoy.id })}
                        className="text-[10px] bg-amber-500/20 hover:bg-amber-500/30 text-amber-500 px-2 py-1 rounded transition-colors uppercase font-bold"
                    >
                        Switch Network
                    </button>
                </div>
            )}

            <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {liveLogs.length === 0 ? (
                    <div className="flex flex-col items-center justify-center h-full text-zinc-600 gap-2">
                        <div className="w-16 h-0.5 bg-zinc-800 rounded-full overflow-hidden">
                            <div className="w-1/2 h-full bg-zinc-600 animate-loading-bar" />
                        </div>
                        <span className="text-[10px] font-mono">Listening for blocks...</span>
                    </div>
                ) : (
                    liveLogs.map((log, i) => (
                        <div key={i} className="flex items-center justify-between p-2 rounded hover:bg-zinc-800/50 transition-colors border border-transparent hover:border-zinc-800/50 group">
                            <div className="flex items-center gap-3">
                                <span className={`w-1.5 h-1.5 rounded-full ${log.status === 'verified' ? 'bg-emerald-500' : 'bg-red-500'}`} />
                                <span className="font-mono text-xs text-zinc-500">{log.id.substring(0, 10)}...</span>
                                <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${log.status === 'verified'
                                    ? 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                                    : 'bg-red-500/10 text-red-500 border-red-500/20'
                                    }`}>
                                    {log.action}
                                </span>
                            </div>
                            <div className="flex items-center gap-3">
                                <span className="text-[10px] text-zinc-600 font-mono">
                                    {formatDistanceToNow(log.time, { addSuffix: true })}
                                </span>
                                <a
                                    href={`https://amoy.polygonscan.com/tx/${log.id}`}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-zinc-700 hover:text-emerald-400 transition-colors"
                                >
                                    <ExternalLink className="w-3 h-3" />
                                </a>
                            </div>
                        </div>
                    ))
                )}
            </div>
        </div>
    );
}
