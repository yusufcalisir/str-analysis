import { create } from "zustand";

interface AuditState {
    highlightedTxHash: string | null;
    setHighlightedTxHash: (hash: string | null) => void;
}

export const useAuditStore = create<AuditState>((set) => ({
    highlightedTxHash: null,
    setHighlightedTxHash: (hash) => set({ highlightedTxHash: hash }),
}));
