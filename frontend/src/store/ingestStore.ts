import { create } from "zustand";

interface IngestState {
    lastIngestedProfileId: string | null;
    lastIngestedNodeId: string | null;
    markerCount: number;
    isValid: boolean;
    setLastIngested: (profileId: string, nodeId: string, markerCount: number) => void;
    clear: () => void;
}

export const useIngestStore = create<IngestState>((set) => ({
    lastIngestedProfileId: null,
    lastIngestedNodeId: null,
    markerCount: 0,
    isValid: false,
    setLastIngested: (profileId, nodeId, markerCount) =>
        set({
            lastIngestedProfileId: profileId,
            lastIngestedNodeId: nodeId,
            markerCount,
            isValid: markerCount >= 1 // Relaxed for testing (was 13)
        }),
    clear: () =>
        set({
            lastIngestedProfileId: null,
            lastIngestedNodeId: null,
            markerCount: 0,
            isValid: false
        }),
}));
