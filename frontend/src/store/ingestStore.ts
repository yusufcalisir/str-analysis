import { create } from "zustand";

interface IngestState {
    lastIngestedProfileId: string | null;
    lastIngestedNodeId: string | null;
    markerCount: number;
    isValid: boolean;
    setLastIngested: (profileId: string, nodeId: string, markerCount: number) => void;
    clear: () => void;
}


import { persist, createJSONStorage } from "zustand/middleware";

export const useIngestStore = create<IngestState>()(
    persist(
        (set) => ({
            lastIngestedProfileId: null,
            lastIngestedNodeId: null,
            markerCount: 0,
            isValid: false,
            setLastIngested: (profileId, nodeId, markerCount) =>
                set({
                    lastIngestedProfileId: profileId,
                    lastIngestedNodeId: nodeId,
                    markerCount,
                    isValid: markerCount >= 13 // CODIS minimum threshold for reliable matching
                }),
            clear: () =>
                set({
                    lastIngestedProfileId: null,
                    lastIngestedNodeId: null,
                    markerCount: 0,
                    isValid: false
                }),
        }),
        {
            name: "vantage-ingest-storage",
            storage: createJSONStorage(() => sessionStorage), // using sessionStorage so it clears on tab close, or localStorage if desired
        }
    )
);
