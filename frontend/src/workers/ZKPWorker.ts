// ZKPWorker.ts
// Handles heavy cryptographic computations off the main thread.

// @ts-ignore
import { groth16 } from 'snarkjs';

interface WorkerData {
    input: any;
    wasmPath: string;
    zkeyPath: string;
}

self.onmessage = async (event: MessageEvent<WorkerData>) => {
    const { input, wasmPath, zkeyPath } = event.data;

    try {
        // Generate Proof
        // @ts-ignore - groth16 types might be tricky in pure worker context without full setup
        const { proof, publicSignals } = await groth16.fullProve(input, wasmPath, zkeyPath);

        // Send back result
        self.postMessage({ type: 'PROOF_GENERATED', proof, publicSignals });
    } catch (error: any) {
        console.error("ZKP Worker Error:", error);
        self.postMessage({ type: 'ERROR', error: error.message || "Unknown error during proof generation" });
    }
};
