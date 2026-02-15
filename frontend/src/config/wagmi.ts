import { http, createConfig } from 'wagmi'
import { mainnet, sepolia, polygonAmoy, hardhat } from 'wagmi/chains'

export const config = createConfig({
    chains: [polygonAmoy, sepolia, mainnet],
    transports: {
        [polygonAmoy.id]: http(),
        [sepolia.id]: http(),
        [mainnet.id]: http(),
    },
})

// ForensicAudit ABI (Events + View Functions)
export const forensicAuditABI = [
    {
        "anonymous": false,
        "inputs": [
            { "indexed": true, "internalType": "uint256", "name": "logIndex", "type": "uint256" },
            { "indexed": true, "internalType": "address", "name": "investigator_id", "type": "address" },
            { "indexed": false, "internalType": "string", "name": "query_type", "type": "string" },
            { "indexed": false, "internalType": "bytes32", "name": "profile_hash", "type": "bytes32" },
            { "indexed": false, "internalType": "uint256", "name": "timestamp", "type": "uint256" }
        ],
        "name": "QueryLogged",
        "type": "event"
    },
    {
        "anonymous": false,
        "inputs": [
            { "indexed": true, "internalType": "address", "name": "investigator_id", "type": "address" },
            { "indexed": false, "internalType": "uint256", "name": "timestamp", "type": "uint256" }
        ],
        "name": "InvestigatorSuspended",
        "type": "event"
    }
] as const
