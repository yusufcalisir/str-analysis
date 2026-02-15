# ForensicAudit — Deployment Guide

## Prerequisites

- **Node.js** ≥ 18
- **npm** ≥ 9

---

## 1. Install Dependencies

```bash
cd contracts
npm install
```

## 2. Compile

```bash
npx hardhat compile
```

Expected: `Compiled 1 Solidity file successfully`.

---

## 3. Deploy — Local (Hardhat Node)

**Terminal 1** — start a local Ethereum node:

```bash
npx hardhat node
```

This spawns a JSON-RPC server at `http://127.0.0.1:8545` with 20 pre-funded accounts.

**Terminal 2** — deploy:

```bash
npx hardhat run scripts/deploy.js --network localhost
```

Output:

```
Deploying ForensicAudit with account: 0xf39F...
ForensicAudit deployed to: 0x5FbD...
```

---

## 4. Deploy — Sepolia Testnet

Set environment variables:

```powershell
$env:SEPOLIA_RPC_URL    = "https://eth-sepolia.g.alchemy.com/v2/YOUR_KEY"
$env:DEPLOYER_PRIVATE_KEY = "0xYOUR_PRIVATE_KEY"
```

> ⚠️ **Never** commit private keys. Use `.env` + `dotenv` in production.

Deploy:

```bash
npx hardhat run scripts/deploy.js --network sepolia
```

---

## 5. Post-Deploy — Grant Investigator Tokens

Use Hardhat console or a script:

```js
const contract = await ethers.getContractAt("ForensicAudit", "DEPLOYED_ADDRESS");
await contract.grantToken("0xInvestigatorAddress");
```

---

## Contract API Summary

| Function | Access | Description |
|---|---|---|
| `logQuery(string, bytes32)` | Authorized | Record a DNA search query |
| `grantToken(address)` | Owner | Authorize an investigator |
| `revokeToken(address)` | Owner | Remove access |
| `reinstateInvestigator(address)` | Owner | Lift suspension & reset rate limit |
| `getLogCount()` | Public | Total logged queries |
| `getLog(uint256)` | Public | Retrieve a specific log entry |
