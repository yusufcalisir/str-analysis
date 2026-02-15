const hre = require("hardhat");

async function main() {
    const [deployer] = await hre.ethers.getSigners();

    console.log("Deploying VantageAudit with account:", deployer.address);

    const VantageAudit = await hre.ethers.getContractFactory("VantageAudit");
    const audit = await VantageAudit.deploy();

    await audit.waitForDeployment();

    const address = await audit.getAddress();

    console.log("VantageAudit deployed to:", address);

    // Optional: Verify on Etherscan if network is not local
    // if (network.name !== "hardhat" && network.name !== "localhost") { ... }
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
