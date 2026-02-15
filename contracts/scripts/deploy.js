const hre = require("hardhat");

async function main() {
    const [deployer] = await hre.ethers.getSigners();
    console.log("Deploying contracts with the account:", deployer.address);

    const VantageAudit = await hre.ethers.getContractFactory("VantageAudit");
    const audit = await VantageAudit.deploy();

    await audit.waitForDeployment();

    const address = await audit.getAddress();
    console.log("VantageAudit deployed to:", address);

    // Whitelist the deployer as the first investigator for testing
    // (In production, you'd separate Admin and Investigator)
    console.log("Authorizing deployer...");
    const tx = await audit.authorizeInvestigator(deployer.address, "Admin/Relayer", "System Root");
    await tx.wait();
    console.log("Deployer authorized.");
}

main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
});
