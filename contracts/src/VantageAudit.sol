// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Pausable.sol";

/**
 * @title VantageAudit
 * @notice High-Security Forensic Audit Ledger for VANTAGE-STR.
 * @dev Manages investigator access, session tokens, and immutable logging.
 *      Includes autonomous rate-limiting and global lockdown security.
 */
contract VantageAudit is Ownable, Pausable {
    
    // ========================= Types =========================

    struct Profile {
        string  name;       // e.g., "Agent Smith - FBI Cyber"
        bool    isAuthorized;
        uint256 createdAt;
    }

    struct LogEntry {
        address investigator;
        string  queryType;    // e.g., "STR_MATCH", "KINSHIP"
        bytes32 profileHash;  // Hash of DNA identifiers
        uint256 timestamp;
    }

    // ===================== State Variables =====================

    // Identify & Access Management
    mapping(address => Profile) public profiles;
    mapping(address => bytes32) private _sessionTokens; // AccessGrant

    // Immutable Ledger
    LogEntry[] public auditTrail;

    // Security: Rate Limiting
    uint256 public constant RATE_LIMIT_THRESHOLD = 5;
    uint256 public constant RATE_LIMIT_WINDOW    = 60; // seconds

    struct RateLimitConfig {
        uint256 count;
        uint256 windowStart;
    }
    mapping(address => RateLimitConfig) private _rateLimits;

    // Security: Global Anomaly Detection
    uint256 public constant GLOBAL_FAILURE_THRESHOLD = 10;
    uint256 public globalFailureCount;
    uint256 public lastFailureTimestamp;

    // ========================= Events ==========================

    event InvestigatorAuthorized(address indexed investigator, string name);
    event InvestigatorRevoked(address indexed investigator);
    event SessionGranted(address indexed investigator, bytes32 sessionToken);
    event QueryLogged(
        uint256 indexed logIndex,
        address indexed investigator,
        string  queryType,
        bytes32 profileHash,
        uint256 timestamp
    );
    event RateLimitExceeded(address indexed investigator);
    event GlobalLockdownTriggered(uint256 timestamp, string reason);

    // ======================= Modifiers =========================

    /**
     * @dev Validates investigator status and session token.
     */
    modifier onlyAuthorized(bytes32 _sessionToken) {
        require(!paused(), "System is in LOCKDOWN mode");
        require(profiles[msg.sender].isAuthorized, "Investigator not authorized");
        require(_sessionTokens[msg.sender] == _sessionToken, "Invalid session token");
        _;
    }

    // ====================== Constructor ========================

    constructor() Ownable(msg.sender) {
        // Initial state is active (not paused)
    }

    // ================= IAM: Admin Functions ====================

    /**
     * @notice Grants authorization to a new or existing investigator.
     */
    function authorizeInvestigator(address _investigator, string calldata _name) external onlyOwner {
        profiles[_investigator] = Profile({
            name: _name,
            isAuthorized: true,
            createdAt: block.timestamp
        });
        emit InvestigatorAuthorized(_investigator, _name);
    }

    /**
     * @notice Revokes an investigator's access immediately.
     */
    function revokeInvestigator(address _investigator) external onlyOwner {
        profiles[_investigator].isAuthorized = false;
        delete _sessionTokens[_investigator]; // Kill session
        emit InvestigatorRevoked(_investigator);
    }

    /**
     * @notice Generates a new session token for an authorized investigator.
     *         In a real scenario, this might involve off-chain signature verification.
     */
    function grantSession(address _investigator, bytes32 _newToken) external onlyOwner {
        require(profiles[_investigator].isAuthorized, "Investigator not authorized");
        _sessionTokens[_investigator] = _newToken;
        emit SessionGranted(_investigator, _newToken);
    }

    /**
     * @notice Emergency reset of the global lockdown.
     */
    function liftLockdown() external onlyOwner {
        _unpause();
        globalFailureCount = 0;
    }

    // =================== Core: Log Query =======================

    /**
     * @notice Records a forensic query on the immutable ledger.
     * @dev Checks rate limits and active session.
     */
    function logQuery(
        string calldata _queryType,
        bytes32 _profileHash,
        bytes32 _sessionToken
    ) external onlyAuthorized(_sessionToken) {
        
        // 1. Check Rate Limit
        if (!_checkRateLimit(msg.sender)) {
            return; // Exit without logging (but suspension is saved)
        }

        // 2. Record Access
        LogEntry memory entry = LogEntry({
            investigator: msg.sender,
            queryType: _queryType,
            profileHash: _profileHash,
            timestamp: block.timestamp
        });

        auditTrail.push(entry);

        emit QueryLogged(
            auditTrail.length - 1,
            msg.sender,
            _queryType,
            _profileHash,
            block.timestamp
        );
    }

    // =================== internal: Security ====================

    function _checkRateLimit(address _investigator) internal returns (bool) {
        RateLimitConfig storage config = _rateLimits[_investigator];

        if (block.timestamp > config.windowStart + RATE_LIMIT_WINDOW) {
            // New window
            config.count = 1;
            config.windowStart = block.timestamp;
            return true;
        } else {
            // Existing window
            config.count++;
            if (config.count > RATE_LIMIT_THRESHOLD) {
                // Auto-Revoke (Persisted by NOT reverting)
                profiles[_investigator].isAuthorized = false;
                delete _sessionTokens[_investigator];
                emit RateLimitExceeded(_investigator);
                
                _registerSystemFailure(); // Contribute to global risk score
                
                return false; // Access Denied
            }
            return true; // Access Granted
        }
    }

    function _registerSystemFailure() internal {
        // If failures happen rapidly, trigger lockdown
        if (block.timestamp > lastFailureTimestamp + RATE_LIMIT_WINDOW) {
            globalFailureCount = 1;
        } else {
            globalFailureCount++;
        }
        
        lastFailureTimestamp = block.timestamp;

        if (globalFailureCount >= GLOBAL_FAILURE_THRESHOLD) {
            _pause(); // Enable Enforced Pause (Lockdown)
            emit GlobalLockdownTriggered(block.timestamp, "Anomaly Threshold Exceeded");
        }
    }

    // ===================== View Helpers ========================

    function getAuditLength() external view returns (uint256) {
        return auditTrail.length;
    }

    function getInvestigator(address _addr) external view returns (Profile memory) {
        return profiles[_addr];
    }
}
