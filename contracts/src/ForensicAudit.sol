// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ForensicAudit
 * @notice VANTAGE-STR Access Control & Forensic Query Ledger
 * @dev Immutable on-chain audit trail for DNA search operations.
 *      Implements token-gated access and automatic rate-limit suspension.
 */
contract ForensicAudit {
    // ========================= Types =========================

    enum Status {
        ACTIVE,
        SUSPENDED
    }

    struct LogEntry {
        address investigator_id;
        string  query_type;
        bytes32 profile_hash;
        uint256 timestamp;
    }

    // ===================== Rate-Limit Config =====================

    uint256 private constant MAX_CALLS    = 5;
    uint256 private constant WINDOW       = 60; // seconds

    struct RateLimitState {
        uint256[5] timestamps; // fixed ring buffer
        uint256    index;      // current write position
    }

    // ========================== Storage ==========================

    address public owner;

    LogEntry[] public logs;

    mapping(address => bool)           public active_tokens;
    mapping(address => Status)         public investigator_status;
    mapping(address => RateLimitState) private _rateLimit;

    // ========================== Events ===========================

    event QueryLogged(
        uint256 indexed logIndex,
        address indexed investigator_id,
        string  query_type,
        bytes32 profile_hash,
        uint256 timestamp
    );

    event InvestigatorSuspended(address indexed investigator_id, uint256 timestamp);
    event TokenGranted(address indexed investigator_id);
    event TokenRevoked(address indexed investigator_id);
    event InvestigatorReinstated(address indexed investigator_id);

    // ========================= Modifiers =========================

    modifier onlyOwner() {
        require(msg.sender == owner, "ForensicAudit: caller is not the owner");
        _;
    }

    modifier onlyAuthorized() {
        require(active_tokens[msg.sender], "ForensicAudit: no valid access token");
        require(
            investigator_status[msg.sender] == Status.ACTIVE,
            "ForensicAudit: investigator is SUSPENDED"
        );
        _;
    }

    // ======================== Constructor ========================

    constructor() {
        owner = msg.sender;
    }

    // ====================== Core -- logQuery ======================

    /**
     * @notice Record a DNA search query on-chain.
     * @param _queryType  Descriptor of the search operation (e.g. "STR_MATCH", "KINSHIP").
     * @param _profileHash  Keccak-256 hash of the queried genomic profile.
     * @dev Automatically suspends the caller if > 5 calls land within 60 s.
     */
    function logQuery(
        string calldata _queryType,
        bytes32 _profileHash
    ) external onlyAuthorized {
        // == Rate-limit check ==
        RateLimitState storage rl = _rateLimit[msg.sender];
        uint256 oldestInWindow    = rl.timestamps[rl.index];

        if (oldestInWindow != 0 && block.timestamp - oldestInWindow <= WINDOW) {
            // All 5 slots filled within the window -> suspend
            investigator_status[msg.sender] = Status.SUSPENDED;
            emit InvestigatorSuspended(msg.sender, block.timestamp);
            revert("ForensicAudit: rate limit exceeded -- investigator SUSPENDED");
        }

        // Write current timestamp into ring buffer
        rl.timestamps[rl.index] = block.timestamp;
        rl.index = (rl.index + 1) % MAX_CALLS;

        // == Persist log entry ==
        LogEntry memory entry = LogEntry({
            investigator_id: msg.sender,
            query_type:      _queryType,
            profile_hash:    _profileHash,
            timestamp:       block.timestamp
        });

        logs.push(entry);

        emit QueryLogged(logs.length - 1, msg.sender, _queryType, _profileHash, block.timestamp);
    }

    // ==================== Admin -- Token Management ===================

    function grantToken(address _investigator) external onlyOwner {
        active_tokens[_investigator]       = true;
        investigator_status[_investigator] = Status.ACTIVE;
        emit TokenGranted(_investigator);
    }

    function revokeToken(address _investigator) external onlyOwner {
        active_tokens[_investigator] = false;
        emit TokenRevoked(_investigator);
    }

    function reinstateInvestigator(address _investigator) external onlyOwner {
        require(
            investigator_status[_investigator] == Status.SUSPENDED,
            "ForensicAudit: investigator is not suspended"
        );
        investigator_status[_investigator] = Status.ACTIVE;

        // Reset their rate-limit ring buffer
        delete _rateLimit[_investigator];

        emit InvestigatorReinstated(_investigator);
    }

    // ======================== View Helpers ========================

    function getLogCount() external view returns (uint256) {
        return logs.length;
    }

    function getLog(uint256 _index) external view returns (LogEntry memory) {
        require(_index < logs.length, "ForensicAudit: index out of bounds");
        return logs[_index];
    }
}
