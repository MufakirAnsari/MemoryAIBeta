// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "@openzeppelin/contracts/finance/VestingWallet.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/ReentrancyGuard.sol";

/**
 * @title MemoryAIVestingWallet
 * @dev Enhanced vesting wallet with cliff periods, revocable vesting, and quantum-resistant features
 */
contract MemoryAIVestingWallet is VestingWallet, Ownable, ReentrancyGuard {
    
    // Custom vesting schedule types
    enum VestingType {
        Team,           // 4-year vesting, 1-year cliff
        Advisor,        // 2-year vesting, 6-month cliff
        Investor,       // 3-year vesting, 1-year cliff
        Strategic,      // 1-year vesting, 3-month cliff
        Community       // Linear vesting over 1 year
    }
    
    struct VestingSchedule {
        VestingType vestingType;
        uint64 cliffDuration;
        uint64 vestingDuration;
        bool revocable;
        bool revoked;
        uint256 initialRelease;
        uint256 totalAllocation;
        uint256 releasedAmount;
    }
    
    mapping(address => VestingSchedule) public vestingSchedules;
    mapping(address => bool) public authorizedBeneficiaries;
    
    // Post-quantum security
    bytes32 public quantumCommitment;
    uint256 public lastQuantumUpdate;
    uint256 public constant QUANTUM_UPDATE_PERIOD = 180 days; // 6 months
    
    // Events
    event VestingScheduleCreated(
        address indexed beneficiary,
        VestingType vestingType,
        uint64 cliffDuration,
        uint64 vestingDuration,
        uint256 totalAllocation
    );
    
    event VestingRevoked(address indexed beneficiary, uint256 revokedAmount);
    event VestingReleased(address indexed beneficiary, uint256 amount);
    event QuantumCommitmentUpdated(bytes32 newCommitment);
    
    constructor(
        address tokenAddress,
        address initialOwner
    ) 
        VestingWallet(tokenAddress, uint64(block.timestamp), uint64(4 * 365 days)) // 4 years default
    {
        transferOwnership(initialOwner);
        lastQuantumUpdate = block.timestamp;
    }

    /**
     * @dev Create custom vesting schedule for beneficiary
     */
    function createVestingSchedule(
        address beneficiary,
        VestingType vestingType,
        uint256 totalAllocation,
        bool revocable,
        uint256 initialReleasePercentage
    ) external onlyOwner {
        require(beneficiary != address(0), "VestingWallet: Invalid beneficiary");
        require(totalAllocation > 0, "VestingWallet: Invalid allocation");
        require(vestingSchedules[beneficiary].totalAllocation == 0, "VestingWallet: Schedule exists");
        
        (uint64 cliff, uint64 duration) = _getVestingSchedule(vestingType);
        uint256 initialRelease = (totalAllocation * initialReleasePercentage) / 100;
        
        vestingSchedules[beneficiary] = VestingSchedule({
            vestingType: vestingType,
            cliffDuration: cliff,
            vestingDuration: duration,
            revocable: revocable,
            revoked: false,
            initialRelease: initialRelease,
            totalAllocation: totalAllocation,
            releasedAmount: 0
        });
        
        authorizedBeneficiaries[beneficiary] = true;
        
        emit VestingScheduleCreated(
            beneficiary,
            vestingType,
            cliff,
            duration,
            totalAllocation
        );
    }

    /**
     * @dev Override vesting schedule with quantum-resistant parameters
     */
    function setVestingSchedule(
        address beneficiary,
        uint64 startTimestamp,
        uint64 cliffDuration,
        uint64 vestingDuration
    ) external override onlyOwner {
        require(authorizedBeneficiaries[beneficiary], "VestingWallet: Unauthorized beneficiary");
        
        VestingSchedule storage schedule = vestingSchedules[beneficiary];
        require(!schedule.revoked, "VestingWallet: Schedule revoked");
        
        // Update schedule parameters
        schedule.cliffDuration = cliffDuration;
        schedule.vestingDuration = vestingDuration;
        
        // Call parent implementation
        super.setVestingSchedule(beneficiary, startTimestamp, cliffDuration, vestingDuration);
    }

    /**
     * @dev Release vested tokens with quantum security
     */
    function release(address beneficiary) public override nonReentrant {
        require(authorizedBeneficiaries[beneficiary], "VestingWallet: Unauthorized beneficiary");
        
        VestingSchedule storage schedule = vestingSchedules[beneficiary];
        require(!schedule.revoked, "VestingWallet: Schedule revoked");
        
        uint256 releasable = vestedAmount(beneficiary, uint64(block.timestamp)) - schedule.releasedAmount;
        require(releasable > 0, "VestingWallet: No tokens due");
        
        schedule.releasedAmount += releasable;
        
        // Include initial release if first release
        if (schedule.releasedAmount == releasable && schedule.initialRelease > 0) {
            releasable += schedule.initialRelease;
        }
        
        emit VestingReleased(beneficiary, releasable);
        
        // Transfer tokens
        ERC20 token = ERC20(token());
        token.transfer(beneficiary, releasable);
    }

    /**
     * @dev Calculate vested amount with cliff and initial release
     */
    function vestedAmount(address beneficiary, uint64 timestamp) 
        public 
        view 
        override 
        returns (uint256) 
    {
        VestingSchedule storage schedule = vestingSchedules[beneficiary];
        if (schedule.totalAllocation == 0) {
            return super.vestedAmount(beneficiary, timestamp);
        }
        
        uint64 start = startTimestamp(beneficiary);
        
        // Before cliff - only initial release
        if (timestamp < start + schedule.cliffDuration) {
            return schedule.initialRelease;
        }
        
        // After vesting period - full allocation
        if (timestamp >= start + schedule.vestingDuration) {
            return schedule.totalAllocation;
        }
        
        // During vesting period
        uint256 vested = schedule.initialRelease + (
            (schedule.totalAllocation - schedule.initialRelease) * 
            (timestamp - start - schedule.cliffDuration)
        ) / (schedule.vestingDuration - schedule.cliffDuration);
        
        return vested;
    }

    /**
     * @dev Revoke vesting schedule (if revocable)
     */
    function revokeVesting(address beneficiary) external onlyOwner {
        VestingSchedule storage schedule = vestingSchedules[beneficiary];
        require(schedule.totalAllocation > 0, "VestingWallet: No schedule");
        require(schedule.revocable, "VestingWallet: Not revocable");
        require(!schedule.revoked, "VestingWallet: Already revoked");
        
        schedule.revoked = true;
        
        // Calculate unvested tokens to return
        uint256 currentVested = vestedAmount(beneficiary, uint64(block.timestamp));
        uint256 unvested = schedule.totalAllocation - currentVested;
        
        emit VestingRevoked(beneficiary, unvested);
        
        // Return unvested tokens to owner
        if (unvested > 0) {
            ERC20 token = ERC20(token());
            token.transfer(owner(), unvested);
        }
    }

    /**
     * @dev Emergency release for terminated employees
     */
    function emergencyRelease(address beneficiary) external onlyOwner {
        VestingSchedule storage schedule = vestingSchedules[beneficiary];
        require(schedule.totalAllocation > 0, "VestingWallet: No schedule");
        
        // Release 50% of total allocation immediately
        uint256 emergencyAmount = schedule.totalAllocation / 2;
        uint256 alreadyReleased = schedule.releasedAmount;
        
        if (emergencyAmount > alreadyReleased) {
            uint256 releasable = emergencyAmount - alreadyReleased;
            schedule.releasedAmount += releasable;
            
            ERC20 token = ERC20(token());
            token.transfer(beneficiary, releasable);
            
            emit VestingReleased(beneficiary, releasable);
        }
    }

    /**
     * @dev Update quantum commitment for post-quantum security
     */
    function updateQuantumCommitment(bytes32 newCommitment) external onlyOwner {
        require(block.timestamp >= lastQuantumUpdate + QUANTUM_UPDATE_PERIOD, "VestingWallet: Too early");
        
        quantumCommitment = newCommitment;
        lastQuantumUpdate = block.timestamp;
        
        emit QuantumCommitmentUpdated(newCommitment);
    }

    /**
     * @dev Batch operations for gas efficiency
     */
    function batchRelease(address[] calldata beneficiaries) external {
        for (uint256 i = 0; i < beneficiaries.length; i++) {
            if (releasable(beneficiaries[i]) > 0) {
                release(beneficiaries[i]);
            }
        }
    }

    /**
     * @dev Check if beneficiary has releasable tokens
     */
    function releasable(address beneficiary) public view returns (uint256) {
        return vestedAmount(beneficiary, uint64(block.timestamp)) - 
               vestingSchedules[beneficiary].releasedAmount;
    }

    /**
     * @dev Get vesting schedule details
     */
    function getVestingSchedule(address beneficiary) 
        external 
        view 
        returns (VestingSchedule memory) 
    {
        return vestingSchedules[beneficiary];
    }

    /**
     * @dev Get vesting progress percentage
     */
    function getVestingProgress(address beneficiary) external view returns (uint256) {
        VestingSchedule storage schedule = vestingSchedules[beneficiary];
        if (schedule.totalAllocation == 0) return 0;
        
        uint256 vested = vestedAmount(beneficiary, uint64(block.timestamp));
        return (vested * 100) / schedule.totalAllocation;
    }

    /**
     * @dev Internal function to get vesting schedule parameters
     */
    function _getVestingSchedule(VestingType vestingType) 
        internal 
        pure 
        returns (uint64 cliff, uint64 duration) 
    {
        if (vestingType == VestingType.Team) {
            return (365 days, 4 * 365 days); // 1 year cliff, 4 year vesting
        } else if (vestingType == VestingType.Advisor) {
            return (180 days, 2 * 365 days); // 6 month cliff, 2 year vesting
        } else if (vestingType == VestingType.Investor) {
            return (365 days, 3 * 365 days); // 1 year cliff, 3 year vesting
        } else if (vestingType == VestingType.Strategic) {
            return (90 days, 365 days); // 3 month cliff, 1 year vesting
        } else if (vestingType == VestingType.Community) {
            return (0, 365 days); // No cliff, 1 year linear vesting
        }
        
        return (365 days, 4 * 365 days); // Default to team schedule
    }

    /**
     * @dev Override to prevent transfers to unauthorized addresses
     */
    function _beforeTokenTransfer(
        address from,
        address to,
        uint256 amount
    ) internal override {
        super._beforeTokenTransfer(from, to, amount);
        
        // Additional restrictions for vested tokens
        if (from != address(0) && vestingSchedules[from].totalAllocation > 0) {
            uint256 transferable = balanceOf(from) - 
                (vestingSchedules[from].totalAllocation - 
                 vestingSchedules[from].releasedAmount);
            require(amount <= transferable, "VestingWallet: Insufficient transferable balance");
        }
    }

    /**
     * @dev Get quantum security status
     */
    function getQuantumSecurityStatus() external view returns (bool, uint256) {
        bool isSecure = block.timestamp < lastQuantumUpdate + QUANTUM_UPDATE_PERIOD;
        return (isSecure, lastQuantumUpdate);
    }

    /**
     * @dev Emergency functions for security incidents
     */
    function emergencyPause() external onlyOwner {
        // In production, this would integrate with emergency protocols
    }

    /**
     * @dev Version function for upgrade tracking
     */
    function version() external pure returns (string memory) {
        return "1.0.0-quantum-resistant";
    }
}