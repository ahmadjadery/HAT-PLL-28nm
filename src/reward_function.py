# src/reward_function.py

import torch

def calculate_reward(state: torch.Tensor, prev_state: torch.Tensor) -> float:
    """
    Calculates the reward for a given state transition, based on the function
    defined in Appendix A.1 of the main paper.

    Args:
        state (torch.Tensor): The current state vector (13 dimensions).
        prev_state (torch.Tensor): The previous state vector (for calculating V_ctrl delta).

    Returns:
        float: The calculated scalar reward.
    """
    
    # --- Unpack state vector components (indices are based on Appendix A.1) ---
    # Note: Assuming state tensors are batched, so we index the first dimension.
    phase_error = state[:, 0]
    v_ctrl = state[:, 1]
    lock_status = state[:, 3]
    
    prev_v_ctrl = prev_state[:, 1]

    # --- Define weights for reward components ---
    w_p = 50.0  # Phase penalty weight
    w_j = 20.0  # Jitter/stability penalty weight

    # --- 1. Lock Reward (r_lock) ---
    # Large penalty for being out of lock, small positive reward for being locked.
    r_lock = torch.where(lock_status == 1, 5.0, -100.0)

    # --- 2. Phase Deviation Penalty (r_phase) ---
    # Proportional to the squared phase error.
    r_phase = -w_p * (phase_error ** 2)
    
    # --- 3. Stability Penalty (r_jitter/r_stability) ---
    # Proportional to the change in V_ctrl as a proxy for jitter.
    r_stability = -w_j * torch.abs(v_ctrl - prev_v_ctrl)

    # --- Total Reward ---
    total_reward = r_lock + r_phase + r_stability
    
    # Return as a scalar if batch size is 1
    return total_reward.item() if total_reward.numel() == 1 else total_reward
