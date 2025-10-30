# src/pll_env.py

import numpy as np
from reward_function import calculate_reward
import torch

class SimplePLLEnv:
    """
    A simplified, behavioral model of a Type-II Charge-Pump PLL.
    This environment is designed for rapid reinforcement learning prototyping
    and follows a structure similar to OpenAI Gym. The parameters are
    derived from the nominal TT corner values in the main paper's supplement.
    """
    def __init__(self, dt=1e-9):  # Default timestep of 1 ns
        # --- Physical Constants (from paper supplement S1.1, nominal) ---
        self.F_REF = 100e6  # Reference frequency: 100 MHz (Assumed for N=80)
        self.K_VCO = 1.8e9   # VCO gain: 1.8 GHz/V
        self.I_CP_NOMINAL = 500e-6  # Nominal charge pump current: 500 µA
        self.R_P = 820      # Loop filter resistance: 820 Ω (R1)
        self.C_P = 47e-12   # Loop filter capacitance: 47 pF (C1)
        self.N = 80         # Division ratio for 8 GHz output

        # Simulation parameters
        self.dt = dt  # Simulation timestep

        # State variables
        self.phase_error = 0.0
        self.v_ctrl = 1.0  # Initial control voltage
        self.lock_status = 1
        
        # We need to maintain a state tensor for the reward function
        self.state_tensor = self._get_state_tensor()
        self.prev_state_tensor = self.state_tensor.clone()

    def reset(self):
        """Resets the environment to a random initial state."""
        self.phase_error = np.random.uniform(-np.pi, np.pi)
        self.v_ctrl = np.random.uniform(0.5, 1.2)
        # Randomize target frequency slightly to improve robustness
        self.target_freq = self.F_REF * (self.N + np.random.randint(-2, 2))
        
        self.state_tensor = self._get_state_tensor()
        self.prev_state_tensor = self.state_tensor.clone()

        return self.state_tensor

    def step(self, action: np.ndarray):
        """
        Executes one timestep in the environment.
        Args:
            action (np.ndarray): A 3-element normalized action vector [-1, 1].
        """
        self.prev_state_tensor = self.state_tensor.clone()

        # --- 1. Decode Action ---
        # Map normalized action [-1, 1] to physical parameter changes
        # For simplicity, we only control I_cp. Kp/Ki are related to filter components.
        icp_modulation = action[0] 
        i_cp = self.I_CP_NOMINAL * (1 + 0.5 * icp_modulation) # Allow +/- 50% modulation

        # --- 2. Update PLL State (Discrete-time simulation) ---
        # Update charge pump current into the filter
        filter_current = (i_cp / (2 * np.pi)) * self.phase_error

        # Update loop filter state (simplified RC filter)
        # v_ctrl_dot = (filter_current - self.v_ctrl / self.R_P) / self.C_P
        # A simpler discrete update:
        self.v_ctrl += filter_current * self.dt / self.C_P # Dominant pole at zero

        # --- 3. Update VCO and Phase ---
        vco_freq = self.K_VCO * self.v_ctrl
        freq_error = self.target_freq - vco_freq
        
        # Integrate frequency error to get phase error
        self.phase_error += 2 * np.pi * freq_error * self.dt
        # Keep phase error wrapped between -pi and pi for stability
        self.phase_error = np.mod(self.phase_error + np.pi, 2 * np.pi) - np.pi

        # --- 4. Determine Lock Status and Termination ---
        self.lock_status = 1 if np.abs(self.phase_error) < (np.pi / 4) else 0
        done = bool(np.abs(self.phase_error) > (np.pi * 0.9)) # End episode if lock is badly lost

        # --- 5. Assemble Next State, Reward ---
        self.state_tensor = self._get_state_tensor()
        reward = calculate_reward(self.state_tensor.unsqueeze(0), self.prev_state_tensor.unsqueeze(0))

        return self.state_tensor, reward, done

    def _get_state_tensor(self) -> torch.Tensor:
        """Assembles a simplified version of the 13-dim state vector."""
        # For this simple environment, we only populate the critical components
        # A full implementation would populate all 13.
        # [phase_err, v_ctrl, 0, lock, N_t, N_c, temp, vdd, act...]
        state = torch.zeros(13)
        state[0] = self.phase_error / np.pi # Normalize
        state[1] = (self.v_ctrl - 0.6) / 0.6 # Normalize around a midpoint
        state[3] = self.lock_status
        return state
