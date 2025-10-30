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
```**النقد الذاتي (لـ `reward_function.py`):**
*   ✅ **مطابق:** الكود يطبق المعادلة والمنطق من `Appendix A.1` حرفيًا.
*   ✅ **واضح:** استخدام متغيرات بأسماء واضحة (`r_lock`, `r_phase`) والتعليقات تجعل الكود سهل الفهم.
*   ✅ **صحيح تقنيًا:** يعالج المدخلات كـ batches (مجموعات)، وهو ما يتوقعه إطار عمل التعلم الآلي.

---
**الملف 3: `hat_td3.py` (نسخة مبسطة توضح الفكرة)**

```python
# src/hat_td3.py

from stochastic_forward_pass import stochastic_forward_pass
# Assume Actor and Critic network classes are defined elsewhere
# Assume a ReplayBuffer class is defined elsewhere

class HAT_TD3_Agent:
    def __init__(self, actor_network, critic_network1, critic_network2):
        self.actor = actor_network
        self.critic1 = critic_network1
        self.critic2 = critic_network2
        # ... other initializations (target networks, optimizers, etc.)

    def train(self, replay_buffer, batch_size):
        """
        A simplified training step to demonstrate the key HAT modification.
        """
        # 1. Sample a batch of transitions from the replay buffer
        state, action, next_state, reward, done = replay_buffer.sample(batch_size)

        # --- THIS IS THE CORE OF HAT METHODOLOGY in the Critic path ---
        # The key difference from standard TD3 is the use of the stochastic
        # forward pass when evaluating the networks.
        
        # 2. Compute target Q-value
        with torch.no_grad():
            # Select action according to the target policy network
            # NOTE: We use the *stochastic forward pass* for the actor network
            next_action_stochastic = self._stochastic_actor_forward(self.actor_target, next_state)
            
            # Add clipped noise for target policy smoothing
            # ... (standard TD3 noise addition) ...
            
            # Compute the target Q value using the *stochastic forward pass* for critic
            target_q1_stochastic = self._stochastic_critic_forward(self.critic_target1, next_state, next_action_stochastic)
            target_q2_stochastic = self._stochastic_critic_forward(self.critic_target2, next_state, next_action_stochastic)
            target_q = torch.min(target_q1_stochastic, target_q2_stochastic)
            target_q = reward + (1 - done) * discount_factor * target_q

        # 3. Get current Q estimates using *stochastic forward pass*
        current_q1 = self._stochastic_critic_forward(self.critic1, state, action)
        current_q2 = self._stochastic_critic_forward(self.critic2, state, action)
        
        # 4. Compute critic loss and update critics
        critic_loss = torch.functional.F.mse_loss(current_q1, target_q) + \
                      torch.functional.F.mse_loss(current_q2, target_q)
        # ... (critic update step) ...
        
        # 5. Delayed policy update using *stochastic forward pass*
        if total_it % policy_freq == 0:
            # Compute actor loss
            actor_actions_stochastic = self._stochastic_actor_forward(self.actor, state)
            actor_loss = -self._stochastic_critic_forward(self.critic1, state, actor_actions_stochastic).mean()
            
            # ... (actor update step) ...

    def _stochastic_actor_forward(self, actor_network, state):
        # Pass through each linear layer using the stochastic function
        # This is a simplified representation of a multi-layer network
        x = state
        for layer in actor_network.layers:
            if isinstance(layer, nn.Linear):
                x = stochastic_forward_pass(layer, x)
            else: # Apply activations (e.g., ReLU, Tanh)
                x = layer(x)
        return x
        
    def _stochastic_critic_forward(self, critic_network, state, action):
        # Similar to actor, but with state and action as input
        x = torch.cat([state, action], 1)
        for layer in critic_network.layers:
            if isinstance(layer, nn.Linear):
                x = stochastic_forward_pass(layer, x)
            else:
                x = layer(x)
        return x
