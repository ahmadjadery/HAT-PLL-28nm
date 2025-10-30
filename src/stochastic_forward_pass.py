# src/stochastic_forward_pass.py

import torch
import torch.nn as nn

def stochastic_forward_pass(linear_layer: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """
    Implements the f_physical stochastic forward pass to model hardware non-idealities.
    This function replaces a standard torch.nn.Linear forward pass during HAT training.

    Args:
        linear_layer (nn.Linear): The PyTorch linear layer whose weights will be used.
        x (torch.Tensor): The input tensor to the layer.

    Returns:
        torch.Tensor: The output tensor after simulating physical effects.
    """
    
    # --- 1. Interconnect Parasitics (IR Drop) ---
    # Simplified model: a small, systematic voltage drop proportional to input magnitude.
    # A real model would be position-dependent, but this captures the first-order effect.
    ir_drop_factor = 0.02 # Assuming a 2% drop at max input
    x_eff = x - (x * ir_drop_factor)

    # --- 2. Synaptic Device Variability (ReRAM Noise) ---
    # Weights are perturbed by a log-normal multiplicative noise.
    # The parameters (mu, sigma) of the underlying normal distribution are derived
    # from Appendix B of the main paper.
    weights = linear_layer.weight
    # Parameters for the underlying Normal distribution of ln(R)
    # Using average parameters for demonstration simplicity
    mu_lnR = torch.tensor(0.0, device=weights.device) 
    sigma_lnR = torch.tensor(0.25, device=weights.device) # Represents a significant variability
    
    # Create log-normal noise with the same shape as the weights
    log_normal_noise = torch.log_normal_(torch.empty_like(weights), mean=mu_lnR, std=sigma_lnR)
    
    # Ensure the mean of the noise is 1 to not systematically shift the weights
    log_normal_noise = log_normal_noise / torch.exp(mu_lnR + (sigma_lnR**2)/2)
    
    w_noisy = weights * log_normal_noise

    # --- 3. Matrix Multiplication ---
    # Perform the MVM with the perturbed inputs and weights
    output_analog = torch.functional.F.linear(x_eff, w_noisy, linear_layer.bias)

    # --- 4. Readout Path Noise ---
    # Add a small amount of Gaussian additive noise to model TIA thermal noise.
    readout_noise_std = 0.05 # Represents 5% of the signal range as noise
    readout_noise = torch.randn_like(output_analog) * readout_noise_std
    output_with_noise = output_analog + readout_noise
    
    # --- 5. ADC Quantization ---
    # Simulate an 8-bit Analog-to-Digital Converter.
    # Normalize the output to the [-1, 1] range, quantize, then de-quantize.
    num_levels = 2**8
    # Clamp the output to a nominal range to prevent saturation
    output_clamped = torch.clamp(output_with_noise, -1.0, 1.0)
    # Quantize and de-quantize
    quantized_output = torch.round((output_clamped + 1) * (num_levels - 1) / 2)
    dequantized_output = 2 * quantized_output / (num_levels - 1) - 1
    
    return dequantized_output
