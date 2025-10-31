# HAT-PLL-28nm: A Hardware-Aware Training Framework for Resilient Mixed-Signal Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the supplementary code and data for the paper: **"Hardware-Aware Training Methodology for a Resilient Mixed-Signal System: A 28nm Cognitive PLL Design Case Study."**

The goal of this repository is to provide a transparent and reproducible foundation for our work. It includes the core algorithmic components of our Hardware-Aware Training (HAT) methodology and a simplified, behavioral PLL environment for demonstration and training.

---

## Prerequisites

- **Conda:** This project uses Conda to manage its dependencies. Please ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

---

## Installation

You can set up the complete Python environment required to run all the code in this repository with a single command:

```bash
conda env create -f environment.yml
This will create a new Conda environment named hat_pll_env. To activate it, run:

```bash
conda activate hat_pll_env
Usage
1. Reproducing Paper Results
To reproduce the statistical analysis and key figures from the paper (e.g., Fig. 4 - Monte Carlo histogram), you can run the Jupyter notebook provided:
```bash
-Activate the Conda environment: conda activate hat_pll_env
-Generat monti_carlo_jitter data : python generate_data.py
-Start Jupyter Lab: jupyter lab
Open and run the cells in notebooks/analysis.ipynb.

This notebook will load the pre-computed Monte Carlo data from the data/ directory and regenerate the plots and tables.
2. Running a New Training Session (Demonstration)
To demonstrate the training process of the HAT agent on the simplified PLL environment, run the main training script:
code
Bash
python src/train.py
(Note: This will run a short training session for demonstration purposes. Full training takes several hours as detailed in the paper.)
Repository Structure
code
Code
HAT-PLL-28nm/
│
├── data/
│   └── monte_carlo_jitter.csv  # Pre-computed data for 1,000 MC jitter samples
│
├── notebooks/
│   └── analysis.ipynb          # Jupyter notebook to reproduce paper figures and tables
│
├── src/
│   ├── hat_td3.py              # Simplified implementation of the HAT-TD3 agent
│   ├── stochastic_forward_pass.py # Core function modeling hardware non-idealities
│   ├── reward_function.py      # Implementation of the PLL reward function
│   ├── pll_env.py              # Simplified behavioral model of the PLL environment
│   └── train.py                # Main script to run a training demonstration
│
├── environment.yml             # Conda environment definition for one-step installation
├── LICENSE                     # MIT License file
└── README.md                   # This file
Citation
If you use the concepts, methodology, or code from this work in your research, please cite our paper:
code
Bibtex
@article{Jadery2025HAT,
  author  = {Ahmad Jadery and Elias Rachid and Mehdi Ehsanian and Zeinab Hammoud and Adnan Harb},
  title   = {Hardware-Aware Training Methodology for a Resilient Mixed-Signal System: A 28nm Cognitive PLL Design Case Study},
  journal = {IEEE Journal of Solid-State Circuits (JSSC)},
  year    = {2025},
  volume  = {XX},
  number  = {Y},
  pages   = {ZZZ-ZZZ},
  doi     = {Your_Paper_DOI_Here}
}
License
This project is licensed under the MIT License. See the LICENSE file for details.
