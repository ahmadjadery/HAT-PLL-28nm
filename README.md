# HAT-PLL-28nm: A Hardware-Aware Training Framework for Resilient Mixed-Signal Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official source code and data for the paper: **"Hardware-Aware Training Methodology for a Resilient Mixed-Signal System: A 28nm Cognitive PLL Design Case Study"**.

The goal of this repository is to provide a transparent and reproducible foundation for our work. It includes the core algorithmic components of our Hardware-Aware Training (HAT) methodology, a simplified behavioral PLL environment for demonstration, and the data required to reproduce the key statistical results from our paper.

---

## Highlights

*   **Core Logic:** A clean Python implementation of the key algorithmic contributions:
    *   The **`stochastic_forward_pass`** function, which models hardware non-idealities.
    *   The **`reward_function`** engineered for PLL stability.
    *   A simplified behavioral **PLL environment** (`pll_env.py`) for rapid prototyping.
*   **Reproducibility:** A Jupyter notebook (`analysis.ipynb`) and pre-computed Monte Carlo data that allow for one-click reproduction of the key figures and tables from the paper.
*   **Demonstration:** A training script (`train.py`) to demonstrate a short training loop of the HAT agent.

---

## Getting Started

### Prerequisites

- **Conda:** This project uses Conda to manage its dependencies. Please ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YourUsername/HAT-PLL-28nm.git
    cd HAT-PLL-28nm
    ```

2.  **Create and activate the Conda environment:**
    This single command will install all necessary dependencies specified in the `environment.yml` file.
    ```bash
    conda env create -f environment.yml
    conda activate hat_pll_env
    ```

---

## Usage

This repository supports two main use cases:

### 1. Reproducing Key Results from the Paper

This is the primary use case. The pre-computed Monte Carlo data is already included in the `data/` folder.

1.  **Launch Jupyter Lab:**
    ```bash
    jupyter lab
    ```
2.  **Open the Notebook:** Navigate to the `notebooks/` directory and open `analysis.ipynb`.
3.  **Run All Cells:** Execute all cells in the notebook. This will load the data and regenerate the Monte Carlo jitter histogram (Fig. 4) and the key benchmarking calculations from Table II.

### 2. Running a Training Demonstration

This script runs a very short training session to demonstrate how the agent learns over a few episodes in the simplified environment. **It is for demonstration purposes only and does not reproduce the final agent from the paper.**

To run the training demo, execute the following command:
```bash
python src/train.py
```
You will see the episodic rewards printed to the console, showing that the agent is learning to improve its performance over time.

```
## Code Overview (Repository Structure)
HAT-PLL-28nm/
│
├── data/
│ └── monte_carlo_jitter.csv # Pre-computed jitter data from 1,000 Monte Carlo runs.
│
├── notebooks/
│ └── analysis.ipynb # Jupyter notebook for reproducing paper results.
│
├── src/
│ ├── pll_env.py # A simplified, behavioral model of the PLL. The "gym" for our agent.
│ ├── reward_function.py # Implements the mathematical reward function from Appendix A.
│ ├── stochastic_forward_pass.py # The core of the HAT methodology: simulates hardware noise.
│ ├── hat_td3.py # A skeleton of the TD3 agent showing where HAT is applied.
│ └── train.py # The main script that pieces everything together for a training demo.
│
├── environment.yml # Conda environment definition for one-step installation.
├── LICENSE # MIT License file.
└── README.md # This documentation.
-   **Why so many files in `src/`?** We have modularized the code for clarity. The `train.py` script acts as the conductor, using the `pll_env` as the stage, the `hat_td3` agent as the actor, and the `reward_function` as the script. The `stochastic_forward_pass` is the "special effect" that makes our agent's training unique.



## Citation

If you use the concepts, methodology, or code from this work in your research, please cite our paper.

```bibtex
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
---
