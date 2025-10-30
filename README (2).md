{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1fdd319f-46fb-4cd7-b25c-b3bd24c1f26e",
   "metadata": {},
   "source": [
    "# HAT-PLL-28nm: A Hardware-Aware Training Framework for Resilient Mixed-Signal Systems\n",
    "\n",
    "[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)\n",
    "\n",
    "This repository contains the supplementary code and data for the paper: **\"Hardware-Aware Training Methodology for a Resilient Mixed-Signal System: A 28nm Cognitive PLL Design Case Study.\"**\n",
    "\n",
    "The goal of this repository is to provide a transparent and reproducible foundation for our work. It includes the core algorithmic components of our Hardware-Aware Training (HAT) methodology and a simplified, behavioral PLL environment for demonstration and training.\n",
    "\n",
    "---\n",
    "\n",
    "## Prerequisites\n",
    "\n",
    "- **Conda:** This project uses Conda to manage its dependencies. Please ensure you have [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.\n",
    "\n",
    "---\n",
    "\n",
    "## Installation\n",
    "\n",
    "You can set up the complete Python environment required to run all the code in this repository with a single command:\n",
    "\n",
    "```bash\n",
    "conda env create -f environment.yml"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28d06bbe-0ea0-4dca-9ff6-d3b65ec67968",
   "metadata": {},
   "source": [
    "This will create a new Conda environment named hat_pll_env. To activate it, run:\n",
    "\n",
    "```bash\n",
    "conda activate hat_pll_env"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6838d4a7-ffe7-4ea6-8cf5-c7baf2a25035",
   "metadata": {},
   "source": [
    "Usage\n",
    "1. Reproducing Paper Results\n",
    "To reproduce the statistical analysis and key figures from the paper (e.g., Fig. 4 - Monte Carlo histogram), you can run the Jupyter notebook provided:\n",
    "Activate the Conda environment: conda activate hat_pll_env\n",
    "Start Jupyter Lab: jupyter lab\n",
    "Open and run the cells in notebooks/analysis.ipynb.\n",
    "This notebook will load the pre-computed Monte Carlo data from the data/ directory and regenerate the plots and tables.\n",
    "2. Running a New Training Session (Demonstration)\n",
    "To demonstrate the training process of the HAT agent on the simplified PLL environment, run the main training script:\n",
    "\n",
    "```bash\n",
    "python src/train.py\n",
    "(Note: This will run a short training session for demonstration purposes. Full training takes several hours as detailed in the paper.)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f2ddc71-628f-48af-9526-1a50cef49bb5",
   "metadata": {},
   "source": [
    "Repository Structure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1000831-db1c-4b0c-9fea-4c623b208764",
   "metadata": {},
   "outputs": [],
   "source": [
    "HAT-PLL-28nm/\n",
    "│\n",
    "├── data/\n",
    "│   └── monte_carlo_jitter.csv  # Pre-computed data for 1,000 MC jitter samples\n",
    "│\n",
    "├── notebooks/\n",
    "│   └── analysis.ipynb          # Jupyter notebook to reproduce paper figures and tables\n",
    "│\n",
    "├── src/\n",
    "│   ├── hat_td3.py              # Simplified implementation of the HAT-TD3 agent\n",
    "│   ├── stochastic_forward_pass.py # Core function modeling hardware non-idealities\n",
    "│   ├── reward_function.py      # Implementation of the PLL reward function\n",
    "│   ├── pll_env.py              # Simplified behavioral model of the PLL environment\n",
    "│   └── train.py                # Main script to run a training demonstration\n",
    "│\n",
    "├── environment.yml             # Conda environment definition for one-step installation\n",
    "├── LICENSE                     # MIT License file\n",
    "└── README.md                   # This file"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "934ea6fe-2277-4fd9-83dd-6e925fa1999d",
   "metadata": {},
   "outputs": [],
   "source": [
    "Citation\n",
    "If you use the concepts, methodology, or code from this work in your research, please cite our paper:\n",
    "code\n",
    "Bibtex\n",
    "@article{Jadery2025HAT,\n",
    "  author  = {Ahmad Jadery and Elias Rachid and Mehdi Ehsanian and Zeinab Hammoud and Adnan Harb},\n",
    "  title   = {Hardware-Aware Training Methodology for a Resilient Mixed-Signal System: A 28nm Cognitive PLL Design Case Study},\n",
    "  journal = {IEEE Journal of Solid-State Circuits (JSSC)},\n",
    "  year    = {2025},\n",
    "  volume  = {XX},\n",
    "  number  = {Y},\n",
    "  pages   = {ZZZ-ZZZ},\n",
    "  doi     = {Your_Paper_DOI_Here}\n",
    "}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "21b9f9b4-b0fe-4903-bc95-5ece813944ca",
   "metadata": {},
   "source": [
    "\n",
    "License\n",
    "This project is licensed under the MIT License. See the LICENSE file for details."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7186778b-9671-4bc0-bc25-81b8c14e2073",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
