Markdown
# Grid Fault Identification and Restoration Research

Research repository for investigating Agile Fault Detection & Classification (Possibly Restoration) using ELM's, XGBoost, CNN's and Graph Neural Network (GNN) architectures for agile fault identification, classification.
Possibly extending to fault restoration where Decomposition Architectures could be used.

## Research Overview

Modern electrical distribution grids face increasing operational complexity with high penetration of distributed energy resources. Traditional protection and restoration schemes rely on rigid rule-based logic or static heuristics. This repository contains the experimental framework for developing data-driven, topology-aware models that perform:
1. **Agile Fault Identification & Classification:** Utilising ML models for edge detection and spatial-temporal GNNs to rapidly classify, and localise anomalies across distribution networks.
2. **Optimal Service Restoration:** Formulating MILP-based optimization models (via OR-Tools/SCIP) constrained by network radiality and thermal limits to generate safe switching sequences.

## Repository Structure

```text
grid-brain/
├── data/                  # Raw grid topologies (IEEE test cases) and simulated datasets
├── notebooks/             # Exploratory analysis and simulation notebooks
├── src/                   # Core research codebase
│   ├── environment/       # Pandapower wrapper and network topology extraction
│   ├── models/            # PyTorch Geometric GNN architectures
│   ├── optimisation/      # MILP restoration and constraints formulation
│   └── utils/             # Evaluation metrics and configuration utilities
├── tests/                 # Unit tests for physics verification and constraint validation
├── pyproject.toml         # Dependency management and package configuration
└── README.md
Core Tech Stack
Power System Simulation: Pandapower (Newton-Raphson power flow calculations)

Graph Deep Learning: PyTorch & PyTorch Geometric (Spatial-temporal network modeling)

Mathematical Optimization: Google OR-Tools (MILP/CP-SAT solvers for topology optimization)

Installation
Clone the repository:

Bash
git clone [https://github.com/your-username/grid-brain.git](https://github.com/your-username/grid-brain.git)
cd grid-brain
Development Setup
Install the package in editable mode along with dependencies:

Bash
pip install -e .
