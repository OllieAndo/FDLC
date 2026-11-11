# FDLC: Fault Detection, Classification & Localisation in DER-rich distribution grids

PhD research code for MSCA DC10 *Agile node-level fault detection and classification*
(SAILING Doctoral Network, IREC / UPC).

## Research overview

Inverter-based resources (IBRs) limit fault current and respond according to their control
software, not just network impedance. This breaks the assumptions behind classical overcurrent
protection and fault passage indicators. FDLC builds a tiered, data-driven pipeline that
explicitly differentiates fault behaviour by **asset class** (IBR, rotating, static):

| Tier | Task | Candidate methods |
|---|---|---|
| 1 | Lightweight edge **detection** | Echo State Network, Isolation Forest, ELM (online-adaptable baselines) |
| 2 | Explainable **classification** per asset class | 1D-CNN + SHAP |
| 3 | System-level **localisation** (stretch) | Topology-aware GNN, optionally travelling-wave physics loss |

Cross-cutting evaluation criteria: reliability (calibration, OOD / abstention), transferability
(unseen grids and control strategies) and drift (DER growth, reconfiguration).

## Pipeline

```text
grid (SimBench / IEEE 33, 123) -> VeraGrid EMT/RMS fault scenarios -> node waveforms
    -> features -> Tier 1 / 2 / 3 models -> explanation -> evaluation
```

## Repository structure

```text
FDLC/
├── data/                     # not tracked: raw grids, simulated waveforms, processed sets
├── notebooks/                # exploration
├── scripts/                  # runnable experiments (e.g. emt_slg_fault.py)
├── src/fdlc/
│   ├── simulation/           # VeraGrid grid building, fault scenario generation
│   ├── features/             # sequence components, Clarke transform, wavelets
│   ├── models/
│   │   ├── tier1/            # edge detectors
│   │   ├── tier2/            # asset-class classifiers
│   │   └── tier3/            # GNN localisation
│   ├── explain/              # SHAP and physics consistency checks
│   ├── eval/                 # metrics, calibration, transfer / drift experiments
│   └── utils/
├── tests/
└── pyproject.toml
```

## Stack

- Simulation: [VeraGrid](https://veragrid.readthedocs.io/) (RMS + EMT), pandapower / SimBench for grid data and static cross-checks
- ML: PyTorch, scikit-learn, SHAP; PyTorch Geometric for Tier 3 (optional extra)
- Validation reference: MATLAB/Simulink

## Setup

Python 3.12 recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"         # add ,gnn for PyTorch Geometric
pytest
python scripts/emt_slg_fault.py
```

## Status and licence

Early-stage research code. All rights reserved; no licence is granted to use, copy or
redistribute. IP terms are subject to the IREC / SAILING agreements.
