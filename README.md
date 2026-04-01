# DruGUI: Structure-Based Virtual Screening Pipeline for AI Agents

> 🔬 An end-to-end executable drug discovery workflow for AI agents.
> Input: PDB ID + candidate SMILES → Output: Ranked drug-like hits with docking scores, ADMET profiles, and synthesis accessibility

[![Claw4S 2026](https://img.shields.io/badge/Claw4S-2026-blue.svg)](https://claw.stanford.edu)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 👥 Authors

- **Max** — BioTender
- **Claw 🐞** — Claw4S Conference

## 🎯 What is DruGUI?

DruGUI automates the complete structure-based virtual screening (SBVS) workflow in a single executable skill for AI agents. No more juggling between PDBFixer, AutoDock Vina, RDKit, and separate ADMET tools — DruGUI串联 them all with full reproducibility guarantees.

## ✨ Features

- **End-to-end execution**: PDB ID + SMILES → ranked hit list in one command
- **Agent-native**: SKILL.md written for AI agents to execute without human intervention
- **Reproducible**: Pinned conda environment + SHA-256 checksums
- **Comprehensive**: Docking + ADMET + PAINS filtering + SA scoring
- **Open-source**: MIT license, community-driven

## 🚀 Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/DruGUI.git
cd DruGUI

# 2. Create environment
conda env create -f environment.yml
conda activate druGUI

# 3. Run the pipeline
python druGUI.py run \
    --pdb-id 6JX0 \
    --smiles-file examples/inputs/smiles_examples.txt \
    --output-dir ./output/egfr_screening \
    --top-k 20
```

## 📁 Output Files

```
output/egfr_screening/
├── final/
│   ├── final_report.json    # Machine-readable ranked hits
│   └── final_report.csv     # Tabular format
├── docking/docking_results.csv
├── admet/admet_results.csv
├── filters/passed_candidates.csv
└── sa_scores/sa_results.csv
```

## 🏗️ Pipeline Steps

| Step | Description |
|------|-------------|
| 1 | Environment setup (conda) |
| 2 | Target preparation (PDB download, fixing, protonation) |
| 3 | Ligand preparation (SMILES → 3D SDF) |
| 4 | Molecular docking (AutoDock Vina) |
| 5 | ADMET prediction |
| 6 | Lipinski + PAINS filtering |
| 7 | Synthesis accessibility scoring |
| 8 | Final ranking + report |

## 📊 Example Results

Screening 50 molecules against EGFR (PDB: 6JX0):

| Rank | Name | Vina Score | Composite | Lipinski | SA Score |
|------|------|-----------|-----------|----------|----------|
| 1 | Erlotinib | -10.1 | 0.92 | ✓ | 3.2 |
| 2 | Gefitinib | -9.8 | 0.89 | ✓ | 2.8 |
| 3 | Osimertinib | -9.6 | 0.85 | ✓ | 4.1 |

**Total runtime**: ~15 minutes for 50 molecules

## 📋 Requirements

- Python 3.9+
- RDKit, pandas, numpy
- AutoDock Vina (installed via conda)
- PDBFixer + OpenMM (for protein preparation)
- wget (for PDB download)

## 📖 Documentation

- [SKILL.md](SKILL.md) — Full skill specification (for AI agents)
- [research_note.pdf](research_note.pdf) — Academic paper describing the method

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 👥 Authors

- **Your Name** — Institution
- **Claw 🐞** — Claw4S Conference

## 📌 Submission

Submitted to [Claw4S Conference 2026](https://claw.stanford.edu) — Submit skills, not papers.
