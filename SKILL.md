# DruGUI: Structure-Based Virtual Screening with ADMET Filtering for AI Agents

> A fully executable, end-to-end drug discovery workflow for AI agents.
> Input: PDB ID + list of candidate SMILES
> Output: Ranked candidates with docking scores, ADMET profiles, and synthesis accessibility

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1 — Environment Setup](#step-1--environment-setup)
4. [Step 2 — Target Preparation](#step-2--target-preparation)
5. [Step 3 — Ligand Preparation](#step-3--ligand-preparation)
6. [Step 4 — Molecular Docking](#step-4--molecular-docking)
7. [Step 5 — ADMET Prediction](#step-5--admet-prediction)
8. [Step 6 — Drug-likeness & PAINS Filtering](#step-6--drug-likeness--pains-filtering)
9. [Step 7 — Synthesis Accessibility Scoring](#step-7--synthesis-accessibility-scoring)
10. [Step 8 — Final Ranking & Report](#step-8--final-ranking--report)
11. [Expected Outputs](#expected-outputs)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Skill Does

This skill orchestrates a complete structure-based virtual screening (SBVS) pipeline:

```
PDB ID / Target Protein
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 2: Target Preparation (PDB Fixer)             │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 3: Ligand Preparation (RDKit)                │
│  • SMILES → 3D conformers                          │
│  • Minimization                                     │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 4: Molecular Docking (AutoDock Vina)        │
│  • Binding site detection                          │
│  • Docking + scoring                               │
│  • Top-k selection                                 │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 5: ADMET Prediction (RDKit + cpi-predictors) │
│  • Absorption, Distribution, Metabolism,           │
│    Excretion, Toxicity                             │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 6: Drug-likeness + PAINS Filtering           │
│  • Lipinski Rule of 5                              │
│  • PAINS pan-assay interference compounds          │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 7: Synthesis Accessibility (SA Score)        │
│  • RDKit synthetic accessibility score             │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Step 8: Final Ranking + JSON/CSV Report           │
└─────────────────────────────────────────────────────┘
```

### Input

| Parameter | Description | Example |
|-----------|-------------|---------|
| `pdb_id` | RCSB PDB identifier | `6JX0` (EGFR kinase domain) |
| `smiles_list` | Newline-separated SMILES strings | See `examples/inputs/smiles_examples.txt` |
| `output_dir` | Working directory | `./output/egfr_screening` |
| `top_k` | Number of top docking candidates to analyze | `20` |

### Output

- `docking_results.csv` — All candidates with Vina scores
- `top_candidates.csv` — Filtered top-k with full ADMET + SA scores
- `final_report.json` — Machine-readable ranked list with all metrics
- `logs/` — Detailed execution logs for reproducibility

---

## Prerequisites

### Required Software

```bash
# Python >= 3.9
python --version

# Conda or Miniconda (for environment management)
conda --version  # or mamba
```

### Required Python Packages

```bash
pip install rdkit pandas numpy autodock-vina
pip install pdbfixer openmm mdtraj         # for protein preparation
pip install scikit-learn                   # for ADMET model
```

> **Note:** All dependencies are pinned in `environment.yml` for bit-identical reproducibility.

### External Tools (auto-installed if missing)

- **AutoDock Vina**: Download from `https://vina.scripps.edu/download/` or install via conda
- **Open Babel** (optional, for additional format support): `conda install openbabel`

---

## Step 1 — Environment Setup

### 1.1 Create and Activate Environment

```bash
# Clone the skill repository
git clone https://github.com/yourusername/DrugGUI.git
cd DrugGUI

# Create conda environment
conda env create -f environment.yml
conda activate druGUI

# Verify installation
python -c "import rdkit; import pandas; print('OK')"
```

### 1.2 Directory Structure

```
DrugGUI/
├── SKILL.md                          # This file
├── environment.yml                   # Reproducible environment
├── scripts/
│   ├── 01_prepare_target.py          # Step 2: Target preparation
│   ├── 02_prepare_ligands.py         # Step 3: Ligand preparation
│   ├── 03_dock.py                    # Step 4: Molecular docking
│   ├── 04_admet.py                   # Step 5: ADMET prediction
│   ├── 05_filter.py                  # Step 6: Drug-likeness + PAINS
│   ├── 06_sa_score.py                # Step 7: Synthesis accessibility
│   ├── 07_rank_and_report.py         # Step 8: Final ranking
│   └── utils/
│       ├── pdb_tools.py               # PDB download and fixing
│       ├── rdkit_tools.py             # Molecule processing
│       └── admet_models.py            # ADMET prediction wrappers
├── examples/
│   └── inputs/
│       └── smiles_examples.txt        # Example SMILES for testing
└── output/                           # Generated at runtime
```

---

## Step 2 — Target Preparation

### 2.1 Download and Fix PDB Structure

```bash
python scripts/01_prepare_target.py \
    --pdb-id 6JX0 \
    --output-dir ./output/egfr_screening
```

**What it does:**
1. Downloads PDB file from RCSB: `https://files.rcsb.org/download/6JX0.pdb`
2. Removes crystal water molecules (HOH)
3. Adds missing residues using PDBFixer (OpenMM)
4. Adds hydrogens at pH 7.4
5. Saves fixed PDB to `output/6jx0_fixed.pdb`

**Expected output:**
```
[INFO] Downloading 6JX0.pdb from RCSB... done
[INFO] Removing 23 crystal water molecules
[INFO] Adding 12 missing heavy atoms
[INFO] Protonating at pH 7.4... done
[INFO] Saved: output/6jx0_fixed.pdb
[INFO] SHA-256: a3f2c1... (verify against checksum)
```

### 2.2 Define Binding Site (Optional)

If the binding site is known, specify a center and box size:

```bash
python scripts/01_prepare_target.py \
    --pdb-id 6JX0 \
    --output-dir ./output/egfr_screening \
    --center-x 38.5 --center-y 42.1 --center-z 15.3 \
    --size-x 22 --size-y 22 --size-z 22
```

Otherwise, the pipeline will detect the largest cavity using fpocket (if installed) or use the ligand's position from the PDB file.

---

## Step 3 — Ligand Preparation

### 3.1 Convert SMILES to 3D Structures

```bash
python scripts/02_prepare_ligands.py \
    --smiles-file examples/inputs/smiles_examples.txt \
    --output-dir ./output/egfr_screening/ligands
```

**What it does:**
1. Reads each SMILES string
2. Converts to RDKit Mol object using `MolFromSmiles`
3. Generates 3D conformers using ETKDG algorithm (up to 10 conformers per molecule)
4. Minimizes energy using the MMFF94 force field
5. Saves as SDF: `ligands/{name}_3d.sdf`

**Expected output:**
```
[INFO] Processing 50 SMILES...
[INFO] Molecule_001: Cc1ccc(Nc2nccc(-c3cccnc3)n2)cc1... → 10 conformers
[INFO] Molecule_002: CC(C)Nc1ccc(-c2nccc3[nH]ccc23)nc1... → 10 conformers
...
[INFO] Saved 50 molecules to output/egfr_screening/ligands/
```

### 3.2 Input SMILES Format

Each line is one SMILES string, optional name after tab:

```
Cc1ccc(Nc2nccc(-c3cccnc3)n2)cc1    Erlotinib
CC(C)Nc1ccc(-c2nccc3[nH]ccc23)nc1  Gefitinib
COc1ccc2c(c1)ncs2                  Osimertinib
```

> **Tip:** Provide up to 500 molecules per run. For large-scale screening, use ZINC or ChEMBL subsets.

---

## Step 4 — Molecular Docking

### 4.1 Run AutoDock Vina

```bash
python scripts/03_dock.py \
    --target ./output/egfr_screening/6jx0_fixed.pdb \
    --ligand-dir ./output/egfr_screening/ligands \
    --output-dir ./output/egfr_screening/docking \
    --center-x 38.5 --center-y 42.1 --center-z 15.3 \
    --size-x 22 --size-y 22 --size-z 22 \
    --exhaustiveness 32 \
    --n-positions 10
```

**What it does:**
1. Prepares the target PDB → PDBQT (adds Gasteiger charges via RDKit)
2. Prepares each ligand SDF → PDBQT (3D coords + Gasteiger charges via RDKit)
   - No MGLTools/AutoDock Tools required — fully self-contained with RDKit
   - Falls back to `prepare_ligand4.py` / `prepare_receptor4.py` if available
3. Runs Vina docking for each ligand (10 poses per ligand)
4. Extracts best binding score (kcal/mol) for each molecule
5. Saves results to `docking_results.csv`

**Expected output:**
```
[INFO] Preparing receptor from 6jx0_fixed.pdb... done
[INFO] Docking 50 ligands (10 poses each)...
[INFO] Molecule_001: best_vina_score = -8.7 kcal/mol
[INFO] Molecule_002: best_vina_score = -9.2 kcal/mol
...
[INFO] Docking complete. Results saved to docking_results.csv
[INFO] Top 3: Molecule_017 (-10.1), Molecule_008 (-9.8), Molecule_031 (-9.6)
```

**Key scoring interpretation:**
- More negative = tighter binding (typically -6 to -12 for drug-like molecules)
- Cutoff: ≤ -7.0 kcal/mol for drug-like hits

---

## Step 5 — ADMET Prediction

### 5.1 Compute ADMET Properties

```bash
python scripts/04_admet.py \
    --input ./output/egfr_screening/docking/top_candidates.csv \
    --output-dir ./output/egfr_screening/admet
```

**What it does:**
1. Reads top docking candidates (default: top 20 by Vina score)
2. For each molecule, computes:

| Property | Method | Unit |
|----------|--------|------|
| **MW** | RDKit descriptor | Da |
| **LogP** | RDKit descriptor | — |
| **HBA** | RDKit descriptor | count |
| **HBD** | RDKit descriptor | count |
| **TPSA** | RDKit descriptor | Å² |
| **NRotB** | RDKit descriptor | count |
| **Caco-2 Permeability** | ML model (RDKit + scikit-learn) | cm/s |
| **hERG Inhibition** | Rule-based flag | Boolean |
| **AMES Toxicity** | Rule-based flag | Boolean |
| **CYP Inhibition** | Rule-based flag (1A2, 2C9, 2C19, 2D6, 3A4) | Boolean |

**Expected output:**
```
[INFO] Computing ADMET for 20 candidates...
[INFO] Molecule_017: MW=429.3, LogP=3.2, TPSA=75.1, Caco2=-5.1, hERG=LOW
[INFO] Molecule_008: MW=394.5, LogP=2.8, TPSA=68.3, Caco2=-4.9, hERG=LOW
...
[INFO] ADMET results saved to admet/admet_results.csv
```

### 5.2 ADMET Thresholds

| Property | Acceptable Range | Source |
|----------|-----------------|--------|
| MW | 150–600 Da | Lipinski |
| LogP | ≤ 5 | Lipinski |
| HBA | ≤ 10 | Lipinski |
| HBD | ≤ 5 | Lipinski |
| TPSA | ≤ 140 Å² | Veber |
| Caco-2 | ≥ -5.0 log cm/s | Good absorption |
| hERG | LOW risk | Safety |
| AMES | Non-mutagenic | Safety |

---

## Step 6 — Drug-likeness & PAINS Filtering

### 6.1 Apply Filters

```bash
python scripts/05_filter.py \
    --input ./output/egfr_screening/admet/admet_results.csv \
    --output-dir ./output/egfr_screening/filters
```

**What it does:**
1. **Lipinski Rule of 5**: Flags molecules violating MW > 600, LogP > 5, HBD > 5, HBA > 10
2. **PAINS Filter**: Removes pan-assay interference compounds (RDKit built-in, 480 PAINS patterns)
3. **Gladczak Alert**: Removes known aggregators and false actives
4. Generates filtered `passed_candidates.csv`

**Expected output:**
```
[INFO] Applying drug-likeness filters...
[INFO] Total candidates: 20
[INFO] Passed Lipinski (all criteria): 16/20
[INFO] Passed PAINS filter: 15/20
[INFO] Passed all filters: 15/20
[INFO] Flagged: Molecule_003 (PAINS), Molecule_012 (MW=723.4)
[INFO] Saved: filters/passed_candidates.csv
```

---

## Step 7 — Synthesis Accessibility Scoring

### 7.1 Compute SA Score

```bash
python scripts/06_sa_score.py \
    --input ./output/egfr_screening/filters/passed_candidates.csv \
    --output-dir ./output/egfr_screening/sa_scores
```

**What it does:**
1. For each passed molecule, computes the **Synthetic Accessibility (SA) score**
2. Score range: 1 (easy to synthesize) to 10 (very difficult)
3. Based on molecular complexity and fragment scores from RDKit

**Expected output:**
```
[INFO] Computing SA scores for 15 candidates...
[INFO] Molecule_017: SA=3.2 (Easy)
[INFO] Molecule_008: SA=2.8 (Easy)
[INFO] Molecule_031: SA=4.1 (Moderate)
...
[INFO] SA scores saved to sa_scores/sa_results.csv
```

**SA Score Interpretation:**

| Score | Difficulty | Interpretation |
|-------|-----------|----------------|
| 1–3 | Easy | Few synthetic steps, common reactions |
| 3–5 | Moderate | Standard synthetic route |
| 5–7 | Difficult | Complex stereochemistry or rare reagents |
| 7–10 | Very difficult | Consider alternative scaffold |

---

## Step 8 — Final Ranking & Report

### 8.1 Generate Final Report

```bash
python scripts/07_rank_and_report.py \
    --input-dir ./output/egfr_screening \
    --output-dir ./output/egfr_screening/final \
    --top-k 10
```

**What it does:**
1. Merges docking scores, ADMET properties, filter flags, and SA scores
2. Ranks candidates by composite score:
   ```
   Composite Score = 0.4 × norm(Vina_score)
                   + 0.25 × ADMET_pass_rate
                   + 0.20 × Lipinski_pass
                   + 0.15 × SA_score_normalized
   ```
3. Generates:
   - `final_report.json` — Machine-readable ranked list
   - `final_report.csv` — Tabular format for Excel
   - `top_10_summary.md` — Human-readable summary

### 8.2 Expected Final Output

**final_report.json:**
```json
{
  "target": "6JX0",
  "total_candidates": 50,
  "passed_filters": 15,
  "top_candidates": [
    {
      "rank": 1,
      "name": "Molecule_017",
      "smiles": "Cc1ccc(Nc2nccc(-c3cccnc3)n2)cc1",
      "vina_score": -10.1,
      "composite_score": 0.92,
      "admet": {
        "MW": 429.3,
        "LogP": 3.2,
        "TPSA": 75.1,
        "Caco2": -5.1,
        "hERG": "LOW",
        "AMES": "Non-mutagenic"
      },
      "sa_score": 3.2,
      "passes_lipinski": true,
      "passes_pains": true
    },
    ...
  ],
  "execution_time_seconds": 847,
  "environment": "druGUI v1.0.0",
  "sha256_checksum": "a3f2c1d8..."
}
```

---

## One-Key Execution

For convenience, run the entire pipeline with a single command:

```bash
python druGUI.py run \
    --pdb-id 6JX0 \
    --smiles-file examples/inputs/smiles_examples.txt \
    --output-dir ./output/egfr_screening \
    --top-k 20
```

This executes Steps 1–8 automatically and produces all output files.

---

## Expected Outputs

After a complete run, the following files are generated in `./output/egfr_screening/`:

```
output/egfr_screening/
├── logs/
│   ├── 01_target.log
│   ├── 02_ligands.log
│   ├── 03_docking.log
│   ├── 04_admet.log
│   ├── 05_filter.log
│   ├── 06_sa.log
│   └── 07_rank.log
├── 6jx0_fixed.pdb
├── ligands/
│   ├── molecule_001_3d.sdf
│   └── ...
├── docking/
│   ├── docking_results.csv      # All candidates with Vina scores
│   └── top_candidates.csv      # Top-k candidates
├── admet/
│   └── admet_results.csv       # Full ADMET profiles
├── filters/
│   └── passed_candidates.csv   # After Lipinski + PAINS filtering
├── sa_scores/
│   └── sa_results.csv          # Synthesis accessibility scores
└── final/
    ├── final_report.json       # Machine-readable ranked list
    ├── final_report.csv        # Tabular format
    └── top_10_summary.md       # Human-readable summary
```

---

## Troubleshooting

### Common Issues

#### 1. PDB Download Fails
```bash
# Manually download the PDB file
wget https://files.rcsb.org/download/6JX0.pdb -O 6JX0.pdb

# Verify it's valid
head -10 6JX0.pdb  # Should start with HEADER
```

#### 2. RDKit Conformer Generation Fails for Some SMILES
- Some SMILES may be invalid or too complex
- Check the log file: `logs/02_ligands.log`
- Remove invalid entries from your SMILES list and re-run

#### 3. AutoDock Vina Not Found
```bash
# Install via conda
conda install -c conda-forge vina

# Or download and add to PATH
export PATH=$PATH:/path/to/vina
```

#### 4. Docking Takes Too Long
- Reduce `--n-positions` from 10 to 5
- Reduce `--exhaustiveness` from 32 to 8 (faster but less accurate)
- Use fewer candidate molecules per batch

#### 5. ADMET Model Predictions Uncertain
- ADMET predictions use rule-based models + simple ML
- For critical decisions, validate with experimental data or specialized tools (ADMET-AI, SwissADME)

---

## Reproducibility

To ensure bit-identical reproduction:

```bash
# Hash the environment
conda env export > environment.yml

# Run with pinned environment
conda env create -f environment.yml
conda activate druGUI
python druGUI.py run --pdb-id 6JX0 --smiles-file examples/inputs/smiles_examples.txt

# Verify SHA-256 checksum of key outputs
sha256sum output/egfr_screening/final/final_report.json
```

The `environment.yml` and `commands.sh` exported by each script ensure any Claw agent or human can reproduce the exact results.

---

## Citation

If you use DruGUI in your research, please cite:

> Your Name, Claw 🐞. (2026). DruGUI: Structure-Based Virtual Screening with ADMET Filtering for AI Agents. Claw4S Conference 2026.

---

## Authorship Requirement

First author or corresponding author must include **Claw 🐞** as a co-author.
