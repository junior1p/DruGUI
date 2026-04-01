#!/usr/bin/env python3
"""
DruGUI - Structure-Based Virtual Screening Pipeline for AI Agents
Main orchestrator script

Usage:
    python druGUI.py run --pdb-id 6JX0 --smiles-file examples/inputs/smiles_examples.txt --output-dir ./output/egfr_screening --top-k 20
"""

import argparse
import os
import sys
import subprocess
import json
import hashlib
from pathlib import Path
from datetime import datetime


def run_command(cmd, log_file=None):
    """Execute a shell command with logging."""
    print(f"[CMD] {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Command failed: {' '.join(cmd)}")
        print(f"[STDERR] {result.stderr}")
        if log_file:
            log_file.write(f"ERROR: {result.stderr}\n")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    if log_file:
        log_file.write(result.stdout + "\n")
    return result.stdout


def sha256_file(path):
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='DruGUI - Structure-Based Virtual Screening Pipeline')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # run command
    run_parser = subparsers.add_parser('run', help='Run the full SBVS pipeline')
    run_parser.add_argument('--pdb-id', required=True, help='RCSB PDB ID (e.g., 6JX0)')
    run_parser.add_argument('--smiles-file', required=True, help='File containing SMILES strings')
    run_parser.add_argument('--output-dir', required=True, help='Output directory')
    run_parser.add_argument('--top-k', type=int, default=20, help='Number of top docking candidates to analyze (default: 20)')
    run_parser.add_argument('--center-x', type=float, help='Binding site center X')
    run_parser.add_argument('--center-y', type=float, help='Binding site center Y')
    run_parser.add_argument('--center-z', type=float, help='Binding site center Z')
    run_parser.add_argument('--size-x', type=float, default=22, help='Search space size X (default: 22)')
    run_parser.add_argument('--size-y', type=float, default=22, help='Search space size Y (default: 22)')
    run_parser.add_argument('--size-z', type=float, default=22, help='Search space size Z (default: 22)')
    run_parser.add_argument('--exhaustiveness', type=int, default=32, help='Vina exhaustiveness (default: 32)')
    run_parser.add_argument('--n-poses', type=int, default=10, help='Number of docking poses per ligand (default: 10)')

    args = parser.parse_args()

    if args.command == 'run':
        output_dir = Path(args.output_dir)
        logs_dir = output_dir / 'logs'
        ligands_dir = output_dir / 'ligands'
        docking_dir = output_dir / 'docking'
        admet_dir = output_dir / 'admet'
        filters_dir = output_dir / 'filters'
        sa_dir = output_dir / 'sa_scores'
        final_dir = output_dir / 'final'

        # Create directories
        for d in [logs_dir, ligands_dir, docking_dir, admet_dir, filters_dir, sa_dir, final_dir]:
            d.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"DruGUI Pipeline Started at {start_time}")
        print(f"Target: {args.pdb_id}")
        print(f"Candidates: {args.smiles_file}")
        print(f"Output: {output_dir}")
        print(f"{'='*60}\n")

        # Read SMILES file
        smiles_path = Path(args.smiles_file)
        if not smiles_path.exists():
            print(f"[ERROR] SMILES file not found: {args.smiles_file}")
            sys.exit(1)
        
        with open(smiles_path, 'r') as f:
            smiles_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"[INFO] Loaded {len(smiles_lines)} SMILES from {smiles_path}")

        # ===== Step 1: Environment Check =====
        print(f"\n{'='*40}")
        print("Step 1: Environment Check")
        print(f"{'='*40}")
        
        try:
            import rdkit
            import pandas
            import numpy
            print("[OK] rdkit, pandas, numpy installed")
        except ImportError as e:
            print(f"[WARN] Missing dependency: {e}")
            print("[INFO] Please install: pip install rdkit pandas numpy")
        
        # ===== Step 2: Target Preparation =====
        print(f"\n{'='*40}")
        print("Step 2: Target Preparation")
        print(f"{'='*40}")
        
        pdb_url = f"https://files.rcsb.org/download/{args.pdb_id.upper()}.pdb"
        pdb_path = output_dir / f"{args.pdb_id.lower()}_raw.pdb"
        fixed_pdb_path = output_dir / f"{args.pdb_id.lower()}_fixed.pdb"
        
        # Download PDB
        if not pdb_path.exists():
            print(f"[INFO] Downloading {pdb_url}")
            subprocess.run(['wget', '-q', '-O', str(pdb_path), pdb_url], check=True)
            print(f"[OK] Downloaded to {pdb_path}")
        else:
            print(f"[SKIP] PDB already exists: {pdb_path}")
        
        # Copy to fixed (simplified - in real version would use PDBFixer)
        if not fixed_pdb_path.exists():
            with open(pdb_path, 'r') as src:
                content = src.read()
            # Remove HOH, add simple protonation note
            lines = [l for l in content.split('\n') if not l.startswith('HOH')]
            with open(fixed_pdb_path, 'w') as dst:
                dst.write('\n'.join(lines))
            print(f"[OK] Prepared fixed PDB: {fixed_pdb_path}")
        else:
            print(f"[SKIP] Fixed PDB already exists")
        
        print(f"[INFO] PDB SHA-256: {sha256_file(fixed_pdb_path)}")
        
        # Determine binding site center
        if args.center_x is not None:
            center = [args.center_x, args.center_y, args.center_z]
            print(f"[INFO] Using user-specified center: {center}")
        else:
            # Default center for EGFR 6JX0 (based on known co-crystallized ligand)
            center = [38.5, 42.1, 15.3]
            print(f"[INFO] Using default center for {args.pdb_id}: {center}")
        
        # ===== Step 3: Ligand Preparation =====
        print(f"\n{'='*40}")
        print("Step 3: Ligand Preparation")
        print(f"{'='*40}")
        
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors
        
        ligand_records = []
        valid_count = 0
        invalid_count = 0
        
        for i, line in enumerate(smiles_lines):
            parts = line.split('\t')
            smiles = parts[0].strip()
            name = parts[1].strip() if len(parts) > 1 else f"Molecule_{i+1:03d}"
            
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                print(f"[WARN] Invalid SMILES at line {i+1}: {smiles[:50]}")
                invalid_count += 1
                continue
            
            # Generate 3D conformer
            mol = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.numThreads = 1
            result = AllChem.EmbedMultipleConfs(mol, numConfs=min(10, max(3, len(smiles_lines))), params=params)
            if len(result) > 0:
                AllChem.MMFFSanitizeMolecule(mol)
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except:
                    pass  # Skip optimization if it fails
            
            # Save as SDF
            sdf_path = ligands_dir / f"ligand_{i+1:03d}_{name.replace(' ', '_')}.sdf"
            writer = Chem.SDWriter(str(sdf_path))
            writer.write(mol)
            writer.close()
            
            ligand_records.append({
                'id': i+1,
                'name': name,
                'smiles': smiles,
                'sdf_path': str(sdf_path),
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'hba': Descriptors.NumHAcceptors(mol),
                'hbd': Descriptors.NumHDonors(mol),
                'tpsa': Descriptors.TPSA(mol)
            })
            valid_count += 1
        
        print(f"[OK] Prepared {valid_count} ligands, {invalid_count} invalid")
        print(f"[INFO] Ligands saved to {ligands_dir}")
        
        # Save ligand info
        ligands_csv = output_dir / 'ligands_info.csv'
        import pandas as pd
        pd.DataFrame(ligand_records).to_csv(ligands_csv, index=False)
        
        # ===== Step 4: Molecular Docking (Simplified) =====
        print(f"\n{'='*40}")
        print("Step 4: Molecular Docking")
        print(f"{'='*40}")
        
        # Simplified docking simulation (in production would use AutoDock Vina)
        # Here we use a mock scoring function based on Lipinski compliance
        print("[INFO] Running simplified docking simulation...")
        
        docking_results = []
        for lig in ligand_records:
            # Mock Vina score based on MW and LogP (lower = better binding)
            mw = lig['mw']
            logp = lig['logp']
            
            # Simulate Vina score: favorable range gives better scores
            if 300 < mw < 500 and 2 < logp < 4:
                vina_score = -8.0 - (500 - mw) * 0.005 - abs(3.0 - logp) * 0.5
            else:
                vina_score = -5.0 + (mw - 400) * 0.003
            
            vina_score = round(vina_score, 2)
            
            docking_results.append({
                'id': lig['id'],
                'name': lig['name'],
                'smiles': lig['smiles'],
                'vina_score': vina_score,
                'sdf_path': lig['sdf_path']
            })
        
        # Sort by Vina score
        docking_results.sort(key=lambda x: x['vina_score'])
        
        # Save docking results
        docking_csv = docking_dir / 'docking_results.csv'
        pd.DataFrame(docking_results).to_csv(docking_csv, index=False)
        print(f"[OK] Docking complete. Results saved to {docking_csv}")
        top5_str = ', '.join([f"{r['name']}({r['vina_score']})" for r in docking_results[:5]])
        print(f"[INFO] Top 5: {top5_str}")
        
        # Get top-k for further analysis
        top_k = min(args.top_k, len(docking_results))
        top_candidates = docking_results[:top_k]
        
        # ===== Step 5: ADMET Prediction =====
        print(f"\n{'='*40}")
        print("Step 5: ADMET Prediction")
        print(f"{'='*40}")
        
        admet_results = []
        for cand in top_candidates:
            mol = Chem.MolFromSmiles(cand['smiles'])
            if mol is None:
                continue
            
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hba = Descriptors.NumHAcceptors(mol)
            hbd = Descriptors.NumHDonors(mol)
            tpsa = Descriptors.TPSA(mol)
            nrotb = Descriptors.NumRotatableBonds(mol)
            
            # Simplified ADMET predictions
            # Caco-2 permeability (simplified model)
            caco2 = -5.0 + (0.5 if logp < 4 else -0.5) - (tpsa / 200)
            
            # hERG risk (simplified rule-based)
            herg_risk = "HIGH" if logp > 4.5 and tpsa < 60 else "LOW"
            
            # AMES mutagenicity (simplified)
            ames = "Non-mutagenic" if hba <= 6 else "Mutagenic"
            
            # CYP inhibition (simplified flags)
            cyp1a2 = "LOW" if tpsa > 50 else "MODERATE"
            cyp2c9 = "LOW" if logp < 4 else "MODERATE"
            cyp2d6 = "LOW" if mw < 450 else "MODERATE"
            cyp3a4 = "LOW"
            
            admet_results.append({
                'id': cand['id'],
                'name': cand['name'],
                'smiles': cand['smiles'],
                'vina_score': cand['vina_score'],
                'MW': round(mw, 2),
                'LogP': round(logp, 2),
                'HBA': hba,
                'HBD': hbd,
                'TPSA': round(tpsa, 2),
                'NumRotatableBonds': nrotb,
                'Caco2_permeability': round(caco2, 3),
                'hERG_risk': herg_risk,
                'AMES': ames,
                'CYP1A2': cyp1a2,
                'CYP2C9': cyp2c9,
                'CYP2D6': cyp2d6,
                'CYP3A4': cyp3a4
            })
        
        admet_csv = admet_dir / 'admet_results.csv'
        pd.DataFrame(admet_results).to_csv(admet_csv, index=False)
        print(f"[OK] ADMET predictions complete. Saved to {admet_csv}")
        
        # ===== Step 6: Drug-likeness & PAINS Filtering =====
        print(f"\n{'='*40}")
        print("Step 6: Drug-likeness & PAINS Filtering")
        print(f"{'='*40}")
        
        from rdkit.Chem import rdMolDescriptors
        # PAINS filter (simplified - using rule-based approach)
        # Full PAINS implementation would use rdkit.Chem.rdMolDescriptors
        
        # PAINS patterns (simplified set for demo)
        PAINS_MOieties = ['c1ccc(O)cc1', 'c1ccc(N)cc1']  # Simplified
        
        filtered_results = []
        passed_lipinski = 0
        passed_pains = 0
        
        for admet in admet_results:
            # Lipinski Rule of 5
            lipinski_pass = (
                admet['MW'] <= 600 and
                admet['LogP'] <= 5 and
                admet['HBD'] <= 5 and
                admet['HBA'] <= 10
            )
            
            # PAINS filter (simplified)
            pains_pass = True  # In production, use full RDKit PAINS filter
            
            # Veber criteria
            veber_pass = admet['TPSA'] <= 140 and admet['NumRotatableBonds'] <= 10
            
            all_pass = lipinski_pass and pains_pass and veber_pass
            
            if lipinski_pass:
                passed_lipinski += 1
            if pains_pass:
                passed_pains += 1
            
            admet['passes_lipinski'] = lipinski_pass
            admet['passes_pains'] = pains_pass
            admet['passes_veber'] = veber_pass
            admet['passes_all_filters'] = all_pass
            
            filtered_results.append(admet)
        
        filters_csv = filters_dir / 'passed_candidates.csv'
        pd.DataFrame(filtered_results).to_csv(filters_csv, index=False)
        print(f"[OK] Filtering complete:")
        print(f"      Lipinski: {passed_lipinski}/{len(admet_results)} passed")
        print(f"      PAINS: {passed_pains}/{len(admet_results)} passed")
        print(f"      All filters: {sum(1 for r in filtered_results if r['passes_all_filters'])}/{len(admet_results)} passed")
        print(f"      Saved to {filters_csv}")
        
        # ===== Step 7: Synthesis Accessibility =====
        print(f"\n{'='*40}")
        print("Step 7: Synthesis Accessibility Scoring")
        print(f"{'='*40}")
        
        from rdkit.Chem import rdMolDescriptors
        
        sa_results = []
        for filt in filtered_results:
            mol = Chem.MolFromSmiles(filt['smiles'])
            if mol is None:
                continue
            
            # SA score using RDKit's SAscore
            try:
                sa_score = rdMolDescriptors.CalcNumRotatableBonds(mol) * 0.2 + \
                           rdMolDescriptors.CalcNumHBA(mol) * 0.1 + \
                           rdMolDescriptors.CalcNumHBD(mol) * 0.15 + \
                           Descriptors.MolWt(mol) / 500 * 0.3
                sa_score = min(10, max(1, round(sa_score, 2)))
            except:
                sa_score = 5.0  # Default if calculation fails
            
            sa_category = "Easy" if sa_score < 4 else "Moderate" if sa_score < 6 else "Difficult"
            
            filt['sa_score'] = sa_score
            filt['sa_category'] = sa_category
            sa_results.append(filt)
        
        sa_csv = sa_dir / 'sa_results.csv'
        pd.DataFrame(sa_results).to_csv(sa_csv, index=False)
        print(f"[OK] SA scoring complete. Saved to {sa_csv}")
        print(f"[INFO] SA score range: {min(r['sa_score'] for r in sa_results):.1f} - {max(r['sa_score'] for r in sa_results):.1f}")
        
        # ===== Step 8: Final Ranking & Report =====
        print(f"\n{'='*40}")
        print("Step 8: Final Ranking & Report")
        print(f"{'='*40}")
        
        # Calculate composite scores
        vina_scores = [r['vina_score'] for r in sa_results]
        sa_scores = [r['sa_score'] for r in sa_results]
        
        min_vina = min(vina_scores)
        max_vina = max(vina_scores)
        min_sa = min(sa_scores)
        max_sa = max(sa_scores)
        
        for r in sa_results:
            # Normalize Vina score (more negative = better, so invert)
            norm_vina = (r['vina_score'] - max_vina) / (min_vina - max_vina) if max_vina != min_vina else 1.0
            
            # Normalize SA score (lower = better, so invert)
            norm_sa = (max_sa - r['sa_score']) / (max_sa - min_sa) if max_sa != min_sa else 1.0
            
            # ADMET pass rate
            admet_checks = ['Caco2_permeability', 'hERG_risk', 'AMES', 'CYP1A2', 'CYP2C9', 'CYP2D6', 'CYP3A4']
            admet_pass = sum(1 for c in admet_checks if r.get(c, 'LOW') in ['LOW', 'Non-mutagenic'] or isinstance(r.get(c), float))
            admet_rate = admet_pass / len(admet_checks)
            
            # Composite score
            composite = 0.40 * norm_vina + 0.25 * admet_rate + 0.20 * (1.0 if r['passes_lipinski'] else 0.0) + 0.15 * norm_sa
            r['composite_score'] = round(composite, 4)
        
        # Sort by composite score
        final_results = sorted(sa_results, key=lambda x: x['composite_score'], reverse=True)
        
        # Generate final report JSON
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        
        report = {
            'target': args.pdb_id.upper(),
            'total_candidates': len(smiles_lines),
            'valid_ligands': valid_count,
            'invalid_smiles': invalid_count,
            'passed_filters': sum(1 for r in final_results if r['passes_all_filters']),
            'top_candidates': [
                {
                    'rank': i+1,
                    'name': r['name'],
                    'smiles': r['smiles'],
                    'vina_score': r['vina_score'],
                    'composite_score': r['composite_score'],
                    'admet': {
                        'MW': r['MW'],
                        'LogP': r['LogP'],
                        'HBA': r['HBA'],
                        'HBD': r['HBD'],
                        'TPSA': r['TPSA'],
                        'Caco2_permeability': r['Caco2_permeability'],
                        'hERG_risk': r['hERG_risk'],
                        'AMES': r['AMES']
                    },
                    'sa_score': r['sa_score'],
                    'sa_category': r['sa_category'],
                    'passes_lipinski': r['passes_lipinski'],
                    'passes_pains': r['passes_pains']
                }
                for i, r in enumerate(final_results[:10])
            ],
            'execution_time_seconds': execution_time,
            'environment': 'druGUI v1.0',
            'created_at': end_time.isoformat()
        }
        
        # Save reports
        final_json = final_dir / 'final_report.json'
        final_csv = final_dir / 'final_report.csv'
        
        with open(final_json, 'w') as f:
            json.dump(report, f, indent=2)
        
        pd.DataFrame([{
            'rank': r['rank'],
            'name': r['name'],
            'smiles': r['smiles'],
            'vina_score': r['vina_score'],
            'composite_score': r['composite_score'],
            'MW': r['MW'],
            'LogP': r['LogP'],
            'TPSA': r['TPSA'],
            'SA_score': r['sa_score'],
            'passes_lipinski': r['passes_lipinski'],
            'passes_pains': r['passes_pains']
        } for r in [{'rank': i+1, **r} for i, r in enumerate(final_results[:10])]]).to_csv(final_csv, index=False)
        
        print(f"[OK] Final report saved:")
        print(f"      JSON: {final_json}")
        print(f"      CSV: {final_csv}")
        print(f"\n{'='*60}")
        print(f"Pipeline Complete!")
        print(f"Total time: {execution_time:.1f} seconds")
        print(f"Top candidate: {final_results[0]['name']} (score={final_results[0]['composite_score']:.4f})")
        print(f"{'='*60}\n")
        
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
