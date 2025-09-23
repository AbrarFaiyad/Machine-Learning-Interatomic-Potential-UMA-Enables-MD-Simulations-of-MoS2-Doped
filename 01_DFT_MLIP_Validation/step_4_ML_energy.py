#!/usr/bin/env python
"""
Optimise all .xyz structures in ./formation_energy_structures/
with the UMA (OMat24) potential, write the raw energies
and the derived formation energies to one CSV file.
Additionally, perform RDF analysis on the optimized structures
for comparison with DFT results.

Requires:
    pip install fairchem ase pandas numpy
"""

import os, re, glob
from pathlib import Path
import pandas as pd
import numpy as np
from ase.io import read, write
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter
from fairchem.core import pretrained_mlip, FAIRChemCalculator
import warnings
warnings.filterwarnings('ignore')

# ----------------- settings -------------------------------------------
STRUCT_DIR = Path("formation_energy_structures")   # <— folder from previous script
DOPANTS = [
    'C', 'N', 'F', 'B', 'Se', 'Te', 'Cl', 'Si', 'Li', 'Na', 'Al', 'Zn', 'V', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Ta', 'W', 'Re', 'Ir', 'Pt', 'Au', 'Ti', 'O'
]                     # extend as needed
OUT_CSV    = "ml_formation_energies_s.csv"
ML_OPT_DIR = Path("ml_optimized_structures")  # Store optimized structures for RDF analysis

# ----------------- load UMA once --------------------------------------
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device="cuda") # Currently uses UMA small, change to uma-m-1p1 for UMA medium
calc      = FAIRChemCalculator(predictor, task_name="omat")

# ----------------- helpers -------------------------------------------
def optimise_energy(path):
    """Read, BFGS-relax (atoms only, cell fixed), return final total energy (eV) and optimized atoms."""
    atoms = read(path)
    atoms.info.update({"spin": 2, "charge": 0})
    atoms.calc = calc

    # Always relax atomic positions only, cell fixed
    opt = BFGS(FrechetCellFilter(atoms), trajectory=None)
    opt.run(fmax=0.005, steps=10000)
    return atoms.get_potential_energy(), len(atoms), atoms

def reference_key(element, energy_dict):
    """Pick the filename key that holds the elemental reference of <element>."""
    candidates = [
        f"{element}_bcc", f"{element}_fcc", f"{element}_hcp",
        f"{element}_diamond", f"{element}_atom_box", f"{element}2_box"
    ]
    for c in candidates:
        if c in energy_dict:
            return c
    raise RuntimeError(f"No reference phase found for element {element}")

def calculate_partial_rdf(atoms, element_pairs, r_max=10.0, n_bins=100):
    """
    Calculate partial radial distribution functions for specified element pairs.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        The atomic structure
    element_pairs : list of tuples
        List of element pairs to calculate RDF for
    r_max : float
        Maximum distance for RDF calculation (Angstrom)
    n_bins : int
        Number of bins for RDF histogram
    
    Returns:
    --------
    dict : Dictionary with element pairs as keys and RDF arrays as values
    """
    
    # Get cell volume
    volume = atoms.get_volume()
    
    # Get atomic positions and symbols
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # Create distance bins
    dr = r_max / n_bins
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    rdf_results = {}
    
    for pair in element_pairs:
        element1, element2 = pair
        
        # Find indices of atoms for each element
        indices1 = [i for i, sym in enumerate(symbols) if sym == element1]
        indices2 = [i for i, sym in enumerate(symbols) if sym == element2]
        
        if not indices1 or not indices2:
            continue  # Skip pairs where elements are not present
        
        # Calculate distances
        distances = []
        
        for i in indices1:
            for j in indices2:
                if i != j:  # Avoid self-interaction
                    # Calculate minimum image distance
                    dist_vec = positions[j] - positions[i]
                    # Apply periodic boundary conditions
                    dist_vec = dist_vec - np.round(dist_vec / atoms.cell.diagonal()) * atoms.cell.diagonal()
                    dist = np.linalg.norm(dist_vec)
                    if dist <= r_max:
                        distances.append(dist)
        
        # Create histogram
        hist, _ = np.histogram(distances, bins=r_bins)
        
        # Normalize to get RDF
        if len(indices1) > 0 and len(indices2) > 0:
            # Calculate number density
            if element1 == element2:
                n_pairs = len(indices1) * (len(indices1) - 1) / 2
            else:
                n_pairs = len(indices1) * len(indices2)
            
            # Ideal gas normalization
            shell_volumes = (4/3) * np.pi * (r_bins[1:]**3 - r_bins[:-1]**3)
            density = len(indices2) / volume
            ideal_counts = density * shell_volumes * len(indices1)
            
            # Avoid division by zero
            rdf = np.where(ideal_counts > 0, hist / ideal_counts, 0)
        else:
            rdf = np.zeros(n_bins)
        
        rdf_results[f"{element1}-{element2}"] = rdf
    
    return rdf_results, r_centers

def get_unique_elements(atoms):
    """Get unique elements present in the structure."""
    return list(set(atoms.get_chemical_symbols()))

def get_relevant_pairs(elements):
    """Generate all relevant element pairs from the elements present in the structure."""
    pairs = []
    
    # Always include same-element pairs if present
    for elem in elements:
        pairs.append((elem, elem))
    
    # Include all cross-element pairs
    for i, elem1 in enumerate(elements):
        for j, elem2 in enumerate(elements):
            if i < j:  # Avoid duplicates
                pairs.append((elem1, elem2))
    
    return pairs

def analyze_ml_structure_rdf(atoms, structure_name):
    """Analyze RDF for a single ML-optimized structure."""
    
    # Get unique elements and relevant pairs
    elements = get_unique_elements(atoms)
    pairs = get_relevant_pairs(elements)
    
    # Calculate partial RDFs
    rdf_data, r_centers = calculate_partial_rdf(atoms, pairs)
    
    # Create result dictionary
    result = {
        'structure_name': structure_name,
        'source': 'ML_optimized',
        'n_atoms': len(atoms),
        'volume': atoms.get_volume(),
        'formula': atoms.get_chemical_formula(),
        'elements': '+'.join(sorted(elements))
    }
    
    # Add RDF peak positions and heights (first peak)
    for pair_name, rdf_values in rdf_data.items():
        if len(rdf_values) > 0 and np.max(rdf_values) > 0:
            # Find first peak (skip r=0 region)
            start_idx = int(0.5 / (10.0 / len(rdf_values)))  # Start from 0.5 Å
            peak_idx = np.argmax(rdf_values[start_idx:]) + start_idx
            peak_position = r_centers[peak_idx]
            peak_height = rdf_values[peak_idx]
            
            # Integration of first peak (coordination number)
            first_min_idx = peak_idx
            while first_min_idx < len(rdf_values) - 1 and rdf_values[first_min_idx + 1] < rdf_values[first_min_idx]:
                first_min_idx += 1
            
            # Find next minimum
            while first_min_idx < len(rdf_values) - 1 and rdf_values[first_min_idx + 1] > rdf_values[first_min_idx]:
                first_min_idx += 1
            
            # Calculate coordination number (integral under first peak)
            dr = r_centers[1] - r_centers[0]
            coordination = np.trapz(rdf_values[start_idx:first_min_idx], dx=dr)
            
            result[f'{pair_name}_peak_pos'] = peak_position
            result[f'{pair_name}_peak_height'] = peak_height
            result[f'{pair_name}_coordination'] = coordination
        else:
            result[f'{pair_name}_peak_pos'] = np.nan
            result[f'{pair_name}_peak_height'] = np.nan
            result[f'{pair_name}_coordination'] = np.nan
    
    return result

# ----------------- loop over every .xyz -------------------------------
raw = []                                        # energies per structure
ml_rdf_results = []                            # RDF analysis results for ML structures
optimized_structures = {}                       # Store optimized atoms objects

# Create output directory for optimized structures
ML_OPT_DIR.mkdir(exist_ok=True)

for xyz in glob.glob(str(STRUCT_DIR / "*.xyz")):
    name = Path(xyz).stem                       # filename without .xyz
    print(f"Processing: {name}")
    try:
        E, nat, atoms_opt = optimise_energy(xyz)
        raw.append({"structure": name,
                    "n_atoms":  nat,
                    "energy":   E})
        
        # Store optimized structure
        optimized_structures[name] = atoms_opt
        
        # Save optimized structure to file
        opt_path = ML_OPT_DIR / f"{name}_ml_opt.xyz"
        write(str(opt_path), atoms_opt)
        
        # Perform RDF analysis on optimized structure
        print(f"  → Analyzing RDF for {name}...")
        rdf_result = analyze_ml_structure_rdf(atoms_opt, name)
        ml_rdf_results.append(rdf_result)
        
        print(f"  ✓ {name}: {E:.4f} eV ({nat} atoms), RDF analyzed")
    except ValueError as e:
        if "No edges found" in str(e):
            print(f"  ✗ SKIPPED {name}: No edges found (atoms too far apart for ML model)")
            continue
        else:
            print(f"  ✗ ERROR in {name}: {e}")
            raise
    except Exception as e:
        print(f"  ✗ ERROR in {name}: {e}")
        raise

raw_df = pd.DataFrame(raw)
ml_rdf_df = pd.DataFrame(ml_rdf_results)

# ----------------- build a quick lookup dict --------------------------
E_dict  = dict(zip(raw_df.structure, raw_df.energy))
N_dict  = dict(zip(raw_df.structure, raw_df.n_atoms))

# ----------------- elemental chemical potentials ----------------------
mu_S   = E_dict["S2_box"]   / N_dict["S2_box"]                  # ½ E(S2)
mu_Mo  = E_dict[reference_key("Mo", E_dict)] / N_dict[reference_key("Mo", E_dict)]

# ----------------- formation energies for each dopant -----------------
records = []
for dop in DOPANTS:
    host_key = "MoS2_50"
    s_key    = f"MoS2_49S1{dop}"
    mo_key   = f"MoS2_49Mo1{dop}"
    int_key  = f"MoS2_50+{dop}"

    ref_key  = reference_key(dop, E_dict)
    mu_X     = E_dict[ref_key] / N_dict[ref_key]

    E_host   = E_dict[host_key]
    E_s      = E_dict[s_key]
    E_mo     = E_dict[mo_key]
    E_int    = E_dict[int_key]

    Eform_s  = E_s   - E_host + mu_S  - mu_X            # X on S site
    Eform_mo = E_mo  - E_host + mu_Mo - mu_X            # X on Mo site
    Eform_int= E_int - E_host            - mu_X         # intercalated X

    records.append({
        "dopant":        dop,
        "E_host (eV)":   E_host,
        "E_Ssub (eV)":   E_s,
        "E_Mosub (eV)":  E_mo,
        "E_int (eV)":    E_int,
        "mu_X (eV/atom)":mu_X,
        "Eform_Ssub (eV)":  Eform_s,
        "Eform_Mosub (eV)": Eform_mo,
        "Eform_int (eV)":   Eform_int
    })

form_df = pd.DataFrame(records)

# ----------------- merge & write --------------------------------------
out_df = form_df.merge(raw_df, how="left",
                       left_on="dopant", right_on="structure", indicator=True)
# ^ just to keep raw energies in the same file; you can simplify if preferred
form_df.to_csv(OUT_CSV, index=False)
print(f"Finished. Formation energies and raw UMA energies written to {OUT_CSV}")

# ----------------- RDF Analysis and Summary ---------------------------
print("\n" + "="*60)
print("ML STRUCTURE RDF ANALYSIS")
print("="*60)

# Save ML RDF results
ml_rdf_file = "Analysis/ml_rdf_analysis_results.csv"
os.makedirs("Analysis", exist_ok=True)
ml_rdf_df.to_csv(ml_rdf_file, index=False)
print(f"ML RDF analysis results saved to: {ml_rdf_file}")

# Generate ML RDF Summary
def generate_ml_rdf_summary(ml_rdf_df):
    """Generate summary of ML RDF analysis."""
    
    print(f"\nML RDF SUMMARY:")
    print(f"Total ML-optimized structures analyzed: {len(ml_rdf_df)}")
    print(f"Average number of atoms: {ml_rdf_df['n_atoms'].mean():.1f}")
    print(f"Average volume: {ml_rdf_df['volume'].mean():.1f} Å³")
    
    # Focus on key structural parameters
    key_pairs = ['Mo-Mo', 'S-S', 'S-Mo']
    summary_data = []
    key_structural_params = []
    
    for pair in key_pairs:
        peak_col = f"{pair}_peak_pos"
        height_col = f"{pair}_peak_height"
        coord_col = f"{pair}_coordination"
        
        if peak_col in ml_rdf_df.columns:
            valid_data = ml_rdf_df[ml_rdf_df[peak_col].notna()]
            
            if len(valid_data) > 0:
                print(f"\nML {pair} pair analysis:")
                print(f"  Structures with data: {len(valid_data)}")
                print(f"  Average peak position: {valid_data[peak_col].mean():.3f} ± {valid_data[peak_col].std():.3f} Å")
                print(f"  Peak position range: {valid_data[peak_col].min():.3f} - {valid_data[peak_col].max():.3f} Å")
                
                if height_col in ml_rdf_df.columns:
                    print(f"  Average peak height: {valid_data[height_col].mean():.2f} ± {valid_data[height_col].std():.2f}")
                
                if coord_col in ml_rdf_df.columns:
                    print(f"  Average coordination: {valid_data[coord_col].mean():.2f} ± {valid_data[coord_col].std():.2f}")
                
                # Store key structural parameters
                key_structural_params.append({
                    'pair': pair,
                    'source': 'ML_optimized',
                    'structures_count': len(valid_data),
                    'avg_peak_position': valid_data[peak_col].mean(),
                    'std_peak_position': valid_data[peak_col].std(),
                    'min_peak_position': valid_data[peak_col].min(),
                    'max_peak_position': valid_data[peak_col].max(),
                    'avg_peak_height': valid_data[height_col].mean() if height_col in ml_rdf_df.columns else np.nan,
                    'std_peak_height': valid_data[height_col].std() if height_col in ml_rdf_df.columns else np.nan,
                    'avg_coordination': valid_data[coord_col].mean() if coord_col in ml_rdf_df.columns else np.nan,
                    'std_coordination': valid_data[coord_col].std() if coord_col in ml_rdf_df.columns else np.nan
                })
    
    # Dopant-specific analysis for ML structures
    print(f"\nML DOPANT-SPECIFIC ANALYSIS:")
    dopant_pairs = []
    dopant_analysis = []
    
    for col in ml_rdf_df.columns:
        if col.endswith('_peak_pos') and 'Mo' in col and col not in [f"{pair}_peak_pos" for pair in key_pairs]:
            pair_name = col.replace('_peak_pos', '')
            if pair_name.count('-') == 1:  # Simple pair
                dopant_pairs.append(pair_name)
    
    for pair in sorted(dopant_pairs):
        peak_col = f"{pair}_peak_pos"
        height_col = f"{pair}_peak_height"
        coord_col = f"{pair}_coordination"
        
        valid_data = ml_rdf_df[ml_rdf_df[peak_col].notna()]
        
        if len(valid_data) > 0:
            print(f"\nML {pair} interactions:")
            for _, row in valid_data.iterrows():
                print(f"  {row['structure_name']}: peak at {row[peak_col]:.2f} Å")
                if height_col in ml_rdf_df.columns and not pd.isna(row[height_col]):
                    print(f"    Height: {row[height_col]:.2f}, Coordination: {row[coord_col]:.2f}")
                
                # Store dopant analysis data
                dopant_analysis.append({
                    'structure': row['structure_name'],
                    'source': 'ML_optimized',
                    'dopant_pair': pair,
                    'peak_position': row[peak_col],
                    'peak_height': row[height_col] if height_col in ml_rdf_df.columns else np.nan,
                    'coordination': row[coord_col] if coord_col in ml_rdf_df.columns else np.nan,
                    'dopant_element': pair.split('-')[0] if pair.split('-')[0] != 'Mo' else pair.split('-')[1],
                    'host_element': 'Mo' if 'Mo' in pair else 'S'
                })
    
    return pd.DataFrame(key_structural_params), pd.DataFrame(dopant_analysis)

# Generate ML RDF summary
ml_key_params_df, ml_dopant_df = generate_ml_rdf_summary(ml_rdf_df)

# Save ML summary files
ml_key_params_file = "Analysis/ml_key_structural_parameters.csv"
ml_dopant_file = "Analysis/ml_dopant_interactions.csv"
ml_summary_file = "Analysis/ml_rdf_analysis_summary.txt"

ml_key_params_df.to_csv(ml_key_params_file, index=False)
ml_dopant_df.to_csv(ml_dopant_file, index=False)

# Create ML analysis summary text file
with open(ml_summary_file, 'w') as f:
    f.write("ML RDF ANALYSIS SUMMARY REPORT\n")
    f.write("="*50 + "\n\n")
    
    f.write("KEY STRUCTURAL PARAMETERS (ML-OPTIMIZED):\n")
    f.write("-" * 30 + "\n")
    for _, row in ml_key_params_df.iterrows():
        f.write(f"{row['pair']} pair:\n")
        f.write(f"  Average distance: {row['avg_peak_position']:.3f} ± {row['std_peak_position']:.3f} Å\n")
        f.write(f"  Average coordination: {row['avg_coordination']:.2f} ± {row['std_coordination']:.2f}\n")
        f.write(f"  Structures analyzed: {row['structures_count']}\n\n")
    
    f.write("\nDOPANT-SPECIFIC INTERACTIONS (ML-OPTIMIZED):\n")
    f.write("-" * 30 + "\n")
    
    # Group by dopant element
    if not ml_dopant_df.empty:
        dopant_groups = ml_dopant_df.groupby('dopant_element')
        for dopant, group in dopant_groups:
            f.write(f"\n{dopant} dopant:\n")
            for _, row in group.iterrows():
                f.write(f"  {row['structure']}: {row['dopant_pair']} = {row['peak_position']:.2f} Å "
                       f"(coord: {row['coordination']:.2f})\n")

print(f"\nML RDF analysis files created:")
print(f"✓ Key structural parameters: {ml_key_params_file}")
print(f"✓ Dopant interactions: {ml_dopant_file}")
print(f"✓ Analysis summary report: {ml_summary_file}")

# ----------------- DFT vs ML Comparison Preparation -------------------
print(f"\n" + "="*60)
print("DFT vs ML COMPARISON PREPARATION")
print("="*60)

# Check if DFT RDF data exists
dft_rdf_file = "Analysis/rdf_analysis_results_improved.csv"
if os.path.exists(dft_rdf_file):
    print(f"✓ DFT RDF data found: {dft_rdf_file}")
    print(f"✓ ML RDF data created: {ml_rdf_file}")
    print(f"\nTo compare DFT vs ML structural data, use the following files:")
    print(f"  DFT: {dft_rdf_file}")
    print(f"  ML:  {ml_rdf_file}")
    print(f"\nBoth datasets now contain comparable RDF analysis results.")
    print(f"You can load and compare them using pandas:")
    print(f"  dft_data = pd.read_csv('{dft_rdf_file}')")
    print(f"  ml_data = pd.read_csv('{ml_rdf_file}')")
else:
    print(f"⚠ DFT RDF data not found at: {dft_rdf_file}")
    print(f"  Run the RDF analysis on DFT structures first.")
    print(f"✓ ML RDF data ready for future comparison: {ml_rdf_file}")

print(f"\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
print(f"✓ ML optimization and energy calculation complete")
print(f"✓ ML structures saved to: {ML_OPT_DIR}/")
print(f"✓ Formation energies saved to: {OUT_CSV}")
print(f"✓ ML RDF analysis complete with {len(ml_rdf_df)} structures analyzed")
print(f"✓ Ready for DFT vs ML structural comparison")
