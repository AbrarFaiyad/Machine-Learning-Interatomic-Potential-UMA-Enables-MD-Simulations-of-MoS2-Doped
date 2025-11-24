"""
Script 2: NPT equilibration with density convergence checks
Reads: 1_{filename}
Outputs: 2_{filename}
"""

from ase.io import read, write
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import units
import numpy as np
import torch
import glob
import os


# Always resolve paths relative to the repo root
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = REPO_ROOT / "02_MD_heating_cooling"
os.chdir(INPUT_DIR)
xyz_files = glob.glob("1_*.xyz")
if not xyz_files:
    raise FileNotFoundError("No 1_*.xyz file found in 02_MD_heating_cooling!")
input_file = xyz_files[0]
base_name = os.path.splitext(os.path.basename(input_file))[0][2:]  # Remove "1_" prefix

print(f"Processing: {input_file}")
atoms = read(input_file)

# --- sanitize total charge & spin metadata from extxyz ---
# FAIRChem expects Python ints (not numpy integers) for these global fields.
for key in ("charge", "spin", "spin_multiplicity"):
    if key in atoms.info:
        try:
            atoms.info[key] = int(atoms.info[key])
        except Exception:
            # If something odd came through, default to neutral / unset.
            if key == "charge":
                atoms.info[key] = 0
            else:
                atoms.info.pop(key, None)

# If no charge was provided, make it explicitly neutral
atoms.info.setdefault("charge", 0)

# Optional: if per-atom charges snuck in and you don't want them, zero them out
if "charges" in atoms.arrays:
    import numpy as np
    atoms.set_initial_charges(np.zeros(len(atoms)))
# --- end sanitize ---

# Setup calculator
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
atoms.calc = FAIRChemCalculator(predictor, task_name="omat")

# NPT parameters
target_temp = 300.0  # K
pressure_au = 1.01325e5 * units.Pascal
timestep_fs = 1.0
npt_steps = 10000
check_window = 500
bulk_modulus_GPa = 80
compressibility_au = 1.0 / (bulk_modulus_GPa * units.GPa)

MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp)
Stationary(atoms)

converged = False
cycle = 0

while not converged:
    cycle += 1
    print(f"NPT cycle {cycle}")
    
    # Setup NPT
    dyn = Inhomogeneous_NPTBerendsen(
        atoms,
        timestep_fs * units.fs,              
        temperature_K=target_temp,           
        pressure_au=pressure_au,             
        taut=100 * units.fs,
        taup=500 * units.fs,
        compressibility_au=compressibility_au,
        mask=(1, 1, 1),
    )
    densities = []
    
    def log_density():
        volume = atoms.get_volume()  # Å^3
        total_mass = np.sum(atoms.get_masses()) * 1.66054e-24  # amu -> g
        volume_cm3 = volume * 1e-24  # Å^3 -> cm^3
        density = total_mass / volume_cm3 if volume_cm3 > 0 else 0.0
        densities.append(density)
    
    dyn.attach(log_density, interval=10)
    dyn.run(npt_steps)
    
    # Check density convergence in last 500 steps
    if len(densities) >= check_window // 10:
        recent_densities = densities[-(check_window // 10):]
        density_change = max(recent_densities) - min(recent_densities)
        
        print(f"Density change in last {check_window} steps: {density_change:.4f} g/cm³")
        
        if density_change <= 0.1:
            converged = True
            print("NPT density converged!")

# Save output
output_file = f"2_{base_name}.xyz"
write(output_file, atoms)
print(f"Output saved as: {output_file}")
