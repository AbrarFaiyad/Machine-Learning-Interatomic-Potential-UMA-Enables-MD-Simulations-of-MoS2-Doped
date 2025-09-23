"""
Script 3: Final NVT equilibration with convergence checks
Reads: 2_{filename}
Outputs: final_{filename}
"""

from ase.io import read, write
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import units
import numpy as np
import torch
import glob
import os

# Find input structure (2_filename)
xyz_files = glob.glob("2_*.xyz")
if not xyz_files:
    raise FileNotFoundError("No 2_*.xyz file found!")
input_file = xyz_files[0]
base_name = os.path.splitext(os.path.basename(input_file))[0][2:]  # Remove "2_" prefix

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

# NVT parameters
target_temp = 300.0  # K
timestep_fs = 1.0
nvt_steps = 10000
check_window = 500

MaxwellBoltzmannDistribution(atoms, temperature_K=target_temp)
Stationary(atoms)

converged = False
cycle = 0

while not converged:
    cycle += 1
    print(f"Final NVT cycle {cycle}")
    
    # Setup NVT
    dyn = NVTBerendsen(atoms, timestep_fs * units.fs, target_temp, taut=100 * units.fs)
    
    temps, energies = [], []
    
    def log_data():
        temps.append(atoms.get_temperature())
        energies.append(atoms.get_kinetic_energy() + atoms.get_potential_energy())
    
    dyn.attach(log_data, interval=10)
    dyn.run(nvt_steps)
    
    # Check convergence in last 500 steps
    if len(temps) >= check_window // 10:
        recent_temps = temps[-(check_window // 10):]
        recent_energies = energies[-(check_window // 10):]
        
        temp_std = np.std(recent_temps)
        energy_std = np.std(recent_energies)
        
        print(f"Temperature std: {temp_std:.2f} K, Energy std: {energy_std:.2f} eV")
        
        if temp_std <= 5.0 and energy_std <= 50.0:
            converged = True
            print("Final NVT converged!")

# Save final output
output_file = f"final_{base_name}.xyz"
write(output_file, atoms)
print(f"Final output saved as: {output_file}")
