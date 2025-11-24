"""
Script 7: NPT cooling from 1000K to 300K
Reads: 6_{filename}
Outputs: 7_{filename}
"""

from ase.io import read, write, Trajectory  # Add Trajectory import
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from fairchem.core import pretrained_mlip, FAIRChemCalculator
from ase import units
import numpy as np
import torch
import glob
import os

# Find input structure (6_filename)
xyz_files = glob.glob("6_*.xyz")
if not xyz_files:
    raise FileNotFoundError("No 6_*.xyz file found!")
input_file = xyz_files[0]
base_name = os.path.splitext(os.path.basename(input_file))[0][2:]  # Remove "6_" prefix

print(f"Processing: {input_file}")
atoms = read(input_file)

# --- sanitize total charge & spin metadata from extxyz ---
for key in ("charge", "spin", "spin_multiplicity"):
    if key in atoms.info:
        try:
            atoms.info[key] = int(atoms.info[key])
        except Exception:
            if key == "charge":
                atoms.info[key] = 0
            else:
                atoms.info.pop(key, None)

atoms.info.setdefault("charge", 0)

if "charges" in atoms.arrays:
    atoms.set_initial_charges(np.zeros(len(atoms)))
# --- end sanitize ---

# Setup calculator
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device)
atoms.calc = FAIRChemCalculator(predictor, task_name="omat")

# NPT cooling parameters
start_temp = 1000.0  # K
end_temp = 300.0     # K
cooling_time_ps = 20.0  # ps
timestep_fs = 1.0
pressure_au = 1.01325e5 * units.Pascal
bulk_modulus_GPa = 80
compressibility_au = 1.0 / (bulk_modulus_GPa * units.GPa)

# Calculate cooling parameters
total_steps = int(cooling_time_ps * 1000 / timestep_fs)  # ps to fs to steps
cooling_rate = (end_temp - start_temp) / cooling_time_ps  # K/ps (negative)

print(f"Cooling from {start_temp} K to {end_temp} K over {cooling_time_ps} ps")
print(f"Cooling rate: {cooling_rate:.1f} K/ps")
print(f"Total steps: {total_steps}")

MaxwellBoltzmannDistribution(atoms, temperature_K=start_temp)
Stationary(atoms)

# Setup NPT with cooling
dyn = Inhomogeneous_NPTBerendsen(
    atoms,
    timestep_fs * units.fs,
    temperature_K=start_temp,
    pressure_au=pressure_au,
    taut=100 * units.fs,
    taup=500 * units.fs,
    compressibility_au=compressibility_au,
    mask=(1, 1, 1),
)

# Setup trajectory output
traj = Trajectory(f"cooling_{base_name}.traj", "w", atoms)
dyn.attach(traj.write, interval=100)  # Save every 100 steps

# Temperature ramping function
def update_temperature():
    current_time_ps = dyn.nsteps * timestep_fs / 1000.0
    target_temp = start_temp + cooling_rate * current_time_ps
    if target_temp < end_temp:
        target_temp = end_temp
    
    dyn.set_temperature(target_temp)
    
    # Log detailed information to file
    temp = atoms.get_temperature()
    cell_volume = atoms.get_volume()  # Å^3
    total_mass = np.sum(atoms.get_masses()) * 1.66054e-24  # amu -> grams
    volume_cm3 = cell_volume * 1e-24  # Å^3 -> cm^3
    density = total_mass / volume_cm3 if volume_cm3 > 0 else 0.0
    
    # Calculate energies
    ke = atoms.get_kinetic_energy()   # eV
    pe = atoms.get_potential_energy() # eV
    total_energy = ke + pe            # eV
    
    step = dyn.nsteps
    with open("density_log.txt", "a") as f:
        # step  time(ps)  T_inst(K)  T_target(K)  V(Å^3)  rho(g/cm^3)  KE(eV)  PE(eV)  E_tot(eV)
        f.write(
            f"{step} {current_time_ps:.4f} {temp:.2f} {target_temp:.2f} "
            f"{cell_volume:.2f} {density:.6f} {ke:.6f} {pe:.6f} {total_energy:.6f}\n"
        )
    
    if dyn.nsteps % 100 == 0:
        print(f"Step {dyn.nsteps}: Target={target_temp:.1f}K, Actual={temp:.1f}K")

dyn.attach(update_temperature, interval=10)  # Log every 10 steps

# Run cooling
print("Starting NPT cooling...")
dyn.run(total_steps)

# Close trajectory
traj.close()

# Save output
output_file = f"7_{base_name}.xyz"
write(output_file, atoms)
print(f"Cooling completed. Final temperature: {atoms.get_temperature():.1f} K")
print(f"Output saved as: {output_file}")
