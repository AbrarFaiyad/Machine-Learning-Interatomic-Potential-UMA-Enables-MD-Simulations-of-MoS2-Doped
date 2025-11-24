#!/usr/bin/env python
"""
Run one QE vc‑relax on <structure>.xyz that lives in the current
working directory and append the total energy to a master CSV
(./dft_energies_raw.csv *one level above* the run folder).

Usage (inside run‑folder):
    python run_qe_single.py               # auto‑detects the only .xyz
"""
import numpy as np
import spglib as spg 
import os, shutil, glob, fcntl, csv
from pathlib import Path
from ase.io import read
from ase.calculators.espresso import Espresso, EspressoProfile
from openbabel import pybel


# Always resolve paths relative to the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
PSEUDO_DIR = str(REPO_ROOT / "pseudopotentials")  # Place your pseudopotentials here
NPROCS     = int(os.environ.get("SLURM_NTASKS", "56"))

KPTS       = (6, 6, 4) # (3, 3, 2) / 'gamma' # k-point mesh for better sampling
ECUTWFC    = 124
ECUTRHO    = 843
FORCE_THR  = 5e-3
CONV_THR   = 1e-4

# ------------------- pseudopotentials ---------------------------------
PP = {
    'Mo': 'Mo.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'S':  'S.pbe-n-kjpaw_psl.1.0.0.UPF',
    'C':  'C.pbe-n-kjpaw_psl.1.0.0.UPF',
    'N':  'N.pbe-n-kjpaw_psl.1.0.0.UPF',
    'O':  'O.pbe-n-kjpaw_psl.1.0.0.UPF',
    'F':  'F.pbe-n-kjpaw_psl.1.0.0.UPF',
    'B':  'B.pbe-n-kjpaw_psl.1.0.0.UPF',
    'P':  'P.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Se': 'Se.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'Te': 'Te.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Cl': 'Cl.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Si': 'Si.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Li': 'Li.pbe-s-kjpaw_psl.1.0.0.UPF',
    'Na': 'Na.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Al': 'Al.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Zn': 'Zn.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'V':  'V.pbe-spnl-kjpaw_psl.1.0.0.UPF',
    'Mn': 'Mn.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Fe': 'Fe.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Co': 'Co.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Ni': 'Ni.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Cu': 'Cu.pbe-dn-kjpaw_psl.1.0.0.UPF',
    'Nb': 'Nb.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Ru': 'Ru.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Rh': 'Rh.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Pd': 'Pd.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Ag': 'Ag.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Cd': 'Cd.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Ta': 'Ta.pbe-spfn-kjpaw_psl.1.0.0.UPF',
    'W':  'W.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Re': 'Re.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Ir': 'Ir.pbe-n-kjpaw_psl.1.0.0.UPF',
    'Pt': 'Pt.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Au': 'Au.pbe-spn-kjpaw_psl.1.0.0.UPF',
    'Ti': 'Ti.pbe-spn-kjpaw_psl.1.0.0.UPF',
}

# ------------------- helper : starting magnetisation ------------------
def start_mag(el):
    non_metals={'B','C','N','O','F','P','Cl','Se','Te','S'}
    early={'V','Cr','Mn'}; mid={'Fe','Co','Ni'}; late={'Cu','Zn','Ag','Au','Pd','Pt'}
    return 0.00 if (el in non_metals or el=='Mo') else 0.70 if el in early \
           else 0.45 if el in mid else 0.10 if el in late else 0.00

# =================== MAIN =============================================
xyz_files = glob.glob("*.xyz")
if len(xyz_files) != 1:
    raise RuntimeError("Exactly one .xyz needed")
xyz = xyz_files[0]
atoms = read(xyz)

# --- spglib irreducible k-point count --------------------------------
cell = (atoms.cell.array,
        atoms.get_scaled_positions(),
        atoms.get_atomic_numbers())

mesh     = KPTS if isinstance(KPTS, tuple) else (1, 1, 1)
grid, mapping = spg.get_ir_reciprocal_mesh(mesh, cell, is_shift=(0,0,0))
n_kirr   = len(np.unique(mapping))



# Guarantee npool divides NPROCS; else fall back to 1
npool = min(n_kirr, NPROCS)
while NPROCS % npool != 0 and npool > 1:
    npool -= 1
ndiag = max(1, NPROCS // npool)

# --- build the run command *after* you know npool/ndiag --------------
PW_CMD = f"mpirun -np {NPROCS} pw.x -npool {npool} -ndiag {ndiag}"

PROFILE    = EspressoProfile(command=PW_CMD, pseudo_dir=PSEUDO_DIR)

name = Path(xyz).stem
print(f"▶  QE job for {name}")
elements = tuple(dict.fromkeys(atoms.get_chemical_symbols()))
pp_dict  = {el: PP[el] for el in elements}

# --- Predict charge and spin multiplicity using Open Babel ---
mol = next(pybel.readfile("xyz", str(xyz)))
tot_charge = mol.charge
spin_mult = mol.spin

# --- SYSTEM namelist --------------------------------------------------
system_block = {
    # plane-wave basis
    'ecutwfc'   : ECUTWFC,          # recommend ≥80 Ry with full-rel PPs
    'ecutrho'   : ECUTRHO,          # keep dual ≥12×, or use 'dual' keyword

    # spin-orbit / non-collinear setup
    #'noncolin'  : True,             # ⇐ non-collinear spinors
    #'lspinorb'  : True,             # ⇐ fully relativistic projectors

    # charge / smearing
    'tot_charge': tot_charge,
    'occupations': 'smearing',
    'smearing'  : 'fd',
    'degauss'   : 0.02 if any(x in elements for x in {'V','Nb','Mo','W','Re','Pt','Au','Ti','Mn','Fe','Co','Ni','Cu','Ru','Rh','Pd','Ag','Ta','Ir','Zn','Cd'}) else 0.005,             # broaden Fermi level (≈0.136 eV)

    # optional: add vdW, +U, etc. here
    #'vdw_corr' : 'grimme-d3'
}

# If your system is magnetic, add spinor starting guesses.
# QE ignores nspin when 'noncolin=.true.', so DO NOT set 'nspin'.
for i, el in enumerate(elements, 1):
    system_block[f'starting_magnetization({i})'] = start_mag(el)



input_data = {
    'control': {
        'calculation' : 'relax',
        'pseudo_dir'  : PSEUDO_DIR,
        'outdir'      : './tmp/',
        'disk_io'     : 'none',       # faster: no wave-function files
        'wf_collect'  : False,        # avoid post-processing gather
        'nstep'       : 1200,         # 1 000 is overkill for 50-atom cells
    },
    'system' : system_block,
    'electrons': {
        'conv_thr'        : CONV_THR,
        'electron_maxstep': 1200,      # 5 000 just wastes cycles
        'mixing_beta'     : 0.3,
        'diagonalization' : 'rmm-diis',   # ~2× faster than Davidson
    },
    'ions': {'ion_dynamics': 'bfgs'}
}


# --- pass PW_CMD directly (or rebuild EspressoProfile here) ----------

calc = Espresso(profile=PROFILE,
                pseudopotentials=pp_dict, input_data=input_data,
                kpts=KPTS, parallel={'npool': npool, 'ndiag': ndiag})
atoms.calc=calc

try:
    energy = atoms.get_potential_energy()   # eV per cell
except Exception as err:
    print(f"❌  QE failed: {err}")
    energy = None

# archive selected outputs
for f in ["espresso.pwo","espresso.pwi","espresso.err"]:
    if os.path.exists(f):
        shutil.copy2(f, f"{name}_{f}")

# ------------------- append to master CSV (thread‑safe) ---------------
master = REPO_ROOT / "01_DFT_MLIP_Validation" / "dft_energies_raw.csv"
header = ["structure","n_atoms","energy"]
row    = [name,len(atoms),energy]

with open(master, "a+", newline="") as fh:
    fcntl.flock(fh, fcntl.LOCK_EX)
    fh.seek(0,2)                       # append pos
    if fh.tell()==0:                   # file empty → write header
        csv.writer(fh).writerow(header)
    csv.writer(fh).writerow(row)
    fcntl.flock(fh, fcntl.LOCK_UN)

print(f"✔  Energy {energy:.6f} eV written to {master}")
