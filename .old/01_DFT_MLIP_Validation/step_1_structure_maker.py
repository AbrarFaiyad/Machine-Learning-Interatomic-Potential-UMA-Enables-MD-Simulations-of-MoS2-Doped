#!/usr/bin/env python
"""
Create:
  • one 50‑atom MoS2 super‑cell  (MoS2_50.xyz)
  • three defect variants for each dopant         (MoS2_49S1X.xyz, MoS2_49Mo1X.xyz, MoS2_50+X.xyz)
  • one reference structure for every element in {Mo, S,  dopants} (e.g. Mo_bcc.xyz, Au_fcc.xyz, N2_box.xyz)

All files end up in ./formation_energy_structures/
"""

import os
from pathlib import Path
import numpy as np
from ase import Atoms, Atom
from ase.build import (
    bulk,
    molecule,
    make_supercell,
)
from ase.io import read, write

# ----------------------------------------------------------------------

# Always resolve paths relative to the repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
CIF_FILE = REPO_ROOT / "00_Structure" / "Mo2S4.cif"
SUPERCELL_SIZE = (2, 2, 1)
OUTPUT_DIR = Path(__file__).resolve().parent / "formation_energy_structures"

DOPANTS = [
    'C', 'N', 'O', 'F', 'B', 'P', 'Se', 'Te', 'Cl', 'Si', 'Li', 'Na', 'Al', 'Zn', 'V', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Ta', 'W', 'Re', 'Ir', 'Pt', 'Au', 'Ti'
]

# hexagonal‑to‑orthorhombic transform
HEX2ORTHO = np.array([[ 1, -1,  0],
                      [ 1,  1,  0],
                      [ 0,  0,  1]])

# ----------------------------------------------------------------------
# -------- reference‑phase builder ------------------------------------
LAT_LOOKUP = {
    #   element : (lattice,   lattice_parameter[Å])
    "Mo" : ("bcc", 3.15),
    "Nb" : ("bcc", 3.30),
    "Ta" : ("bcc", 3.30),
    "W"  : ("bcc", 3.16),
    "V"  : ("bcc", 3.03),
    "Fe" : ("bcc", 2.87),
    

    "Cu" : ("fcc", 3.61),
    "Ag" : ("fcc", 4.09),
    "Au" : ("fcc", 4.08),
    "Ni" : ("fcc", 3.52),
    "Pd" : ("fcc", 3.89),
    "Pt" : ("fcc", 3.92),
    "Al" : ("fcc", 4.05),
    "Ir" : ("fcc", 3.84),
    "Rh" : ("fcc", 3.80),

    "Ti" : ("hcp", 2.95), # a (c will be 1.633*a)
    "Co" : ("hcp", 2.51),   
    "Zn" : ("hcp", 2.66),
    "Cd" : ("hcp", 2.98),
    "Ru" : ("hcp", 2.71),

    "Re" : ("hcp", 2.76),

    "C"  : ("diamond", 3.57),
    "Si" : ("diamond", 5.43),
    "B"  : ("diamond", 4.75),   # β‑rhombohedral would be better, but OK

    # solid forms for Se and Te
    "Se" : ("hcp", 4.36),   # hcp, a=4.36 Å (approximate)
    "Te" : ("hcp", 4.45),   # hcp, a=4.45 Å (approximate)

    # Added missing elements
    "Na" : ("bcc", 4.23),
    "Li" : ("bcc", 3.49),
    "Mn" : ("bcc", 8.91),  # α-Mn is complex, bcc is a common approximation

    # di‑atomic non‑metals handled separately below
}

DIATOMIC = {"N", "O", "F", "Cl", "S"}


def reference_structure(element: str) -> Atoms:
    """Return an ASE Atoms object for the elemental reference phase."""
    Z = element.capitalize()

    # --- diatomic gases / molecular references -----------------------
    if Z in DIATOMIC:
        box = 12.0
        if Z == "S":
            # S8 ring (cyclooctasulfur)
            # Coordinates from literature, centered at origin
            s8_coords = np.array([
                [ 2.05,  0.00,  0.00],
                [ 0.63,  1.93,  0.00],
                [-1.66,  1.19,  0.00],
                [-1.66, -1.19,  0.00],
                [ 0.63, -1.93,  0.00],
                [ 3.68, -1.19,  0.00],
                [ 3.68,  1.19,  0.00],
                [ 2.05,  2.38,  0.00],
            ])
            mol = Atoms("S8", positions=s8_coords)
            mol.set_cell([box, box, box])
            mol.center()
            return mol
        else:
            mol = molecule(f"{Z}2")  # ASE built‑in (N2, O2, F2, Cl2)
            mol.set_cell([box, box, box])
            mol.center()
            return mol

    # --- phosphorus: use P4 molecule (white phosphorus) --------------
    if Z == "P":
        # Coordinates for P4 tetrahedron (white phosphorus)
        box = 12.0
        # Tetrahedral coordinates, centered at origin
        a = 2.21  # P-P bond length in Å
        coords = [
            [ a,  a,  a],
            [-a, -a,  a],
            [-a,  a, -a],
            [ a, -a, -a],
        ]
        mol = Atoms("P4", positions=coords)
        mol.set_cell([box, box, box])
        mol.center()
        return mol

    # --- tabulated bulk lattices -------------------------------------
    if Z in LAT_LOOKUP:
        phase, a = LAT_LOOKUP[Z]
        if phase == "bcc":
            return bulk(Z, "bcc", a=a, cubic=True)            # 2 atoms
        if phase == "fcc":
            return bulk(Z, "fcc", a=a, cubic=True)            # 4 atoms
        if phase == "hcp":
            return bulk(Z, "hcp", a=a, c=1.633*a)             # 2 atoms
        if phase == "diamond":
            return bulk(Z, "diamond", a=a, cubic=True)        # 2 atoms

    # --- fallback: isolated atom in a vacuum box ----------------------
    box = 12.0
    at = Atoms(Z, cell=[box, box, box], pbc=False)
    at.center()
    return at

# ----------------------------------------------------------------------
# -------- MoS2 super‑cell + defects -----------------------------------
def build_supercell():
    atoms = read(CIF_FILE)
    ortho = make_supercell(atoms, HEX2ORTHO)
    sup   = ortho.repeat(SUPERCELL_SIZE)
    sup.set_pbc(True)
    out = OUTPUT_DIR / "MoS2_50.xyz"
    if not out.exists():
        write(out, sup)
        print(f"  • pristine cell  → {out}  ({len(sup)} atoms)")
    return sup

def make_derivatives(cell, dop):
    """Return (S‑sub, Mo‑sub, intercalated) structures."""
    # ----- S substitution -----
    s_sub = cell.copy()
    s_indices = [i for i,a in enumerate(s_sub) if a.symbol == "S"]
    if s_indices:
        mid_z = s_sub.get_cell()[2,2] / 2
        tgt = min(s_indices, key=lambda i: abs(s_sub[i].z - mid_z))
        s_sub[tgt].symbol = dop

    # ----- Mo substitution ----
    mo_sub = cell.copy()
    mo_indices = [i for i,a in enumerate(mo_sub) if a.symbol == "Mo"]
    if mo_indices:
        tgt = min(mo_indices, key=lambda i: mo_sub[i].z)
        mo_sub[tgt].symbol = dop

    # ----- intercalation -------
    inter = cell.copy()
    centre = 0.5*(cell.get_cell()[0]+cell.get_cell()[1]+cell.get_cell()[2])
    inter.append(Atom(dop, position=centre))

    return s_sub, mo_sub, inter

def write_structures(dop, s_sub, mo_sub, inter):
    write(OUTPUT_DIR / f"MoS2_49S1{dop}.xyz",  s_sub)
    write(OUTPUT_DIR / f"MoS2_49Mo1{dop}.xyz", mo_sub)
    write(OUTPUT_DIR / f"MoS2_50+{dop}.xyz",   inter)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)

    # -------- host & defects ------------------------
    host = build_supercell()
    all_elements = {"Mo", "S", *DOPANTS}       # references we need

    for dop in DOPANTS:
        s_sub, mo_sub, inter = make_derivatives(host, dop)
        write_structures(dop, s_sub, mo_sub, inter)
        print(f"  • {dop:>2s}‑doped variants written")

    # -------- reference phases ----------------------
    for el in sorted(all_elements):
        ref = reference_structure(el)
        # craft a filename that encodes the phase type roughly
        if el in DIATOMIC:
            fname = f"{el}2_box.xyz"
        elif el in LAT_LOOKUP:
            phase, _ = LAT_LOOKUP[el]
            fname = f"{el}_{phase}.xyz"
        else:
            fname = f"{el}_atom_box.xyz"  # fallback
        out = OUTPUT_DIR / fname
        if not out.exists():
            write(out, ref)
            print(f"  • reference {el:<2} → {out}")

    print("\nAll pristine, defect and reference structures are in:", OUTPUT_DIR.resolve())
