#!/usr/bin/env python
"""
Read dft_energies_raw.csv and build the concise formation‑energy
table (same columns as the ML CSV).

Run this *after* all QE jobs have finished.
"""

import pandas as pd
from pathlib import Path

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "01_DFT_MLIP_Validation" / "dft_energies_raw.csv"
OUT_CSV = REPO_ROOT / "01_DFT_MLIP_Validation" / "dft_formation_energies.csv"
DOPANTS = [
    "Au", "C", "Ta", "Pd", "V", "Pt", "Ag", "Re", "Ru", "Nb", "W", "Cu", "Al", "Ir", "Rh", "Co", "Zn", "Cd", "Si", "B", "Ti",
    "N", "O", "F", "Cl", "S", "Se", "Te"
]                     # keep in sync with ML script

DIATOMIC = {"N","O","F","Cl","S","Se","Te"}

def ref_key(el, E):
    cands=[f"{el}_{p}" for p in ["bcc","fcc","hcp","diamond","atom_box"]] + [f"{el}2_box"]
    for c in cands:
        if c in E: return c
    raise KeyError(el)

df = pd.read_csv(RAW_CSV)
E = dict(zip(df.structure, df.energy))
N = dict(zip(df.structure, df.n_atoms))

mu_Mo = E[ref_key("Mo",E)] / N[ref_key("Mo",E)]
mu_S  = E["S2_box"] / 2

rows=[]
for X in DOPANTS:
    try:
        host, ssub, mosub, inter = ("MoS2_50",f"MoS2_49S1{X}",
                                    f"MoS2_49Mo1{X}",f"MoS2_50+{X}")
        mu_X = E[ref_key(X,E)]/N[ref_key(X,E)]
        rows.append({
            "dopant":X,
            "E_host (eV)" :E[host],
            "E_Ssub (eV)" :E[ssub],
            "E_Mosub (eV)":E[mosub],
            "E_int (eV)"  :E[inter],
            "mu_X (eV/atom)":mu_X,
            "Eform_Ssub (eV)" :E[ssub]-E[host]+mu_S-mu_X,
            "Eform_Mosub (eV)":E[mosub]-E[host]+mu_Mo-mu_X,
            "Eform_int (eV)"  :E[inter]-E[host]-mu_X,
        })
    except KeyError as err:
        print(f"⚠  missing data for {X}: {err}")

pd.DataFrame(rows).to_csv(OUT_CSV,index=False)
print(f"✅  Wrote {OUT_CSV}")
