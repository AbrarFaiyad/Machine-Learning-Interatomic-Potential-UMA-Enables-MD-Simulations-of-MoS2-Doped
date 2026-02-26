"""Helpers for constructing and running Quantum ESPRESSO relaxations."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from ase.io import read

from .structures import DEFAULT_DOPANTS


@dataclass
class QEInputs:
    """Container for QE control/system/electron blocks."""

    control: Dict[str, object] = field(default_factory=dict)
    system: Dict[str, object] = field(default_factory=dict)
    electrons: Dict[str, object] = field(default_factory=dict)
    kpoints: Tuple[int, int, int] = (6, 6, 4)


@dataclass
class QERunResult:
    """Energy and metadata returned from a QE run."""

    structure: str
    n_atoms: int
    energy: float


DEFAULT_PSEUDOS: Dict[str, str] = {
    "Mo": "Mo.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "S": "S.pbe-n-kjpaw_psl.1.0.0.UPF",
    "C": "C.pbe-n-kjpaw_psl.1.0.0.UPF",
    "N": "N.pbe-n-kjpaw_psl.1.0.0.UPF",
    "O": "O.pbe-n-kjpaw_psl.1.0.0.UPF",
    "F": "F.pbe-n-kjpaw_psl.1.0.0.UPF",
    "B": "B.pbe-n-kjpaw_psl.1.0.0.UPF",
    "P": "P.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Se": "Se.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Te": "Te.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cl": "Cl.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Si": "Si.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Li": "Li.pbe-s-kjpaw_psl.1.0.0.UPF",
    "Na": "Na.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Al": "Al.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Zn": "Zn.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "V": "V.pbe-spnl-kjpaw_psl.1.0.0.UPF",
    "Mn": "Mn.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Fe": "Fe.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Co": "Co.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ni": "Ni.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Cu": "Cu.pbe-dn-kjpaw_psl.1.0.0.UPF",
    "Nb": "Nb.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ru": "Ru.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Rh": "Rh.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Pd": "Pd.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ag": "Ag.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Cd": "Cd.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Ta": "Ta.pbe-spfn-kjpaw_psl.1.0.0.UPF",
    "W": "W.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Re": "Re.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ir": "Ir.pbe-n-kjpaw_psl.1.0.0.UPF",
    "Pt": "Pt.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Au": "Au.pbe-spn-kjpaw_psl.1.0.0.UPF",
    "Ti": "Ti.pbe-spn-kjpaw_psl.1.0.0.UPF",
}


def build_default_inputs(total_charge: float = 0.0) -> QEInputs:
    """Return QE input dictionaries reflecting the original script settings."""

    system = {
        "ecutwfc": 124,
        "ecutrho": 843,
        "tot_charge": total_charge,
        "occupations": "smearing",
        "smearing": "fd",
        "degauss": 0.02,
    }
    control = {
        "calculation": "relax",
        "outdir": "./tmp/",
        "disk_io": "none",
        "wf_collect": False,
        "nstep": 1200,
    }
    electrons = {
        "conv_thr": 1e-4,
        "electron_maxstep": 1200,
        "mixing_beta": 0.3,
        "diagonalization": "rmm-diis",
    }
    return QEInputs(control=control, system=system, electrons=electrons)


def starting_magnetisation(element: str) -> float:
    """Heuristic starting magnetisation copied from the QE helper script."""

    non_metals = {"B", "C", "N", "O", "F", "P", "Cl", "Se", "Te", "S"}
    early = {"V", "Cr", "Mn"}
    mid = {"Fe", "Co", "Ni"}
    late = {"Cu", "Zn", "Ag", "Au", "Pd", "Pt"}
    if element in non_metals or element == "Mo":
        return 0.0
    if element in early:
        return 0.7
    if element in mid:
        return 0.45
    if element in late:
        return 0.10
    return 0.0


def run_relaxation(
    xyz_path: Path | str,
    pseudo_dir: Path | str,
    inputs: QEInputs | None = None,
    pseudo_table: Dict[str, str] | None = None,
    profile_command: str | None = None,
) -> QERunResult:
    """Run a relax calculation with ASE's Espresso calculator.

    The function keeps the procedural defaults from ``step_2_run_qe_single.py``
    but is library-friendly: all paths are arguments and the caller may
    supply a custom MPI command.
    """

    from ase.calculators.espresso import Espresso, EspressoProfile
    from openbabel import pybel
    import spglib as spg

    path = Path(xyz_path)
    atoms = read(path)
    inputs = inputs or build_default_inputs()
    pseudo_table = pseudo_table or DEFAULT_PSEUDOS

    elements = tuple(dict.fromkeys(atoms.get_chemical_symbols()))
    pseudopotentials = {el: pseudo_table[el] for el in elements}

    cell = (atoms.cell.array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
    grid, mapping = spg.get_ir_reciprocal_mesh(inputs.kpoints, cell, is_shift=(0, 0, 0))
    n_kirr = len(np.unique(mapping))

    nprocs = int(
        Path("/proc/self/status").read_text().split("Threads:")[-1].splitlines()[0].strip()
    )
    npool = min(max(1, n_kirr), nprocs)
    while npool > 1 and nprocs % npool != 0:
        npool -= 1
    ndiag = max(1, nprocs // npool)

    command = profile_command or f"mpirun -np {nprocs} pw.x -npool {npool} -ndiag {ndiag}"
    profile = EspressoProfile(command=command, pseudo_dir=str(pseudo_dir))

    mol = next(pybel.readfile("xyz", str(path)))
    inputs.system["tot_charge"] = mol.charge
    inputs.system["degauss"] = inputs.system.get("degauss", 0.02)

    for idx, element in enumerate(elements, 1):
        inputs.system[f"starting_magnetization({idx})"] = starting_magnetisation(element)

    calc = Espresso(
        profile=profile,
        pseudopotentials=pseudopotentials,
        input_data={"control": inputs.control, "system": inputs.system, "electrons": inputs.electrons},
        kpts=inputs.kpoints,
        parallel={"npool": npool, "ndiag": ndiag},
    )
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    return QERunResult(structure=path.stem, n_atoms=len(atoms), energy=float(energy))


def formation_energy(
    energies: Dict[str, float],
    n_atoms: Dict[str, int],
    dopant: str,
    host_key: str = "MoS2_50",
    sulfur_ref: str = "S2_box",
) -> Dict[str, float]:
    """Compute formation energies for the three defect motifs."""

    def ref_key(element: str) -> str:
        candidates = [
            f"{element}_{phase}" for phase in ["bcc", "fcc", "hcp", "diamond", "atom_box"]
        ] + [f"{element}2_box"]
        for candidate in candidates:
            if candidate in energies:
                return candidate
        raise KeyError(element)

    mu_s = energies[sulfur_ref] / n_atoms[sulfur_ref]
    mu_mo = energies[ref_key("Mo")] / n_atoms[ref_key("Mo")]
    mu_x = energies[ref_key(dopant)] / n_atoms[ref_key(dopant)]

    host = energies[host_key]
    ssub = energies[f"MoS2_49S1{dopant}"]
    mosub = energies[f"MoS2_49Mo1{dopant}"]
    inter = energies[f"MoS2_50+{dopant}"]

    return {
        "Eform_Ssub (eV)": ssub - host + mu_s - mu_x,
        "Eform_Mosub (eV)": mosub - host + mu_mo - mu_x,
        "Eform_int (eV)": inter - host - mu_x,
    }
