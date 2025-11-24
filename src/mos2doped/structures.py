"""Structure generation utilities for doped MoS₂.

This module consolidates the logic from ``step_1_structure_maker.py``
into reusable functions. All paths are decoupled from the working
folder so the functions can be called from Python code or the CLI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atom, Atoms
from ase.build import bulk, make_supercell, molecule
from ase.io import read, write

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CIF = DATA_DIR / "Mo2S4.cif"

# Transformation used to convert the hexagonal cell to an orthorhombic
# representation before building the supercell.
STRUCTURE_TRANSFORM = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])

DEFAULT_DOPANTS: List[str] = [
    "C",
    "N",
    "O",
    "F",
    "B",
    "P",
    "Se",
    "Te",
    "Cl",
    "Si",
    "Li",
    "Na",
    "Al",
    "Zn",
    "V",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Nb",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Ta",
    "W",
    "Re",
    "Ir",
    "Pt",
    "Au",
    "Ti",
]

LAT_LOOKUP: Dict[str, Tuple[str, float]] = {
    "Mo": ("bcc", 3.15),
    "Nb": ("bcc", 3.30),
    "Ta": ("bcc", 3.30),
    "W": ("bcc", 3.16),
    "V": ("bcc", 3.03),
    "Fe": ("bcc", 2.87),
    "Cu": ("fcc", 3.61),
    "Ag": ("fcc", 4.09),
    "Au": ("fcc", 4.08),
    "Ni": ("fcc", 3.52),
    "Pd": ("fcc", 3.89),
    "Pt": ("fcc", 3.92),
    "Al": ("fcc", 4.05),
    "Ir": ("fcc", 3.84),
    "Rh": ("fcc", 3.80),
    "Ti": ("hcp", 2.95),
    "Co": ("hcp", 2.51),
    "Zn": ("hcp", 2.66),
    "Cd": ("hcp", 2.98),
    "Ru": ("hcp", 2.71),
    "Re": ("hcp", 2.76),
    "C": ("diamond", 3.57),
    "Si": ("diamond", 5.43),
    "B": ("diamond", 4.75),
    "Se": ("hcp", 4.36),
    "Te": ("hcp", 4.45),
    "Na": ("bcc", 4.23),
    "Li": ("bcc", 3.49),
    "Mn": ("bcc", 8.91),
}
DIATOMIC = {"N", "O", "F", "Cl", "S"}


@dataclass
class StructureOutputs:
    """Container describing generated structure paths."""

    host: Path
    sulfur_substitution: Path
    molybdenum_substitution: Path
    intercalated: Path
    references: Dict[str, Path]


def load_host_structure(cif: Path | str | None = None) -> Atoms:
    """Load the pristine MoS₂ structure from the packaged CIF.

    Parameters
    ----------
    cif:
        Optional custom CIF path. When omitted the bundled Mo2S4.cif is
        used.
    """

    cif_path = Path(cif) if cif else DEFAULT_CIF
    return read(cif_path)


def build_supercell(
    atoms: Atoms,
    transform: np.ndarray = STRUCTURE_TRANSFORM,
    repeat: Tuple[int, int, int] = (2, 2, 1),
) -> Atoms:
    """Convert the hexagonal cell to an orthorhombic representation and
    expand it to a 50-atom supercell."""

    ortho = make_supercell(atoms, transform)
    sup = ortho.repeat(repeat)
    sup.set_pbc(True)
    return sup


def reference_structure(element: str) -> Atoms:
    """Return an ASE :class:`Atoms` object for the elemental reference."""

    symbol = element.capitalize()

    if symbol in DIATOMIC:
        box = 12.0
        if symbol == "S":
            s8_coords = np.array(
                [
                    [2.05, 0.00, 0.00],
                    [0.63, 1.93, 0.00],
                    [-1.66, 1.19, 0.00],
                    [-1.66, -1.19, 0.00],
                    [0.63, -1.93, 0.00],
                    [3.68, -1.19, 0.00],
                    [3.68, 1.19, 0.00],
                    [2.05, 2.38, 0.00],
                ]
            )
            mol = Atoms("S8", positions=s8_coords)
            mol.set_cell([box, box, box])
            mol.center()
            return mol
        mol = molecule(f"{symbol}2")
        mol.set_cell([box, box, box])
        mol.center()
        return mol

    if symbol == "P":
        box = 12.0
        a = 2.21
        coords = [[a, a, a], [-a, -a, a], [-a, a, -a], [a, -a, -a]]
        mol = Atoms("P4", positions=coords)
        mol.set_cell([box, box, box])
        mol.center()
        return mol

    if symbol in LAT_LOOKUP:
        phase, a = LAT_LOOKUP[symbol]
        if phase == "bcc":
            return bulk(symbol, "bcc", a=a, cubic=True)
        if phase == "fcc":
            return bulk(symbol, "fcc", a=a, cubic=True)
        if phase == "hcp":
            return bulk(symbol, "hcp", a=a, c=1.633 * a)
        if phase == "diamond":
            return bulk(symbol, "diamond", a=a, cubic=True)

    box = 12.0
    at = Atoms(symbol, cell=[box, box, box], pbc=False)
    at.center()
    return at


def make_derivatives(cell: Atoms, dopant: str) -> Tuple[Atoms, Atoms, Atoms]:
    """Create S-substituted, Mo-substituted, and intercalated variants."""

    sulfur_sub = cell.copy()
    sulfur_indices = [i for i, atom in enumerate(sulfur_sub) if atom.symbol == "S"]
    if sulfur_indices:
        mid_z = sulfur_sub.get_cell()[2, 2] / 2
        target = min(sulfur_indices, key=lambda idx: abs(sulfur_sub[idx].z - mid_z))
        sulfur_sub[target].symbol = dopant

    molybdenum_sub = cell.copy()
    molybdenum_indices = [i for i, atom in enumerate(molybdenum_sub) if atom.symbol == "Mo"]
    if molybdenum_indices:
        target = min(molybdenum_indices, key=lambda idx: molybdenum_sub[idx].z)
        molybdenum_sub[target].symbol = dopant

    intercalated = cell.copy()
    centre = 0.5 * (cell.get_cell()[0] + cell.get_cell()[1] + cell.get_cell()[2])
    intercalated.append(Atom(dopant, position=centre))

    return sulfur_sub, molybdenum_sub, intercalated


def write_structures(
    dopant: str, output_dir: Path, sulfur_sub: Atoms, molybdenum_sub: Atoms, intercalated: Atoms
) -> StructureOutputs:
    """Persist generated structures using the naming convention from the preprint."""

    output_dir.mkdir(parents=True, exist_ok=True)
    host = output_dir / "MoS2_50.xyz"
    outputs = StructureOutputs(
        host=host,
        sulfur_substitution=output_dir / f"MoS2_49S1{dopant}.xyz",
        molybdenum_substitution=output_dir / f"MoS2_49Mo1{dopant}.xyz",
        intercalated=output_dir / f"MoS2_50+{dopant}.xyz",
        references={},
    )

    if not host.exists():
        raise FileNotFoundError(
            "The pristine supercell must be written before writing derivatives. "
            "Call generate_structure_set() first."
        )

    write(outputs.sulfur_substitution, sulfur_sub)
    write(outputs.molybdenum_substitution, molybdenum_sub)
    write(outputs.intercalated, intercalated)
    return outputs


def generate_structure_set(
    dopants: Iterable[str] = DEFAULT_DOPANTS,
    cif: Path | str | None = None,
    output_dir: Path | str = "formation_energy_structures",
    repeat: Tuple[int, int, int] = (2, 2, 1),
) -> List[StructureOutputs]:
    """Generate pristine, defect, and reference structures.

    Returns the locations of all outputs to enable downstream processing.
    """

    output_path = Path(output_dir)
    host_atoms = build_supercell(load_host_structure(cif), repeat=repeat)
    host_file = output_path / "MoS2_50.xyz"
    output_path.mkdir(parents=True, exist_ok=True)
    write(host_file, host_atoms)

    outputs: List[StructureOutputs] = []

    for dopant in dopants:
        sulfur_sub, molybdenum_sub, intercalated = make_derivatives(host_atoms, dopant)
        outputs.append(write_structures(dopant, output_path, sulfur_sub, molybdenum_sub, intercalated))

    references: Dict[str, Path] = {}
    all_elements = set(dopants) | {"Mo", "S"}
    for element in sorted(all_elements):
        ref_atoms = reference_structure(element)
        if element in DIATOMIC:
            name = f"{element}2_box.xyz"
        elif element in LAT_LOOKUP:
            phase, _ = LAT_LOOKUP[element]
            name = f"{element}_{phase}.xyz"
        else:
            name = f"{element}_atom_box.xyz"
        ref_path = output_path / name
        write(ref_path, ref_atoms)
        references[element] = ref_path

    if outputs:
        outputs[0].references.update(references)
    return outputs
