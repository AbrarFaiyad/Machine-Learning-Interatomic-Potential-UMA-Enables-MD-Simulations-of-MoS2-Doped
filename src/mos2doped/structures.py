"""Structure generation utilities for doped MoS₂.

This module consolidates the logic from ``step_1_structure_maker.py``
into reusable functions. All paths are decoupled from the working
folder so the functions can be called from Python code or the CLI.

Elemental reference structures are loaded from CIF files in the data folder,
which contain Materials Project structures with zero energy above hull.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import glob

import numpy as np
from ase import Atom, Atoms
from ase.build import make_supercell
from ase.io import read, write

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_CIF = DATA_DIR / "Mo2S4.cif"

# Transformation used to convert the hexagonal cell to an orthorhombic
# representation before building the supercell.
STRUCTURE_TRANSFORM = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])

# Build a lookup of available elemental CIF files from the data folder
# Pattern: {Element}_mp-{id}.cif (e.g., Fe_mp-13.cif, Au_mp-81.cif)
ELEMENT_CIF_LOOKUP: Dict[str, Path] = {}

for cif_file in DATA_DIR.glob("*_mp-*.cif"):
    # Extract element symbol from filename (e.g., "Fe" from "Fe_mp-13.cif")
    element = cif_file.stem.split("_mp-")[0]
    # Skip multi-element compounds like Mo2S4
    if element.isalpha() and element[0].isupper():
        ELEMENT_CIF_LOOKUP[element] = cif_file

# List of available elements for reference
AVAILABLE_ELEMENTS = sorted(ELEMENT_CIF_LOOKUP.keys())

# Radioactive elements to exclude from default dopants
RADIOACTIVE_ELEMENTS = {
    "Tc", "Pm", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", 
    "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", 
    "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
    "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
}

# Non-radioactive elements available in the data folder
NON_RADIOACTIVE_ELEMENTS = [
    el for el in AVAILABLE_ELEMENTS if el not in RADIOACTIVE_ELEMENTS
]

# Default dopants: all non-radioactive elements except Mo and S (the host elements)
DEFAULT_DOPANTS: List[str] = [
    el for el in NON_RADIOACTIVE_ELEMENTS if el not in {"Mo", "S"}
]


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
    """Return an ASE Atoms object for the elemental reference.
    
    Loads the structure from the Materials Project CIF files in the data folder.
    If the structure has only 1 atom, it is replicated 2x1x1 to ensure
    proper periodic boundary conditions.
    
    Parameters
    ----------
    element : str
        Element symbol (e.g., 'Fe', 'Au', 'C')
    
    Returns
    -------
    Atoms
        ASE Atoms object for the elemental reference structure
    
    Raises
    ------
    ValueError
        If the element is not found in the data folder
    """
    symbol = element.capitalize()
    
    # Special case for two-letter elements like 'Se', 'Te', etc.
    if len(element) == 2:
        symbol = element[0].upper() + element[1].lower()
    
    if symbol not in ELEMENT_CIF_LOOKUP:
        available = ", ".join(sorted(ELEMENT_CIF_LOOKUP.keys())[:10]) + "..."
        raise ValueError(
            f"Element '{symbol}' not found in data folder. "
            f"Available elements: {available}"
        )
    
    cif_path = ELEMENT_CIF_LOOKUP[symbol]
    atoms = read(cif_path)
    
    # If structure has only 1 atom, replicate 2x1x1
    if len(atoms) == 1:
        atoms = atoms.repeat((2, 1, 1))
    
    atoms.set_pbc(True)
    return atoms


def get_available_elements() -> List[str]:
    """Return a list of all available elements in the data folder."""
    return AVAILABLE_ELEMENTS.copy()


def get_element_cif_path(element: str) -> Optional[Path]:
    """Get the CIF file path for a given element.
    
    Parameters
    ----------
    element : str
        Element symbol
    
    Returns
    -------
    Path or None
        Path to the CIF file, or None if not found
    """
    symbol = element.capitalize()
    if len(element) == 2:
        symbol = element[0].upper() + element[1].lower()
    return ELEMENT_CIF_LOOKUP.get(symbol)


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
    
    Reference structures are loaded from Materials Project CIF files in the
    data folder. If a structure has only 1 atom, it is replicated 2x1x1.
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

    # Generate reference structures from CIF files
    references: Dict[str, Path] = {}
    all_elements = set(dopants) | {"Mo", "S"}
    
    for element in sorted(all_elements):
        try:
            ref_atoms = reference_structure(element)
            # Use material project ID in filename
            cif_path = get_element_cif_path(element)
            if cif_path:
                mp_id = cif_path.stem.split("_")[1]  # e.g., "mp-13" from "Fe_mp-13"
                name = f"{element}_{mp_id}.xyz"
            else:
                name = f"{element}_ref.xyz"
            
            ref_path = output_path / name
            write(ref_path, ref_atoms)
            references[element] = ref_path
        except ValueError as e:
            print(f"Warning: Could not generate reference for {element}: {e}")

    if outputs:
        outputs[0].references.update(references)
    return outputs
