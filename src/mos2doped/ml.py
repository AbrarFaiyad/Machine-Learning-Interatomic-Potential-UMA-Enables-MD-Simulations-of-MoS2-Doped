"""MLIP utilities for UMA/OMat24 inference and RDF analysis."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import BFGS

if TYPE_CHECKING:  # pragma: no cover - import guarded for optional dependency
    from fairchem.core import FAIRChemCalculator


@dataclass
class MLOutput:
    structure: str
    n_atoms: int
    energy: float
    path: Path
    converged: bool = True
    error_message: str = ""


def load_calculator(device: str | None = None):
    """Load the UMA calculator lazily to keep imports optional."""

    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    import torch

    chosen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=chosen_device)
    return FAIRChemCalculator(predictor, task_name="omat")


def optimise_structure(path: Path | str, calculator=None, fmax: float = 0.005) -> Tuple[MLOutput, object]:
    """Relax atomic positions with the supplied calculator and return energies.
    
    If optimization fails, returns MLOutput with converged=False and error_message set.
    """

    path_obj = Path(path)
    
    try:
        atoms = read(path_obj)
        atoms.info.update({"spin": 1, "charge": 0})
        atoms.calc = calculator or load_calculator()

        opt = BFGS(atoms, trajectory=None)
        opt.run(fmax=fmax, steps=10000)
        energy = atoms.get_potential_energy()
        output = MLOutput(
            structure=path_obj.stem,
            n_atoms=len(atoms),
            energy=float(energy),
            path=path_obj,
            converged=True,
            error_message=""
        )
        return output, atoms
    
    except Exception as error:
        # Return failed output with error details for debugging
        error_msg = f"{type(error).__name__}: {str(error)}"
        print(f"⚠️  Failed to optimize {path_obj.name}: {error_msg}")
        
        output = MLOutput(
            structure=path_obj.stem,
            n_atoms=0,
            energy=0.0,
            path=path_obj,
            converged=False,
            error_message=error_msg
        )
        return output, None


def radial_distribution(atoms, r_max: float = 10.0, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a simple radial distribution function ignoring species labels."""

    positions = atoms.get_positions()
    cell = atoms.cell
    dr = r_max / n_bins
    r_bins = np.linspace(0, r_max, n_bins + 1)
    r_centres = 0.5 * (r_bins[:-1] + r_bins[1:])
    distances: List[float] = []
    for i, pos_i in enumerate(positions):
        deltas = positions[i + 1 :] - pos_i
        deltas -= np.round(deltas / cell.lengths()) * cell.lengths()
        norms = np.linalg.norm(deltas, axis=1)
        distances.extend([d for d in norms if d <= r_max])
    hist, _ = np.histogram(distances, bins=r_bins)
    shell_volumes = (4 / 3) * np.pi * (r_bins[1:] ** 3 - r_bins[:-1] ** 3)
    density = len(positions) / atoms.get_volume()
    rdf = hist / (density * shell_volumes * len(positions))
    return rdf, r_centres


def optimise_directory(
    structure_dir: Path | str,
    output_dir: Path | str,
    calculator=None,
    fmax: float = 0.005,
) -> Tuple[List[MLOutput], Dict[str, np.ndarray]]:
    """Optimise every ``.xyz`` file and return energy + RDF summaries.
    
    Skips structures that fail to optimize and tracks them in the output list.
    """

    structure_dir = Path(structure_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    calculator = calculator or load_calculator()

    energies: List[MLOutput] = []
    rdfs: Dict[str, np.ndarray] = {}
    failed_structures = []

    for path in sorted(structure_dir.glob("*.xyz")):
        result, atoms = optimise_structure(path, calculator=calculator, fmax=fmax)
        energies.append(result)
        
        if result.converged and atoms is not None:
            dest = output_dir / f"{result.structure}_ml_opt.xyz"
            write(dest, atoms)
            rdf, r_centres = radial_distribution(atoms)
            rdfs[result.structure] = np.vstack([r_centres, rdf])
            print(f"✓ {result.structure}: {result.energy:.6f} eV")
        else:
            failed_structures.append((result.structure, result.error_message))
            print(f"✗ {result.structure}: {result.error_message}")
    
    if failed_structures:
        print(f"\n⚠️  {len(failed_structures)} structure(s) failed to optimize:")
        for name, error in failed_structures:
            print(f"  - {name}: {error}")
    
    return energies, rdfs
