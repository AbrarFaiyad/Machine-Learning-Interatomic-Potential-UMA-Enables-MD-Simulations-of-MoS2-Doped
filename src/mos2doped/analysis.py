"""Post-processing helpers for formation energies and RDF tables."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .dft import formation_energy
from .structures import DEFAULT_DOPANTS


def load_energy_table(path: Path | str) -> pd.DataFrame:
    """Load a CSV with ``structure``, ``n_atoms`` and ``energy`` columns."""

    df = pd.read_csv(path)
    expected = {"structure", "n_atoms", "energy"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}")
    return df


def build_formation_table(
    energy_table: pd.DataFrame,
    dopants: Iterable[str] = DEFAULT_DOPANTS,
    host_key: str = "MoS2_50",
) -> pd.DataFrame:
    """Compute formation energies for every dopant found in the input table."""

    energies: Dict[str, float] = dict(zip(energy_table.structure, energy_table.energy))
    n_atoms: Dict[str, int] = dict(zip(energy_table.structure, energy_table.n_atoms))

    rows: List[Dict[str, float]] = []
    for dopant in dopants:
        try:
            terms = formation_energy(energies, n_atoms, dopant, host_key=host_key)
            row = {
                "dopant": dopant,
                "E_host (eV)": energies[host_key],
                "E_Ssub (eV)": energies[f"MoS2_49S1{dopant}"],
                "E_Mosub (eV)": energies[f"MoS2_49Mo1{dopant}"],
                "E_int (eV)": energies[f"MoS2_50+{dopant}"],
                "mu_X (eV/atom)": energies.get(f"{dopant}_atom_box", float("nan")),
            }
            row.update(terms)
            rows.append(row)
        except KeyError:
            continue
    return pd.DataFrame(rows)


def summarise_dft(raw_csv: Path | str, output_csv: Path | str) -> pd.DataFrame:
    """Load raw QE energies and emit the concise formation energy file."""

    df = load_energy_table(raw_csv)
    formation = build_formation_table(df)
    formation.to_csv(output_csv, index=False)
    return formation
