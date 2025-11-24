"""Reusable MD workflows built from the original stepwise scripts."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Sequence
from ase import units
from ase.io import read, write
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.optimize import BFGS

if TYPE_CHECKING:  # pragma: no cover - import guarded for optional dependency
    from fairchem.core import FAIRChemCalculator


@dataclass
class MDStepResult:
    name: str
    output: Path
    temperatures: List[float]
    energies: List[float]


def load_calculator(device: str | None = None) -> FAIRChemCalculator:
    """Return an UMA calculator; shared between MD stages."""

    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    import torch

    device_choice = device or ("cuda" if torch.cuda.is_available() else "cpu")
    predictor = pretrained_mlip.get_predict_unit("uma-s-1p1", device=device_choice)
    return FAIRChemCalculator(predictor, task_name="omat")


def optimise(atoms, fmax: float = 0.1) -> None:
    """Run a basic geometry optimisation."""

    opt = BFGS(atoms, trajectory=None)
    opt.run(fmax=fmax)


def run_nvt(
    atoms,
    temperature: float,
    steps: int,
    timestep_fs: float = 1.0,
    tau_fs: float = 100.0,
    log_interval: int = 10,
) -> tuple[list[float], list[float]]:
    """Run an NVT Berendsen thermostat MD block."""

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    dyn = NVTBerendsen(atoms, timestep_fs * units.fs, temperature, taut=tau_fs * units.fs)
    temps: List[float] = []
    energies: List[float] = []

    def log():
        temps.append(atoms.get_temperature())
        energies.append(atoms.get_kinetic_energy() + atoms.get_potential_energy())

    dyn.attach(log, interval=log_interval)
    dyn.run(steps)
    return temps, energies


def run_npt(
    atoms,
    temperature: float,
    pressure: float,
    steps: int,
    timestep_fs: float = 1.0,
    tau_fs: float = 100.0,
    compressibility: float = 4.5e-5,
    log_interval: int = 10,
) -> tuple[list[float], list[float]]:
    """Run an NPT Berendsen block (isotropic barostat)."""

    MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)
    Stationary(atoms)
    dyn = NPTBerendsen(
        atoms,
        timestep_fs * units.fs,
        temperature,
        pressure=pressure * units.bar,
        taut=tau_fs * units.fs,
        pfactor=compressibility,
    )
    temps: List[float] = []
    energies: List[float] = []

    def log():
        temps.append(atoms.get_temperature())
        energies.append(atoms.get_kinetic_energy() + atoms.get_potential_energy())

    dyn.attach(log, interval=log_interval)
    dyn.run(steps)
    return temps, energies


def sequential_workflow(
    input_structure: Path | str,
    output_dir: Path | str,
    workflow: Sequence[dict],
    calculator=None,
) -> List[MDStepResult]:
    """Execute a user-defined series of MD stages.

    ``workflow`` is a list of dictionaries describing the stage type and
    parameters, e.g. ``{"type": "nvt", "temperature": 300, "steps": 10000}``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = read(input_structure)
    atoms.calc = calculator or load_calculator()

    results: List[MDStepResult] = []
    current_atoms = atoms
    for idx, stage in enumerate(workflow, 1):
        stage_type = stage["type"].lower()
        label = stage.get("name", f"step{idx}")
        temps: List[float]
        energies: List[float]

        if stage_type == "optimise":
            optimise(current_atoms, fmax=stage.get("fmax", 0.1))
            temps, energies = [], []
        elif stage_type == "nvt":
            temps, energies = run_nvt(
                current_atoms,
                temperature=stage.get("temperature", 300.0),
                steps=stage.get("steps", 10000),
                timestep_fs=stage.get("timestep_fs", 1.0),
                tau_fs=stage.get("tau_fs", 100.0),
                log_interval=stage.get("log_interval", 10),
            )
        elif stage_type == "npt":
            temps, energies = run_npt(
                current_atoms,
                temperature=stage.get("temperature", 300.0),
                pressure=stage.get("pressure", 1.0),
                steps=stage.get("steps", 10000),
                timestep_fs=stage.get("timestep_fs", 1.0),
                tau_fs=stage.get("tau_fs", 100.0),
                compressibility=stage.get("compressibility", 4.5e-5),
                log_interval=stage.get("log_interval", 10),
            )
        else:
            raise ValueError(f"Unsupported stage type: {stage_type}")

        output_path = output_dir / f"{idx}_{Path(input_structure).stem}.xyz"
        write(output_path, current_atoms)
        results.append(MDStepResult(name=label, output=output_path, temperatures=temps, energies=energies))
    return results
