"""Command line interface for the MoS₂ doping toolkit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from ase.io import read

from . import __version__
from .analysis import summarise_dft
from .dft import build_default_inputs, run_relaxation
from .md import load_calculator as load_md_calculator
from .md import sequential_workflow
from .ml import load_calculator as load_ml_calculator
from .ml import optimise_directory, radial_distribution
from .structures import (
    DEFAULT_DOPANTS,
    NON_RADIOACTIVE_ELEMENTS,
    AVAILABLE_ELEMENTS,
    RADIOACTIVE_ELEMENTS,
    generate_structure_set,
    get_available_elements,
    reference_structure,
)


def write_energy_rows(rows: Iterable[dict], output: Path, append: bool = False) -> Path:
    """Persist structure/n_atoms/energy rows in CSV form."""

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=["structure", "n_atoms", "energy"])
    mode = "a" if append and output.exists() else "w"
    header = not (append and output.exists())
    frame.to_csv(output, index=False, mode=mode, header=header)
    return output


def load_workflow_file(path: Path) -> Sequence[dict]:
    """Load a JSON or YAML workflow description."""

    text = Path(path).read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        return yaml.safe_load(text)
    return json.loads(text)


def cmd_generate(args: argparse.Namespace) -> None:
    if args.list_elements:
        print("Available elements (from Materials Project CIF files):")
        print(f"  Total: {len(AVAILABLE_ELEMENTS)}")
        print(f"  Non-radioactive: {len(NON_RADIOACTIVE_ELEMENTS)}")
        print(f"\nNon-radioactive elements:")
        print(f"  {', '.join(NON_RADIOACTIVE_ELEMENTS)}")
        print(f"\nRadioactive elements in data folder:")
        radioactive_available = [el for el in AVAILABLE_ELEMENTS if el in RADIOACTIVE_ELEMENTS]
        print(f"  {', '.join(radioactive_available) if radioactive_available else 'None'}")
        return
    
    repeat = tuple(args.repeat)
    dopants = args.dopants
    
    # Handle special dopant options
    if args.all_elements:
        dopants = [el for el in AVAILABLE_ELEMENTS if el not in {"Mo", "S"}]
        print(f"Using all {len(dopants)} available elements as dopants")
    elif args.include_radioactive:
        dopants = [el for el in AVAILABLE_ELEMENTS if el not in {"Mo", "S"}]
        print(f"Using all {len(dopants)} elements (including radioactive) as dopants")
    
    generate_structure_set(dopants=dopants, cif=args.cif, output_dir=args.output, repeat=repeat)
    print(f"Structures written to {Path(args.output).resolve()}")
    print(f"Generated structures for {len(dopants)} dopant elements")


def cmd_qe_relax(args: argparse.Namespace) -> None:
    inputs = build_default_inputs(total_charge=args.charge)
    result = run_relaxation(
        args.xyz,
        args.pseudo_dir,
        inputs=inputs,
        profile_command=args.command,
    )
    csv_path = write_energy_rows(
        [{"structure": result.structure, "n_atoms": result.n_atoms, "energy": result.energy}],
        args.save,
        append=args.append,
    )
    print(f"Relaxed {result.structure}: {result.energy:.6f} eV (written to {csv_path.resolve()})")


def cmd_summarise(args: argparse.Namespace) -> None:
    out = summarise_dft(args.raw, args.output)
    print(out)
    print(f"Formation energies written to {Path(args.output).resolve()}")


def cmd_ml(args: argparse.Namespace) -> None:
    calculator = load_ml_calculator(device=args.device)
    energies, rdfs = optimise_directory(args.input, args.output, calculator=calculator, fmax=args.fmax)

    csv_path = write_energy_rows(
        [
            {"structure": res.structure, "n_atoms": res.n_atoms, "energy": res.energy}
            for res in energies
        ],
        args.save,
        append=args.append,
    )

    if args.rdf_dir:
        rdf_dir = Path(args.rdf_dir)
        rdf_dir.mkdir(parents=True, exist_ok=True)
        for name, arr in rdfs.items():
            np.savetxt(rdf_dir / f"{name}_rdf.csv", arr.T, delimiter=",", header="r,g(r)", comments="")

    print(
        "Optimised {count} structures. Energies appended to {csv}. Optimised XYZ files live in {out}.".format(
            count=len(energies), csv=csv_path.resolve(), out=Path(args.output).resolve()
        )
    )


def cmd_md(args: argparse.Namespace) -> None:
    workflow = load_workflow_file(args.workflow)
    calculator = load_md_calculator(device=args.device)
    results = sequential_workflow(args.input, args.output, workflow, calculator=calculator)

    summary_rows = []
    for step in results:
        summary_rows.append(
            {
                "step": step.name,
                "output": str(step.output),
                "avg_temperature": float(np.mean(step.temperatures)) if step.temperatures else np.nan,
                "avg_energy": float(np.mean(step.energies)) if step.energies else np.nan,
                "n_records": len(step.temperatures),
            }
        )

    summary_path = Path(args.save)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(
        "Ran {count} MD stages. Trajectories are under {out}. Summary saved to {summary}.".format(
            count=len(results), out=Path(args.output).resolve(), summary=summary_path.resolve()
        )
    )


def cmd_rdf(args: argparse.Namespace) -> None:
    atoms = read(args.xyz)
    rdf, centres = radial_distribution(atoms, r_max=args.r_max, n_bins=args.bins)
    arr = np.vstack([centres, rdf]).T
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output, arr, delimiter=",", header="r,g(r)", comments="")
    print(f"RDF written to {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workflow utilities for doped MoS2")
    parser.add_argument("--version", action="version", version=f"mos2doped {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-structures", help="Create pristine + doped structures")
    gen.add_argument("--cif", type=Path, help="Optional CIF to override the bundled Mo2S4.cif")
    gen.add_argument("--output", type=Path, default=Path("formation_energy_structures"))
    gen.add_argument("--dopants", nargs="*", default=DEFAULT_DOPANTS,
                     help="List of dopant elements (default: all non-radioactive elements)")
    gen.add_argument("--repeat", nargs=3, type=int, default=(2, 2, 1), help="Supercell repeat (abc)")
    gen.add_argument("--all-elements", action="store_true",
                     help="Use all available elements as dopants (excluding Mo and S)")
    gen.add_argument("--include-radioactive", action="store_true",
                     help="Include radioactive elements in dopant list")
    gen.add_argument("--list-elements", action="store_true",
                     help="List all available elements and exit")
    gen.set_defaults(func=cmd_generate)

    qe = subparsers.add_parser("qe-relax", help="Run a QE vc-relax calculation for one XYZ")
    qe.add_argument("xyz", type=Path, help="Structure to relax")
    qe.add_argument("pseudo_dir", type=Path, help="Directory containing QE pseudopotentials")
    qe.add_argument("--charge", type=float, default=0.0, help="Total system charge")
    qe.add_argument("--command", type=str, help="Custom MPI launch command for pw.x")
    qe.add_argument("--save", type=Path, default=Path("qe_energies.csv"), help="CSV file for energies")
    qe.add_argument("--append", action="store_true", help="Append to the existing CSV instead of overwriting")
    qe.set_defaults(func=cmd_qe_relax)

    summ = subparsers.add_parser("summarise-dft", help="Create formation energy table from QE outputs")
    summ.add_argument("raw", type=Path, help="CSV with structure,n_atoms,energy columns")
    summ.add_argument("output", type=Path, help="Destination CSV for formation energies")
    summ.set_defaults(func=cmd_summarise)

    ml = subparsers.add_parser("ml-opt", help="Optimise structures with UMA/OMat24")
    ml.add_argument("input", type=Path, help="Directory containing .xyz files")
    ml.add_argument("output", type=Path, help="Where optimised structures should be stored")
    ml.add_argument("--fmax", type=float, default=0.005)
    ml.add_argument("--save", type=Path, default=Path("ml_energies.csv"), help="CSV file for ML energies")
    ml.add_argument("--append", action="store_true", help="Append to the existing CSV instead of overwriting")
    ml.add_argument("--rdf-dir", type=Path, help="Optional folder to store RDF curves for each structure")
    ml.add_argument("--device", type=str, help="Force a device for UMA (cpu or cuda)")
    ml.set_defaults(func=cmd_ml)

    md = subparsers.add_parser("md-run", help="Execute an MD workflow described in JSON or YAML")
    md.add_argument("input", type=Path, help="Input structure (XYZ, CIF, etc.)")
    md.add_argument("workflow", type=Path, help="Path to JSON/YAML describing MD stages")
    md.add_argument("output", type=Path, help="Directory for outputs")
    md.add_argument("--save", type=Path, default=Path("md_summary.csv"), help="CSV file summarising MD blocks")
    md.add_argument("--device", type=str, help="Force a device for UMA (cpu or cuda)")
    md.set_defaults(func=cmd_md)

    rdf = subparsers.add_parser("rdf", help="Compute the RDF for a single structure")
    rdf.add_argument("xyz", type=Path, help="Input structure")
    rdf.add_argument("--r-max", dest="r_max", type=float, default=10.0, help="Maximum radius (Å)")
    rdf.add_argument("--bins", type=int, default=100, help="Number of histogram bins")
    rdf.add_argument("--output", type=Path, default=Path("rdf.csv"), help="CSV destination for r,g(r)")
    rdf.set_defaults(func=cmd_rdf)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
