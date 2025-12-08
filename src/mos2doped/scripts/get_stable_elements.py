"""
Fetch the lowest energy (zero energy above hull) structure for each element
in the periodic table from the Materials Project API.

Requirements:
    pip install mp-api

Usage:
    python get_stable_elements.py --api-key YOUR_API_KEY --output stable_elements/
    
    Or set the environment variable MP_API_KEY:
    export MP_API_KEY="your_api_key_here"
    python get_stable_elements.py --output stable_elements/
"""

import os
import argparse
from pathlib import Path

from mp_api.client import MPRester


# All elements in the periodic table (excluding noble gases and synthetic elements)
ELEMENTS = [
    "H", "Li", "Be", "B", "C", "N", "O", "F",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    # Noble gases (often no stable solid structures)
    "He", "Ne", "Ar", "Kr", "Xe", "Rn"
]


def get_stable_structures(api_key=None, output_dir="stable_elements", fmt="cif"):
    """
    Fetch the lowest energy structure (energy_above_hull = 0) for each element.
    
    Parameters
    ----------
    api_key : str, optional
        Materials Project API key. If not provided, uses MP_API_KEY env variable.
    output_dir : str
        Directory to save the structure files.
    fmt : str
        Output format: 'cif', 'poscar', or 'xyz'.
    
    Returns
    -------
    dict
        Dictionary mapping element symbol to structure info.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Use provided key or fall back to environment variable
    if api_key is None:
        api_key = os.environ.get("MP_API_KEY")
    
    if not api_key:
        raise ValueError(
            "No API key provided. Set MP_API_KEY environment variable or pass --api-key"
        )
    
    results = {}
    failed = []
    
    with MPRester(api_key) as mpr:
        for element in ELEMENTS:
            print(f"Fetching stable structure for {element}...", end=" ")
            
            try:
                # Search for materials containing only this element
                # with zero energy above hull (thermodynamically stable)
                docs = mpr.materials.summary.search(
                    chemsys=element,
                    energy_above_hull=(0, 0.001),  # essentially zero
                    fields=[
                        "material_id",
                        "formula_pretty",
                        "energy_per_atom",
                        "energy_above_hull",
                        "structure",
                        "symmetry",
                        "density"
                    ]
                )
                
                if not docs:
                    # Try with slightly higher tolerance
                    docs = mpr.materials.summary.search(
                        chemsys=element,
                        energy_above_hull=(0, 0.05),
                        fields=[
                            "material_id",
                            "formula_pretty",
                            "energy_per_atom",
                            "energy_above_hull",
                            "structure",
                            "symmetry",
                            "density"
                        ]
                    )
                
                if docs:
                    # Sort by energy above hull and pick the lowest
                    docs_sorted = sorted(docs, key=lambda x: x.energy_above_hull)
                    best = docs_sorted[0]
                    
                    # Save structure to file
                    structure = best.structure
                    if fmt == "cif":
                        filepath = output_path / f"{element}_{best.material_id}.cif"
                        structure.to(filename=str(filepath), fmt="cif")
                    elif fmt == "poscar":
                        filepath = output_path / f"{element}_{best.material_id}.vasp"
                        structure.to(filename=str(filepath), fmt="poscar")
                    elif fmt == "xyz":
                        filepath = output_path / f"{element}_{best.material_id}.xyz"
                        structure.to(filename=str(filepath), fmt="xyz")
                    
                    results[element] = {
                        "material_id": best.material_id,
                        "formula": best.formula_pretty,
                        "energy_per_atom": best.energy_per_atom,
                        "energy_above_hull": best.energy_above_hull,
                        "spacegroup": best.symmetry.symbol if best.symmetry else "N/A",
                        "density": best.density,
                        "filepath": str(filepath)
                    }
                    
                    print(f"✓ {best.material_id} ({best.formula_pretty})")
                else:
                    print("✗ No stable structure found")
                    failed.append(element)
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                failed.append(element)
    
    # Save summary to CSV
    summary_file = output_path / "summary.csv"
    with open(summary_file, "w") as f:
        f.write("element,material_id,formula,energy_per_atom,energy_above_hull,spacegroup,density,filepath\n")
        for elem, info in sorted(results.items()):
            f.write(f"{elem},{info['material_id']},{info['formula']},{info['energy_per_atom']:.6f},"
                    f"{info['energy_above_hull']:.6f},{info['spacegroup']},{info['density']:.4f},"
                    f"{info['filepath']}\n")
    
    print(f"\n{'='*60}")
    print(f"Successfully fetched: {len(results)} elements")
    print(f"Failed: {len(failed)} elements - {failed}")
    print(f"Summary saved to: {summary_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch lowest energy (zero energy above hull) structures from Materials Project"
    )
    parser.add_argument(
        "--api-key", "-k",
        help="Materials Project API key (or set MP_API_KEY env variable)"
    )
    parser.add_argument(
        "--output", "-o",
        default="stable_elements",
        help="Output directory for structure files (default: stable_elements)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["cif", "poscar", "xyz"],
        default="cif",
        help="Output file format (default: cif)"
    )
    
    args = parser.parse_args()
    
    get_stable_structures(
        api_key=args.api_key,
        output_dir=args.output,
        fmt=args.format
    )


if __name__ == "__main__":
    main()
