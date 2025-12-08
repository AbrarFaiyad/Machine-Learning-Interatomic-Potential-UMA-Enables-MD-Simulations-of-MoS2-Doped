# mos2doped

A lightweight Python package that refactors the scripts from the MoS₂ doping
preprint into reusable modules. It provides structure generation, Quantum
ESPRESSO setup helpers, UMA/OMat24 MLIP optimisation, modular MD workflows, and
formation-energy post-processing.

All original notebooks, scripts, and inputs have been preserved under `.old/`.

## Installation

It is recommended to create a new conda environment before installing the package to avoid conflicts.

### Prerequisites
1. fairchem-core
2. QuantumEspresso

### Clone the repository

```bash
git clone https://github.com/yourusername/Machine-Learning-Interatomic-Potential-UMA-Enables-MD-Simulations-of-MoS2-Doped.git
cd Machine-Learning-Interatomic-Potential-UMA-Enables-MD-Simulations-of-MoS2-Doped
```

### Install dependencies

**fairchem-core:**
```bash
pip install fairchem-core
```

**Quantum Espresso:**

Install directly from the website:
```
https://www.quantum-espresso.org/
```

### Install the package

```bash
pip install -e .
```

## Quick start

Generate pristine + doped structures using the bundled CIF:

```bash
mos2doped generate-structures --output formation_energy_structures
```

Summarise raw QE energies into a formation-energy table:

```bash
mos2doped summarise-dft .old/01_DFT_MLIP_Validation/dft_energies_raw.csv dft_formation.csv
```

Optimise all structures in a folder with UMA/OMat24 and write optimised XYZ
files + RDF arrays:

```bash
mos2doped ml-opt formation_energy_structures ml_optimised
```

Relax a single structure with Quantum ESPRESSO and log its energy to a CSV:

```bash
mos2doped qe-relax formation_energy_structures/MoS2_50.xyz /path/to/pseudos --save qe_energies.csv
```

Run a custom MD schedule described in JSON (or YAML if ``pyyaml`` is installed):

```bash
cat > workflow.json <<'EOF'
[
  {"type": "optimise", "fmax": 0.1},
  {"type": "nvt", "temperature": 300, "steps": 5000},
  {"type": "npt", "temperature": 700, "pressure": 1.0, "steps": 5000}
]
EOF

mos2doped md-run formation_energy_structures/MoS2_50.xyz workflow.json md_outputs --save md_summary.csv
```

Compute a radial distribution function for any structure:

```bash
mos2doped rdf formation_energy_structures/MoS2_50.xyz --output rdf.csv
```

## Module overview

- `mos2doped.structures` — build 50-atom MoS₂ supercells, doped variants, and
  reference phases.
- `mos2doped.dft` — helper utilities for Quantum ESPRESSO relaxations and
  formation-energy bookkeeping.
- `mos2doped.ml` — UMA/OMat24 optimisation and simple RDF evaluation.
- `mos2doped.md` — configurable ASE-based MD workflows.
- `mos2doped.analysis` — post-processing helpers that mirror the paper's
  formation-energy tables.

Refer to `.old/` for the unmodified scripts and notebooks from the preprint.
