# mos2doped

A lightweight Python package that refactors the scripts from the MoS₂ doping
preprint into reusable modules. It provides structure generation, Quantum
ESPRESSO setup helpers, UMA/OMat24 MLIP optimisation, modular MD workflows, and
formation-energy post-processing.

Main paper- 
```
https://arxiv.org/abs/2510.05339
```

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

### Create a new conda environment

```bash
conda create -n mos2doped python=3.10
conda activate mos2doped
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

### Generate structures

Generate pristine + doped structures using the bundled CIF. By default, structures are generated for all non-radioactive elements:

```bash
mos2doped generate-structures --output formation_energy_structures
```

List all available elements from the Materials Project database:

```bash
mos2doped generate-structures --list-elements
```

Generate structures for specific dopants only:

```bash
mos2doped generate-structures --dopants Fe Cu Au Pt --output selected_dopants
```


### Optimise structures

Optimise all structures in a folder with UMA/OMat24 and write optimised XYZ
files + RDF arrays:

```bash
mos2doped ml-opt formation_energy_structures ml_optimised --save ml_energy.csv
```

Relax a single structure with Quantum ESPRESSO and log its energy to a CSV:

```bash
mos2doped qe-relax formation_energy_structures/MoS2_50.xyz /path/to/pseudos --save qe_energies.csv
```


Summarise raw QE energies into a formation-energy table:

```bash
mos2doped summarise-dft qe_energies.csv dft_formation.csv
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
  reference phases. Elemental references are loaded from Materials Project CIF 
  files (zero energy above hull structures) for all available elements.
- `mos2doped.dft` — helper utilities for Quantum ESPRESSO relaxations and
  formation-energy bookkeeping.
- `mos2doped.ml` — UMA/OMat24 optimisation and simple RDF evaluation.
- `mos2doped.md` — configurable ASE-based MD workflows.
- `mos2doped.analysis` — post-processing helpers that mirror the paper's
  formation-energy tables.

## Available Elements

The package includes Materials Project CIF files for ~80 elements with zero 
energy above hull (thermodynamically stable structures). By default, only 
non-radioactive elements are used for doping. Use `--list-elements` to see 
all available elements.
