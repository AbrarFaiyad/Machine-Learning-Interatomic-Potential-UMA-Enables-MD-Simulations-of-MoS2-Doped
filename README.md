
# Machine-Learning-Interatomic-Potential-UMA-Enables-MD-Simulations-of-MoS2-Doped

This repository provides a complete, modular workflow for generating, simulating, and analyzing doped MoS₂ systems using both Density Functional Theory (DFT) and Machine Learning Interatomic Potentials (MLIPs, e.g., UMA/OMat24). The workflow covers structure generation, DFT and MLIP validation, molecular dynamics (MD) simulations, and in-depth analysis of structural, energetic, and dynamical properties.

---

## Folder Structure & Purpose

### `00_Structure/`
- **Mo2S4.cif**: Crystallographic information file for pristine MoS₂. Used as the starting point for all structure generation.

### `01_DFT_MLIP_Validation/`
Scripts and tools for generating structures, running DFT calculations, summarizing results, and benchmarking MLIP predictions against DFT.
	- **step_1_structure_maker.py**: Generates pristine and doped MoS₂ supercells and elemental reference structures.
	- **step_2_run_qe_single.py**: Runs a single Quantum ESPRESSO (QE) DFT relaxation for a given structure and logs the total energy.
	- **step_3_summarize_dft.py**: Summarizes DFT energies into formation energies for each dopant/defect type.
	- **step_4_ML_energy.py**: Optimizes the same structures using a MLIP, computes ML formation energies, and performs RDF analysis.
	- **DFT_Job_submitter.sh**: Automates the DFT workflow and job submission on HPC clusters.
	- **README.md**: Details the DFT/MLIP validation workflow.

### `02_MD_heating_cooling/`
Scripts for running and automating MD simulations of doped MoS₂, including heating, cooling, and equilibration cycles.
	- **script1_opt_nvt.py**: Structure optimization and initial NVT equilibration.
	- **script2_npt.py**: NPT equilibration with density convergence.
	- **script3_final_nvt.py**: Final NVT equilibration at 300K.
	- **script4_npt_heat.py**: NPT heating from 300K to 1000K.
	- **script5_npt_hightemp_hold.py**: NPT equilibration at 1000K.
	- **script6_nvt_high_temp_hold.py**: NVT equilibration at 1000K (long hold).
	- **script7_npt_cool.py**: NPT cooling from 1000K to 300K.
	- **script8_npt_lowtemp_hold.py**: NPT equilibration at 300K.
	- **script9_nvt_lowtemp_hold.py**: NVT equilibration at 300K (long hold).
	- **equilibration_submitter.sh**: Automates the first three MD steps for all structures.
	- **heating_submitter.sh**: Automates the full heating/cooling cycle for all structures.
	- **README.md**: Details the MD workflow and script usage.

### `04_Analysis/`
Scripts and a notebook for analyzing and visualizing the results of DFT, ML, and MD simulations.
	- **compare_ml_dft.py**: Compares MLIP and DFT formation energies, generates error metrics and visualizations.
	- **density_analysis.py**: Analyzes density changes before and after MD equilibration for each doped system.
	- **msd_analysis.py**: Computes and plots mean squared displacement (MSD) for each doped system from MD trajectories.
	- **Nearest_neighbour_Analysis.ipynb**: Jupyter notebook for bond and clustering analysis, tracking dopant clustering and bond persistence during MD.
	- **README.md**: Describes the analysis scripts and their role in the workflow.

---

## Workflow Summary

1. **Structure Generation**
		- Start with `Mo2S4.cif` in `00_Structure/`.
		- Use `step_1_structure_maker.py` to generate all pristine, doped, and reference structures.

2. **DFT & MLIP Validation**
		- Run DFT relaxations for each structure using `step_2_run_qe_single.py` (automated via `DFT_Job_submitter.sh`).
		- Summarize DFT results with `step_3_summarize_dft.py`.
		- Run MLIP relaxations and energy calculations with `step_4_ML_energy.py`.
		- Compare MLIP and DFT results using `compare_ml_dft.py` in `04_Analysis/`.

3. **Molecular Dynamics (MD) Simulations**
		- Use scripts in `02_MD_heating_cooling/` to run full heating, equilibration, and cooling cycles for each doped system.
		- Automate batch runs with `equilibration_submitter.sh` and `heating_submitter.sh`.

4. **Analysis**
		- Use `density_analysis.py` to assess density changes after MD.
		- Use `msd_analysis.py` to analyze atomic mobility and diffusion.
		- Use `Nearest_neighbour_Analysis.ipynb` for bond and clustering analysis.
		- All analysis scripts are in `04_Analysis/` and produce figures and tables for publication or further study.

---

## Getting Started


### Path Conventions & Running the Workflow

All scripts now use paths relative to the repository root. You do not need to edit any paths after cloning the repository. The folder structure is:

```
00_Structure/                # Pristine MoS2 CIF
01_DFT_MLIP_Validation/      # Structure generation, DFT, MLIP, and results
02_MD_heating_cooling/       # MD simulation scripts and outputs
04_Analysis/                 # Analysis scripts and notebooks
```

**Key conventions:**
- All scripts expect to be run from anywhere, but will always look for inputs/outputs in the correct subfolders relative to the repo root.
- DFT/MLIP scripts read/write to `01_DFT_MLIP_Validation/formation_energy_structures/` and `01_DFT_MLIP_Validation/` for CSVs.
- MD scripts read from `01_DFT_MLIP_Validation/formation_energy_structures/` and write to `02_MD_heating_cooling/`.
- Analysis scripts read from `01_DFT_MLIP_Validation/`, `02_MD_heating_cooling/`, and write to `04_Analysis/`.

### How to Run the Workflow

1. **Clone the repository:**
	```bash
	git clone https://github.com/AbrarFaiyad/Machine-Learning-Interatomic-Potential-UMA-Enables-MD-Simulations-of-MoS2-Doped.git
	cd Machine-Learning-Interatomic-Potential-UMA-Enables-MD-Simulations-of-MoS2-Doped
	```
2. **Install dependencies:**
	- See the top of each script for required Python packages (e.g., `ase`, `fairchem`, `numpy`, `pandas`, `matplotlib`, `seaborn`, etc.).
	- You can use `pip install -r requirements.txt` if a requirements file is provided, or install manually.
3. **Run the workflow in order:**
	- Generate structures: `python 01_DFT_MLIP_Validation/step_1_structure_maker.py`
	- Run DFT and MLIP calculations: `python 01_DFT_MLIP_Validation/step_2_run_qe_single.py`, `step_3_summarize_dft.py`, `step_4_ML_energy.py`
	- Run MD simulations: Use scripts in `02_MD_heating_cooling/` (see that folder's README for details)
	- Run analysis: Use scripts in `04_Analysis/` (see that folder's README for details)

**You should not need to change any paths.** All outputs will be placed in the correct subfolders automatically.

---

## Citation
If you use this workflow or scripts in your research, please cite the relevant papers for UMA/OMat24 and this repository.