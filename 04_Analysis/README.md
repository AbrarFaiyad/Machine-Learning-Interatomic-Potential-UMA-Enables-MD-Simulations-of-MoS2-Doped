## Analysis Scripts for MoS₂ Doping Study

This folder contains Python scripts and a Jupyter notebook for analyzing the results of DFT, ML, and MD simulations of doped MoS₂. These tools help you compare machine learning interatomic potentials (MLIPs) to DFT, assess structural and dynamical properties from molecular dynamics, and understand dopant clustering behavior.

### Workflow Overview

1. **Structure Generation & DFT/ML Calculations (`01_DFT_MLIP_Validation/`)**
	- `step_1_structure_maker.py`: Generates pristine and doped MoS₂ supercells and reference structures for each dopant.
	- `step_2_run_qe_single.py`: Runs Quantum ESPRESSO (QE) DFT relaxations for each structure and collects total energies.
	- `step_3_summarize_dft.py`: Summarizes DFT energies into formation energies for each dopant/defect type.
	- `step_4_ML_energy.py`: Optimizes the same structures using a MLIP (UMA/OMat24), computes ML formation energies, and performs RDF analysis.

2. **MD Simulations (`02_MD_heating_cooling/`)**
	- Scripts here run heating, equilibration, and cooling MD for each doped system, producing trajectory and output files for further analysis.

3. **Analysis (`04_Analysis/`)**
	- The scripts in this folder analyze and visualize the results from the above steps.

---

### Script Descriptions

#### 1. `compare_ml_dft.py`

**Purpose:**  
Compares formation energies predicted by MLIP and DFT for various dopants and defect types in MoS₂.

**Key Features:**
- Reads summarized DFT and ML formation energy CSVs.
- Generates parity plots, error distributions, and stacked bar plots to visualize ML vs DFT agreement.
- Calculates error metrics (MAE, R², MAPE, Pearson correlation).
- Optionally produces publication-quality figures and PDF reports.

**Inputs:**  
- `dft_formation_energies.csv` (from `step_3_summarize_dft.py`)
- `ml_formation_energies_s.csv` (from `step_4_ML_energy.py`)

**Outputs:**  
- PNG/PDF figures comparing ML and DFT results.
- Error analysis by dopant and defect type.

---

#### 2. `density_analysis.py`

**Purpose:**  
Analyzes the change in density before and after MD equilibration for each doped system.

**Key Features:**
- Reads `lowtemp_output.txt` files from each dopant's MD output directory.
- Extracts and averages the final density values.
- Plots a bar chart comparing initial and final densities for all dopants.

**Inputs:**  
- MD output files (typically from `02_MD_heating_cooling/` or a similar directory).

**Outputs:**  
- `dopant_density_comparison.png`: Bar plot of density changes for all dopants.

---

#### 3. `msd_analysis.py`

**Purpose:**  
Calculates and visualizes the mean squared displacement (MSD) for each doped system, quantifying atomic mobility and diffusion during MD.

**Key Features:**
- Processes trajectory files for all dopants.
- Computes MSD as a function of time for each system.
- Generates individual MSD plots for each dopant.

**Inputs:**  
- Trajectory files from MD runs (e.g., `equilibriated_structures`).

**Outputs:**  
- MSD plots for each dopant, saved in a results directory.

---

#### 4. `Nearest_neighbour_Analysis.ipynb`

**Purpose:**  
Performs bond and clustering analysis to determine how dopant atoms interact and cluster during MD.

**Key Features:**
- Loads MD trajectories and identifies atomic bonds using neighbor lists.
- Tracks which bonds persist or break over time.
- Quantifies dopant clustering and stability.

**Inputs:**  
- Trajectory files from MD runs.

**Outputs:**  
- Visualizations and statistics on bond persistence and dopant clustering.

---

### How the Analysis Fits the Workflow

- **DFT and ML formation energies** are compared to validate the accuracy of the MLIP for different dopants and defect types.
- **MD simulations** provide trajectories and output files for each doped system, which are then analyzed for structural (density), dynamical (MSD), and clustering (neighbor analysis) properties.
- The **analysis scripts** in this folder are designed to be run after the DFT/ML calculations and MD simulations are complete.

---

### Usage

1. Ensure all required dependencies are installed (see script headers for requirements).
2. Run the scripts in this folder after generating the necessary input files from the DFT/ML and MD workflows.
3. Review the generated plots and tables to interpret the effects of different dopants on MoS₂ properties.

---

For more details on each script's options and configuration, see the comments and docstrings within each file.