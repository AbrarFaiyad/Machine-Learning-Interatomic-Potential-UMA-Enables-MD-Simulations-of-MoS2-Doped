# MD Heating & Cooling Workflow for Doped MoS₂

This repository provides a modular workflow for molecular dynamics (MD) simulations of doped MoS₂, including structure optimization, equilibration, heating, high-temperature holds, cooling, and final low-temperature equilibration. The workflow is designed for use with MLIP (e.g., UMA/OMat24) and is optimized for GPU acceleration.

---

## Folder Structure

- **Python Scripts (MD Steps):**
  - `script1_opt_nvt.py` — Structure optimization + initial NVT equilibration
  - `script2_npt.py` — NPT equilibration with density convergence
  - `script3_final_nvt.py` — Final NVT equilibration at 300K
  - `script4_npt_heat.py` — NPT heating from 300K to 1000K
  - `script5_npt_hightemp_hold.py` — NPT equilibration at 1000K
  - `script6_nvt_high_temp_hold.py` — NVT equilibration at 1000K (long hold)
  - `script7_npt_cool.py` — NPT cooling from 1000K to 300K
  - `script8_npt_lowtemp_hold.py` — NPT equilibration at 300K
  - `script9_nvt_lowtemp_hold.py` — NVT equilibration at 300K (long hold)

- **Job Submission Scripts:**
  - `equilibration_submitter.sh` — Automates the first three MD steps (optimization, NPT, final NVT) for all structures.
  - `heating_submitter.sh` — Automates the full heating/cooling cycle (scripts 4–9) for all structures.

---

## Workflow Overview

### 1. Structure Optimization & Initial Equilibration

- **`script1_opt_nvt.py`**  
  Optimizes the input structure and performs initial NVT equilibration at 300K with convergence checks.  
  **Input:** `*.xyz`  
  **Output:** `1_{filename}.xyz`

- **`script2_npt.py`**  
  Performs NPT equilibration at 300K, monitoring density convergence.  
  **Input:** `1_{filename}.xyz`  
  **Output:** `2_{filename}.xyz`

- **`script3_final_nvt.py`**  
  Final NVT equilibration at 300K with convergence checks.  
  **Input:** `2_{filename}.xyz`  
  **Output:** `final_{filename}.xyz`

### 2. Heating, High-Temperature Hold, Cooling, and Final Equilibration

- **`script4_npt_heat.py`**  
  NPT heating from 300K to 1000K over a user-defined time (default: 20 ps).  
  **Input:** `final_{filename}.xyz`  
  **Output:** `4_{filename}.xyz`

- **`script5_npt_hightemp_hold.py`**  
  NPT equilibration at 1000K (default: 10 ps).  
  **Input:** `4_{filename}.xyz`  
  **Output:** `5_{filename}.xyz`

- **`script6_nvt_high_temp_hold.py`**  
  NVT equilibration at 1000K for a long hold (default: 100 ps).  
  **Input:** `5_{filename}.xyz`  
  **Output:** `6_{filename}.xyz`

- **`script7_npt_cool.py`**  
  NPT cooling from 1000K to 300K (default: 20 ps).  
  **Input:** `6_{filename}.xyz`  
  **Output:** `7_{filename}.xyz`

- **`script8_npt_lowtemp_hold.py`**  
  NPT equilibration at 300K (default: 20 ps).  
  **Input:** `7_{filename}.xyz`  
  **Output:** `8_{filename}.xyz`

- **`script9_nvt_lowtemp_hold.py`**  
  NVT equilibration at 300K for a long hold (default: 100 ps).  
  **Input:** `8_{filename}.xyz`  
  **Output:** `9_{filename}.xyz`

---

## Job Submission & Automation

- **`equilibration_submitter.sh`**  
  Submits the first three MD steps (scripts 1–3) for all structures in `formation_energy_structures/`.  
  - Handles job dependencies and GPU resource allocation.
  - Results are stored in per-structure subfolders.

- **`heating_submitter.sh`**  
  Submits the full heating/cooling cycle (scripts 4–9) for all structures in `equilibrated_structures/`.  
  - Handles job dependencies and GPU resource allocation.
  - Results are stored in per-structure subfolders.

---

## How to Use

### 1. Prepare Input Structures

- Place your initial `.xyz` files in the appropriate directory (e.g., `formation_energy_structures/`).

### 2. Run Equilibration

- Submit the equilibration workflow:
  ```bash
  sbatch equilibration_submitter.sh
  ```
- This will run scripts 1–3 for all input structures.

### 3. Run Heating/Cooling Cycle

- After equilibration, move the final structures to `equilibrated_structures/` (or as required).
- Submit the heating/cooling workflow:
  ```bash
  sbatch heating_submitter.sh
  ```
- This will run scripts 4–9 for all equilibrated structures.

---

## Script Details

- All scripts are optimized for GPU usage (CUDA). They will fall back to CPU if no GPU is available, but performance will be much slower.
- Each script expects its input file to be the only matching file in the working directory.
- Output files are named with a prefix indicating the step (e.g., `1_`, `2_`, ..., `final_`, `4_`, etc.).
- Trajectory and log files are produced for monitoring simulation progress and convergence.

---

## Requirements

- Python 3.x
- ASE (`pip install ase`)
- fairchem (for MLIP/UMA/OMat24)
- numpy, torch
- SLURM-based HPC cluster with GPU nodes (for job submission scripts)

---

## Notes & Customization

- **Simulation Parameters:**  
  You can adjust temperatures, time steps, and simulation lengths by editing the relevant variables in each script.
- **Dopant/Structure List:**  
  The workflow is agnostic to the specific dopants or structures; simply provide the desired `.xyz` files.
- **Resource Allocation:**  
  Edit the job submission scripts to match your cluster's partition names, GPU requirements, and job limits.

---

## Example & Reference

*Link to paper and example results will be added here.*

---

## Post-processing & Analysis

Post-processing scripts for further analysis and visualization are available in the `../03_Analysis` folder (to be uploaded separately).

---

For questions, issues, or suggestions, please open an issue or contact the maintainer.
