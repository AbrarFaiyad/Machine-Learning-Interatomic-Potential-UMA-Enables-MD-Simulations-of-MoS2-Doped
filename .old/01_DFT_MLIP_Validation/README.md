# DFT & MLIP Validation Workflow for Doped MoS₂

This repository provides a complete workflow for generating, running, and analyzing DFT and MLIP calculations for doped MoS₂ systems. The workflow is modular, allowing you to generate structures, run DFT calculations, summarize results, and benchmark MLIP predictions against DFT.

---

## Folder Structure

- `step_1_structure_maker.py` — Generate pristine and doped MoS₂ structures, plus elemental references.
- `step_2_run_qe_single.py` — Run a single DFT (Quantum ESPRESSO) calculation for a given structure.
- `step_3_summarize_dft.py` — Summarize all DFT results and compute formation energies.
- `step_4_ML_energy.py` — Run the full workflow using MLIPs (e.g., UMA/OMat24) and compute ML-predicted formation energies.
- `DFT_Job_submitter.sh` — Central job manager: automates steps 1–3 and manages job submission on HPC clusters.

---

## 1. Structure Generation (`step_1_structure_maker.py`)

**Purpose:**  
Creates all required structures for DFT/MLIP calculations:
- Pristine MoS₂ supercell (`MoS2_50.xyz`)
- Three defect variants for each dopant:
  - S-substitution: `MoS2_49S1X.xyz`
  - Mo-substitution: `MoS2_49Mo1X.xyz`
  - Intercalation: `MoS2_50+X.xyz`
- Reference structures for each element (e.g., `Mo_bcc.xyz`, `Au_fcc.xyz`, `N2_box.xyz`)

**How to use:**
```bash
python step_1_structure_maker.py
```
- All output files are placed in `./formation_energy_structures/`.

---

## 2. Single DFT Calculation (`step_2_run_qe_single.py`)

**Purpose:**  
Runs a single Quantum ESPRESSO (QE) calculation (`vc-relax`) for a given `.xyz` structure.

**How to use:**
- Place the `.xyz` file in your working directory.
- Run:
  ```bash
  python step_2_run_qe_single.py
  ```
- The script:
  - Detects the `.xyz` file.
  - Sets up and runs QE with appropriate pseudopotentials and k-points.
  - Appends the total energy to `../dft_energies_raw.csv`.
  - Archives output files (`espresso.pwo`, etc.) with structure-specific names.

**Note:**  
This script is designed to be run inside a job folder (see job manager below).

---

## 3. Summarize DFT Results (`step_3_summarize_dft.py`)

**Purpose:**  
After all DFT jobs are complete, this script reads `dft_energies_raw.csv` and computes formation energies for all dopants and defect types.

**How to use:**
```bash
python step_3_summarize_dft.py
```
- Produces `dft_formation_energies.csv` with all relevant energies and formation energies.

---

## 4. MLIP Workflow (`step_4_ML_energy.py`)

**Purpose:**  
Runs the entire workflow using a Machine-Learned Interatomic Potential (MLIP, e.g., UMA/OMat24):
- Optimizes all structures in `formation_energy_structures/`
- Computes raw and formation energies
- Optionally performs RDF analysis for comparison with DFT

**How to use:**
```bash
python step_4_ML_energy.py
```
- Requires: `fairchem`, `ase`, `pandas`, `numpy`
- Produces ML-optimized structures and a CSV of ML-predicted formation energies.

**⚡ GPU RECOMMENDED:**
- This script is optimized for GPU usage. For best performance, run on a GPU node (HPC) or ensure your local Python environment has GPU access (CUDA-enabled). Running on CPU is possible but will be ~10x slower.

---

## 5. Job Submission & Automation (`DFT_Job_submitter.sh`)

**Purpose:**  
Automates the entire DFT workflow on an HPC cluster:
- Submits DFT jobs for all structures in `formation_energy_structures/`
- Manages job concurrency per partition
- Waits for all jobs to finish
- Runs the summarization script at the end

**How to use:**
1. Edit the script to set correct paths and partition names for your cluster.
2. Submit the script to your cluster's scheduler (e.g., `sbatch DFT_Job_submitter.sh`).

---

## Typical Workflow

1. **Generate structures:**
   ```bash
   python step_1_structure_maker.py
   ```
2. **Submit all DFT jobs:**
   ```bash
   sbatch DFT_Job_submitter.sh
   ```
3. **(Optional) Run MLIP workflow:**
   ```bash
   python step_4_ML_energy.py
   ```

---

## Requirements

- Python 3.x
- ASE (`pip install ase`)
- pandas, numpy
- fairchem (for MLIP workflow)
- Quantum ESPRESSO (for DFT)
- Open Babel (for charge/spin detection in DFT runs)
- Access to an HPC cluster with SLURM (for job manager)

---

## Example & Reference

*Link to paper and example results will be added here.*

---

## Post-processing & Analysis

Post-processing scripts for further analysis and visualization are available in the `../03_Analysis` folder (to be uploaded separately).

---

## Notes & Customization

- **Dopant List:**  
  The list of dopants is defined in each script and can be customized as needed.
- **Pseudopotentials:**  
  Make sure the `PSEUDO_DIR` in `step_2_run_qe_single.py` points to your QE pseudopotential library.
- **Cluster Settings:**  
  Edit `DFT_Job_submitter.sh` to match your cluster's partition names and job limits.

---

## Questions?

For questions, issues, or suggestions, please open an issue or contact the maintainer.
