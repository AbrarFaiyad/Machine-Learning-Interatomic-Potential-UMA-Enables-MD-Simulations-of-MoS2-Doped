#!/bin/bash
#SBATCH --job-name=mos2_job_manager
#SBATCH --partition=pi.amartini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1      # ← launcher itself
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=15-00:00:00

# ────────────────────────── user paths ────────────────────────────
STRUCTURE_MAKER="step_1.py"
RUN_SCRIPT="step_2_run_qe_single.py"
SUM_SCRIPT="step_4_summarise_dft.py"
MASTER_CSV="dft_energies_raw.csv"
STRUCT_DIR="formation_energy_structures"

# ───────────────────── final post-processing ───────────────────────
#echo "Creating structures"
#python "$STRUCTURE_MAKER"
#echo "Done."

# ───────────────────── queue limits per partition ─────────────────
declare -A max_jobs=( ["pi.amartini"]=8 ["long"]=3 ["bigmem"]=2 ["cenvalarc.bigmem"]=2 ) # Replace with your node allocations sum of partitions will be the maximum number of jobs you run simultaneously.
declare -A submitted_jobs=( ["pi.amartini"]=0 ["long"]=0 ["bigmem"]=0 ["cenvalarc.bigmem"]=0 )
partitions=("pi.amartini" "long" "bigmem" "cenvalarc.bigmem") # Nodes you want to use for the current jobs 

# ───────────────────── gather structures to run ───────────────────
mapfile -t xyz_files < <(ls ${STRUCT_DIR}/*.xyz)
total=${#xyz_files[@]}
idx=0

echo "☞  Will submit $total QE jobs."

count_jobs () { squeue -u "$USER" -p "$1" | grep -c "$USER"; }

# ───────────────────────── submission loop ────────────────────────
while (( idx < total )); do
  for part in "${partitions[@]}"; do
      submitted_jobs[$part]=$(count_jobs $part)
  done

  for part in "${partitions[@]}"; do
    while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
        xyz="${xyz_files[$idx]}"
        base=$(basename "$xyz" .xyz)

        # ── NEW: decide core count from #atoms ─────────────────────
        n_atoms=$(head -n 1 "$xyz")
        if (( n_atoms < 10 )); then
            NPROC=12           # small reference → 12 ranks
        else
            NPROC=56           # default (big defect cells)
        fi
        echo "   • $base  →  $NPROC MPI ranks"

        # create run folder and copy files
        run_dir="${STRUCT_DIR}/QE_${base}"
        mkdir -p "$run_dir"
        cp "$xyz"         "$run_dir/"
        cp "$RUN_SCRIPT"   "$run_dir/"

# ------------- generate the per-structure Slurm script -------------
cat > "$run_dir/submit_${base}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=${base}_dft
#SBATCH --partition=$part
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=$NPROC
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=job.o%j
#SBATCH --error=job.e%j

module purge
module load openmpi/4.1.1-gcc-8.4.1
module load anaconda3
module load quantum-espresso
eval "\$(conda shell.bash hook)"
conda activate fair
export OMP_NUM_THREADS=1

python $RUN_SCRIPT
EOF
        ( cd "$run_dir" && sbatch "submit_${base}.sbatch" )
        ((idx++))
        ((submitted_jobs[$part]++))
    done
  done
  sleep 10
done

echo "✈  All jobs submitted. Waiting for completion ..."

# ───────────────────── wait until all jobs done ───────────────────
while true; do
    # Count only the DFT jobs we submitted (exclude the manager job)
    running_dft_jobs=$(squeue -u "$USER" | grep "_dft" | grep -c " R ")
    
    if (( running_dft_jobs == 0 )); then
        echo "✔  All QE jobs finished."
        break
    fi
    
    echo "   → $running_dft_jobs DFT jobs still running..."
    sleep 60
done

# ───────────────────── final post-processing ───────────────────────
echo "Running final summariser ..."
cp "$SUM_SCRIPT"   "$STRUCT_DIR/"
cd "$STRUCT_DIR"
python "$SUM_SCRIPT"
echo "Done."
