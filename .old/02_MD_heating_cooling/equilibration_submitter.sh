#!/bin/bash
#SBATCH --job-name=equilibration_manager
#SBATCH --partition=pi.amartini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=25-00:00:00

# ────────────────────────── Configuration ────────────────────────────
SCRIPT1="script1_opt_nvt.py"
SCRIPT2="script2_npt.py"
SCRIPT3="script3_final_nvt.py"
STRUCT_DIR="formation_energy_structures"

# ───────────────────── Queue limits per partition ─────────────────
declare -A max_jobs=( ["gpu"]=4 ["cenvalarc.gpu"]=4 )
declare -A submitted_jobs=( ["gpu"]=0 ["cenvalarc.gpu"]=0 )
partitions=("gpu" "cenvalarc.gpu")

# ───────────────────── Gather structures to process ───────────────────
mapfile -t xyz_files < <(ls ${STRUCT_DIR}/*.xyz)
total=${#xyz_files[@]}

echo "☞  Will submit $total equilibration job sets (3 jobs each)."

count_jobs() { squeue -u "$USER" -p "$1" | grep -c "$USER"; }

# ───────────────────────── Job submission functions ────────────────────
submit_job() {
    local script=$1
    local base=$2
    local run_dir=$3
    local part=$4
    local job_name=$5
    local depends_on=$6

    cat > "$run_dir/submit_${job_name}.sbatch" <<EOF
#!/bin/bash
#SBATCH --job-name=${base}_${job_name}
#SBATCH --partition=$part
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH --output=${job_name}.o%j
#SBATCH --error=${job_name}.e%j
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
$([ -n "$depends_on" ] && echo "#SBATCH --dependency=afterok:$depends_on")

module purge
module load anaconda3
module load cuda
eval "\$(conda shell.bash hook)"
conda activate fair
export CUDA_VISIBLE_DEVICES=0

python $script 2>&1 | tee -a ${job_name}_output.txt
EOF

    job_id=$(cd "$run_dir" && sbatch "submit_${job_name}.sbatch" | awk '{print $4}')
    echo "$job_id"
}

# ───────────────────────── Prepare all directories first ────────────────────
echo "☞  Preparing directories and copying files..."
declare -a base_names
declare -a run_dirs

for xyz in "${xyz_files[@]}"; do
    base=$(basename "$xyz" .xyz)
    base_names+=("$base")
    
    # Create run folder and copy files
    run_dir="${STRUCT_DIR}/EQ_${base}"
    run_dirs+=("$run_dir")
    mkdir -p "$run_dir"
    cp "$xyz" "$run_dir/"
    
    # Explicitly delete existing Python scripts and replace with fresh copies
    rm -f "$run_dir/$SCRIPT1" "$run_dir/$SCRIPT2" "$run_dir/$SCRIPT3"
    cp "$SCRIPT1" "$SCRIPT2" "$SCRIPT3" "$run_dir/"
done

# ───────────────────────── Phase 1: Submit all script1 jobs ────────────────────
echo "☞  Phase 1: Submitting all optimization + NVT jobs..."
declare -a job1_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script1 output already exists
            if [ -f "$run_dir/1_${base}.xyz" ]; then
                echo "   • $base → Phase 1 already completed, skipping"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (Phase 1)"
            job1_id=$(submit_job "$SCRIPT1" "$base" "$run_dir" "$part" "opt_nvt" "")
            job1_ids+=("$job1_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

# Wait for all Phase 1 jobs to complete
echo "☞  Waiting for all Phase 1 jobs to complete..."
while true; do
    running_jobs=$(squeue -u "$USER" | grep "_opt_nvt" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_opt_nvt" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 1 completed!"
        echo "   → Waiting 60 seconds for files to be written to disk..."
        sleep 60
        break
    fi
    echo "   → Phase 1: $running_jobs running, $pending_jobs pending..."
    sleep 30
done

# ───────────────────────── Phase 2: Submit all script2 jobs ────────────────────
echo "☞  Phase 2: Submitting all NPT jobs..."
declare -a job2_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script2 output already exists or script1 output missing
            if [ -f "$run_dir/2_${base}.xyz" ]; then
                echo "   • $base → Phase 2 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/1_${base}.xyz" ]; then
                echo "   • $base → Phase 1 output missing, skipping Phase 2"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (Phase 2)"
            job2_id=$(submit_job "$SCRIPT2" "$base" "$run_dir" "$part" "npt" "")
            job2_ids+=("$job2_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

# Wait for all Phase 2 jobs to complete
echo "☞  Waiting for all Phase 2 jobs to complete..."
while true; do
    running_jobs=$(squeue -u "$USER" | grep "_npt" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_npt" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 2 completed!"
        echo "   → Waiting 60 seconds for files to be written to disk..."
        sleep 60
        break
    fi
    echo "   → Phase 2: $running_jobs running, $pending_jobs pending..."
    sleep 30
done

# ───────────────────────── Phase 3: Submit all script3 jobs ────────────────────
echo "☞  Phase 3: Submitting all final NVT jobs..."
declare -a job3_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script3 output already exists or script2 output missing
            if [ -f "$run_dir/final_${base}.xyz" ]; then
                echo "   • $base → Phase 3 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/2_${base}.xyz" ]; then
                echo "   • $base → Phase 2 output missing, skipping Phase 3"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (Phase 3)"
            job3_id=$(submit_job "$SCRIPT3" "$base" "$run_dir" "$part" "final_nvt" "")
            job3_ids+=("$job3_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

echo "✈  All job phases submitted. Waiting for Phase 3 completion..."

# ───────────────────── Wait until all Phase 3 jobs done ───────────────────
while true; do
    # Count all final NVT jobs
    running_jobs=$(squeue -u "$USER" | grep "_final_nvt" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_final_nvt" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  All equilibration phases finished."
        echo "   → Waiting 60 seconds for final files to be written to disk..."
        sleep 60
        break
    fi
    
    echo "   → Phase 3: $running_jobs running, $pending_jobs pending final NVT jobs..."
    sleep 60
done

echo "✔  All equilibration simulations completed successfully!"
echo "Results are in individual EQ_* folders within $STRUCT_DIR/"
echo "Final structures: final_*.xyz"
