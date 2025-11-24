#!/bin/bash
#SBATCH --job-name=heating_manager
#SBATCH --partition=pi.amartini
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=25-00:00:00

# ────────────────────────── Configuration ────────────────────────────
SCRIPT4="script4_npt_heat.py"
SCRIPT5="script5_npt_hightemp_hold.py"
SCRIPT6="script6_nvt_high_temp_hold.py"
SCRIPT7="script7_npt_cool.py"
SCRIPT8="script8_npt_lowtemp_hold.py"
SCRIPT9="script9_nvt_lowtemp_hold.py"
STRUCT_DIR="equilibrated_structures"

# ───────────────────── Queue limits per partition ─────────────────
declare -A max_jobs=( ["cenvalarc.gpu"]=1 ) #["gpu"]=4
declare -A submitted_jobs=( ["cenvalarc.gpu"]=0 ) #["gpu"]=0 
partitions=( "cenvalarc.gpu")

# ───────────────────── Gather structures to process ───────────────────
mapfile -t xyz_files < <(ls ${STRUCT_DIR}/*.xyz)
total=${#xyz_files[@]}

echo "☞  Will submit $total heating job sets (6 jobs each)."

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
    # Remove "final_" prefix if present
    if [[ $base == final_* ]]; then
        base=${base#final_}
    fi
    base_names+=("$base")
    
    # Create run folder and copy files
    run_dir="${STRUCT_DIR}/HEAT_${base}"
    run_dirs+=("$run_dir")
    mkdir -p "$run_dir"
    cp "$xyz" "$run_dir/"
    
    # Explicitly delete existing Python scripts and replace with fresh copies
    rm -f "$run_dir/$SCRIPT4" "$run_dir/$SCRIPT5" "$run_dir/$SCRIPT6" "$run_dir/$SCRIPT7" "$run_dir/$SCRIPT8" "$run_dir/$SCRIPT9"
    cp "$SCRIPT4" "$SCRIPT5" "$SCRIPT6" "$SCRIPT7" "$SCRIPT8" "$SCRIPT9" "$run_dir/"
done

# ───────────────────────── Phase 1: Submit all script4 jobs ────────────────────
echo "☞  Phase 1: Submitting all heating jobs..."
declare -a job4_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script4 output already exists
            if [ -f "$run_dir/4_${base}.xyz" ]; then
                echo "   • $base → Phase 1 already completed, skipping"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (Heating)"
            job4_id=$(submit_job "$SCRIPT4" "$base" "$run_dir" "$part" "heat" "")
            job4_ids+=("$job4_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

# Wait for all Phase 1 jobs to complete
echo "☞  Waiting for all heating jobs to complete..."
while true; do
    running_jobs=$(squeue -u "$USER" | grep "_heat" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_heat" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 1 (heating) completed!"
        break
    fi
    echo "   → Phase 1: $running_jobs running, $pending_jobs pending..."
    sleep 30
done

# ───────────────────────── Phase 2: Submit all script5 jobs ────────────────────
echo "☞  Phase 2: Submitting all high-temperature equilibration jobs..."
declare -a job5_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script5 output already exists or script4 output missing
            if [ -f "$run_dir/5_${base}.xyz" ]; then
                echo "   • $base → Phase 2 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/4_${base}.xyz" ]; then
                echo "   • $base → Phase 1 output missing, skipping Phase 2"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (High-temp hold)"
            job5_id=$(submit_job "$SCRIPT5" "$base" "$run_dir" "$part" "hightemp" "")
            job5_ids+=("$job5_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

echo "✈  All heating phases submitted. Waiting for completion..."

# ───────────────────── Wait until all Phase 2 jobs done ───────────────────
while true; do
    # Count all high-temp equilibration jobs
    running_jobs=$(squeue -u "$USER" | grep "_hightemp" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_hightemp" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 2 (high-temp equilibration) completed!"
        break
    fi
    
    echo "   → Phase 2: $running_jobs running, $pending_jobs pending high-temp jobs..."
    sleep 60
done

# ───────────────────────── Phase 3: Submit all script6 jobs ────────────────────
echo "☞  Phase 3: Submitting all NVT high-temperature equilibration jobs..."
declare -a job6_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script6 output already exists or script5 output missing
            if [ -f "$run_dir/6_${base}.xyz" ]; then
                echo "   • $base → Phase 3 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/5_${base}.xyz" ]; then
                echo "   • $base → Phase 2 output missing, skipping Phase 3"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (NVT high-temp hold)"
            job6_id=$(submit_job "$SCRIPT6" "$base" "$run_dir" "$part" "nvthightemp" "")
            job6_ids+=("$job6_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

echo "✈  All heating phases submitted. Waiting for final completion..."

# ───────────────────── Wait until all Phase 3 jobs done ───────────────────
while true; do
    # Count all NVT high-temp equilibration jobs
    running_jobs=$(squeue -u "$USER" | grep "_nvthightemp" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_nvthightemp" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 3 (NVT high-temp equilibration) completed!"
        break
    fi
    
    echo "   → Phase 3: $running_jobs running, $pending_jobs pending NVT high-temp jobs..."
    sleep 60
done

# ───────────────────────── Phase 4: Submit all script7 jobs ────────────────────
echo "☞  Phase 4: Submitting all cooling jobs..."
declare -a job7_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script7 output already exists or script6 output missing
            if [ -f "$run_dir/7_${base}.xyz" ]; then
                echo "   • $base → Phase 4 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/6_${base}.xyz" ]; then
                echo "   • $base → Phase 3 output missing, skipping Phase 4"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (Cooling)"
            job7_id=$(submit_job "$SCRIPT7" "$base" "$run_dir" "$part" "cool" "")
            job7_ids+=("$job7_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

# Wait for all Phase 4 jobs to complete
echo "☞  Waiting for all cooling jobs to complete..."
while true; do
    running_jobs=$(squeue -u "$USER" | grep "_cool" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_cool" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 4 (cooling) completed!"
        break
    fi
    echo "   → Phase 4: $running_jobs running, $pending_jobs pending..."
    sleep 30
done

# ───────────────────────── Phase 5: Submit all script8 jobs ────────────────────
echo "☞  Phase 5: Submitting all low-temperature equilibration jobs..."
declare -a job8_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script8 output already exists or script7 output missing
            if [ -f "$run_dir/8_${base}.xyz" ]; then
                echo "   • $base → Phase 5 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/7_${base}.xyz" ]; then
                echo "   • $base → Phase 4 output missing, skipping Phase 5"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (Low-temp hold)"
            job8_id=$(submit_job "$SCRIPT8" "$base" "$run_dir" "$part" "lowtemp" "")
            job8_ids+=("$job8_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

# Wait for all Phase 5 jobs to complete
echo "☞  Waiting for all low-temperature equilibration jobs to complete..."
while true; do
    running_jobs=$(squeue -u "$USER" | grep "_lowtemp" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_lowtemp" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  Phase 5 (low-temp equilibration) completed!"
        break
    fi
    echo "   → Phase 5: $running_jobs running, $pending_jobs pending..."
    sleep 30
done

# ───────────────────────── Phase 6: Submit all script9 jobs ────────────────────
echo "☞  Phase 6: Submitting all NVT low-temperature equilibration jobs..."
declare -a job9_ids

idx=0
while (( idx < total )); do
    for part in "${partitions[@]}"; do
        submitted_jobs[$part]=$(count_jobs $part)
    done

    for part in "${partitions[@]}"; do
        while (( submitted_jobs[$part] < max_jobs[$part] && idx < total )); do
            base="${base_names[$idx]}"
            run_dir="${run_dirs[$idx]}"
            
            # Check if script9 output already exists or script8 output missing
            if [ -f "$run_dir/9_${base}.xyz" ]; then
                echo "   • $base → Phase 6 already completed, skipping"
                ((idx++))
                continue
            elif [ ! -f "$run_dir/8_${base}.xyz" ]; then
                echo "   • $base → Phase 5 output missing, skipping Phase 6"
                ((idx++))
                continue
            fi
            
            echo "   • $base → partition: $part (NVT low-temp hold)"
            job9_id=$(submit_job "$SCRIPT9" "$base" "$run_dir" "$part" "nvtlowtemp" "")
            job9_ids+=("$job9_id")
            
            ((idx++))
            ((submitted_jobs[$part]++))
        done
    done
    sleep 10
done

echo "✈  All simulation phases submitted. Waiting for final completion..."

# ───────────────────── Wait until all Phase 6 jobs done ───────────────────
while true; do
    # Count all NVT low-temp equilibration jobs
    running_jobs=$(squeue -u "$USER" | grep "_nvtlowtemp" | grep -c " R ")
    pending_jobs=$(squeue -u "$USER" | grep "_nvtlowtemp" | grep -c " PD ")
    
    if (( running_jobs == 0 && pending_jobs == 0 )); then
        echo "✔  All simulation phases finished."
        break
    fi
    
    echo "   → Phase 6: $running_jobs running, $pending_jobs pending NVT low-temp jobs..."
    sleep 60
done

echo "✔  All heating and cooling simulations completed successfully!"
echo "Results are in individual HEAT_* folders within $STRUCT_DIR/"
echo "Heated structures: 4_*.xyz"
echo "High-temp equilibrated structures: 5_*.xyz"
echo "NVT high-temp equilibrated structures: 6_*.xyz"
echo "Cooled structures: 7_*.xyz"
echo "Low-temp equilibrated structures: 8_*.xyz"
echo "Final NVT low-temp equilibrated structures: 9_*.xyz"
