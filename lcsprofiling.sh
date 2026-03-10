#!/bin/bash
#SBATCH --job-name=lcs_scale
#SBATCH --output=lcs_scale_%j.out
#SBATCH --error=lcs_scale_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=06:00:00

cd $SLURM_SUBMIT_DIR

echo "Compiling..."
nvcc -O3 lcsprofiling.cu -o lcsprofiling

# Input sizes
sizes=(10000 100000 1000000 10000000 100000000)

for N in "${sizes[@]}"
do
    echo "===================================="
    echo "Profiling N = $N"
    echo "===================================="

    nsys profile \
        --stats=true \
        --trace=cuda \
        --output=lcs_profile_N${N} \
        ./lcsprofiling $N

    echo "Done N=$N"
done

echo "All scalability runs completed."
ls -lh lcs_profile_N*