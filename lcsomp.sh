#!/bin/bash
#SBATCH --job-name=lcs_sa_omp
#SBATCH --output=lcs_sa_omp_%j.out
#SBATCH --error=lcs_sa_omp_%j.err

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --partition=debug

module purge
module load gcc

export OMP_NUM_THREADS=32
export OMP_PROC_BIND=close
export OMP_PLACES=cores

ulimit -s unlimited

gcc -O3 -march=native -fopenmp lcsomp.c -o lcsomp

./lcsomp