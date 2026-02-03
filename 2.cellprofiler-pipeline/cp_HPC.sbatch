#!/bin/bash
#SBATCH --job-name=cphep
#SBATCH --partition=zen5_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out


module purge
module load Mamba
source $EBROOTMAMBA/etc/profile.d/conda.sh
conda activate cytomining_a

cd /data/brussel/vo/000/bvo00026/vsc11013/Projects/IA-Yuwei-Bodipy-Pipeline/cp_analysis

mkdir -p scripts

jupyter nbconvert --to python --output-dir=scripts/ *.ipynb

python scripts/cp_analysis.py 

python scripts/rename_sqlite_files.py

conda deactivate

export MAX_WORKERS=$SLURM_CPUS_PER_TASK

