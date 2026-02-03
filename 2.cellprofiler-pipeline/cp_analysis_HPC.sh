#!/bin/bash
#SBATCH --job-name=cpheps
#SBATCH --partition=zen5_mpi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

set -euo pipefail

# =============================================================================
# CONFIGURATION - Edit these paths for your setup
# =============================================================================
SIF="/data/brussel/vo/000/bvo00026/vsc11013/Projects/IA-Yuwei-Bodipy-Pipeline/singularity/cellprofiler.sif"
WORKDIR="/data/brussel/vo/000/bvo00026/vsc11013/Projects/IA-Yuwei-Bodipy-Pipeline/cp_analysis"

# Bind paths - add any additional paths your analysis needs
BIND_PATHS="${WORKDIR},/scratch,/data"

# Path to the cellprofiler environment inside the container
ENV_BIN="/opt/conda/envs/cellprofiler/bin"

# =============================================================================
# SETUP
# =============================================================================
# Create logs directory
mkdir -p logs

# Set environment variables for parallel processing
export MAX_WORKERS="${SLURM_CPUS_PER_TASK}"

# =============================================================================
# STEP 1: Convert notebooks to Python scripts (inside container)
# =============================================================================
echo "[$(date)] Converting notebooks to Python scripts..."

apptainer exec --cleanenv --bind "${BIND_PATHS}" "${SIF}" \
    bash -c "
        export PATH=${ENV_BIN}:\$PATH
        export MAMBA_ROOT_PREFIX=/opt/conda
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        cd '${WORKDIR}'
        mkdir -p scripts
        ${ENV_BIN}/jupyter nbconvert --to python --output-dir=scripts/ *.ipynb
    "

echo "[$(date)] Notebook conversion complete."

# =============================================================================
# STEP 2: Run CellProfiler analysis (inside container)
# =============================================================================
echo "[$(date)] Starting CellProfiler analysis..."

apptainer exec --cleanenv --bind "${BIND_PATHS}" "${SIF}" \
    bash -c "
        export PATH=${ENV_BIN}:\$PATH
        export MAMBA_ROOT_PREFIX=/opt/conda
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MAX_WORKERS=${MAX_WORKERS}
        cd '${WORKDIR}'
        ${ENV_BIN}/python scripts/cp_analysis.py
    "

echo "[$(date)] CellProfiler analysis complete."

# =============================================================================
# STEP 3: Rename SQLite files (inside container)
# =============================================================================
echo "[$(date)] Renaming SQLite files..."

apptainer exec --cleanenv --bind "${BIND_PATHS}" "${SIF}" \
    bash -c "
        export PATH=${ENV_BIN}:\$PATH
        export MAMBA_ROOT_PREFIX=/opt/conda
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        cd '${WORKDIR}'
        ${ENV_BIN}/python scripts/rename_sqlite_files.py
    "

echo "[$(date)] All steps completed successfully!"
