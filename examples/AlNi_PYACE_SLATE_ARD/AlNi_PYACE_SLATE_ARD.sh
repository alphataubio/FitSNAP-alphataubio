#!/bin/bash
#SBATCH --job-name=AlNi_PYACE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=192
#SBATCH --mem=0
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=alphataubio@gmail.com

module load StdEnv/2023 gcc/13.3 aocl-lapack/5.1 aocl-blas/5.1 openmpi/5.0.3 python/3.12.4 mpi4py/4.0.0 scipy-stack/2025a 

source ~/fitsnap-venv/bin/activate

export PYTHONPATH=~/scratch/FitSNAP-alphataubio:~/scratch/lammps-alphataubio/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/.local/lib64:~/scratch/lammps-alphataubio/build-fitsnap:$LD_LIBRARY_PATH

srun python -m fitsnap3 AlNi_PYACE_SLATE_ARD.in --overwrite

