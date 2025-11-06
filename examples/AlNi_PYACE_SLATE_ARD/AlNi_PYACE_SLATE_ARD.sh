#!/bin/bash
#SBATCH --job-name=AlNi_PYACE
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=192
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=all
#SBATCH --mail-user=alphataubio@gmail.com
#SBATCH --output=AlNi_PYACE_%j.out
#SBATCH --error=AlNi_PYACE_%j.err

module load StdEnv/2023 intel/2024.2.0 openblas/0.3.27 openmpi/5.0.3 python/3.11.5 hdf5-mpi/1.14.5 mpi4py/4.0.0 scipy-stack/2025a

source ~/fitsnap-venv/bin/activate
export PYTHONPATH=~/scratch/FitSNAP-alphataubio:~/scratch/lammps-alphataubio/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/.local/lib64:~/scratch/lammps-alphataubio/build-fitsnap:$LD_LIBRARY_PATH

srun python -m fitsnap3 AlNi_PYACE_SLATE_ARD.in

