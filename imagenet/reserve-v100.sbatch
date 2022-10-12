#!/bin/bash
# Allowable settings for gpu:n range from 1 to 4. If your application is entirely GPU driven,
# then you do not need to explicilty request cores as one CPU core will be assigned by
# default to act as a master to launch the GPU based calculation. If however your application
# is mixed CPU-GPU then you will need to request the number of cores with --ntasks as is
# required by your job.
# Place your SLURM batch settings here.
# --------------------------------------------------
#SBATCH -J lowgpu-session
#SBATCH -o /homes/zdeng/work/slurm/job_name-%x.job_number-%j.nodes-%N.out
#SBATCH -e /homes/zdeng/work/slurm/job_name-%x.job_number-%j.nodes-%N.err
#SBATCH -p 1xV100GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
######### SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
######### FOO SBATCH --nodelist highgpu-pg0-1
#SBATCH --exclusive
#
# Resource settings
# --------------------------------------------------
# --------------------------------------------------
# Place your job/script settings here.
# --------------------------------------------------
# Job Commands go here
# --------------------------------------------------
conda activate /homes/zdeng/.conda/envs/mltls
python3 ltilde.py &> ltilde.log
exit 0

