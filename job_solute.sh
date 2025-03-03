#!/bin/bash
#SBATCH --job-name=Solute_benchmark
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=10G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1



module load cuda/12.6
module load python/3.12.4

python -m src.cli --test-case SoluteTest --solver direct  --preconditioner none --output-dir Soluteresults --mesh-size 25  --dt 1e-4  --tmax 1
