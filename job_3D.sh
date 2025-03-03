#!/bin/bash
#SBATCH --job-name=richards3D_benchmark
#SBATCH --output=logs3D/%j.out
#SBATCH --error=logs3D/%j.err
#SBATCH --time=1-00:00:00
#SBATCH --mem=40G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1


# Print CPU usage message
echo "=== Running on GPU ==="
echo "CPU cores allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: 80G"
echo "=========================="
echo ""

module load cuda/12.6
module load python/3.12.4
mkdir -p logs3D 
python -m src.cli2 --test-case Test3D --solver bicgstab  --preconditioner none --output-dir results3D --dt 1e-4  --tmax 0.1

