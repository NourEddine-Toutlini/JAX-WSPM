#!/bin/bash
#SBATCH --job-name=richards_benchmark
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=1




module load cuda/12.6
module load python/3.12.4

mkdir -p logs results

echo "test_case,mesh_size,solver,preconditioner,simulation_time,iterations,l2_pressure,linf_pressure,l2_saturation,linf_saturation,l2_relative_pressure,l2_relative_saturation,memory_usage" > results/benchmark_results_cpu_200.csv

for test_case in Test1 Test2 Test3; do
   for mesh in 200; do
       for solver in bicgstab gmres; do
           for precond in none; do
               echo "Running: $test_case - mesh:$mesh - solver:$solver - precond:$precond"
               
               python -m src.cli \
                   --test-case $test_case \
                   --mesh-size $mesh \
                   --solver $solver \
                   --preconditioner $precond \
                   --output-dir results \
                   --tmax 0.25 \
                   --dt 1e-4
               
               TEST_CASE=$test_case MESH=$mesh SOLVER=$solver PRECOND=$precond python -c '



import os

file_name = f"results/numerical_results_{os.environ["TEST_CASE"]}.npz" #_{os.environ["SOLVER"]}_{os.environ["PRECOND"]}_mesh{os.environ["MESH"]


import numpy as np
data = np.load(file_name)
print(f"{os.environ["TEST_CASE"]},{os.environ["MESH"]},{os.environ["SOLVER"]},{os.environ["PRECOND"]},"
     f"{data["simulation_time"]:.4f},"
     f"{sum(data["iterations"])},"
     f"{float(data["l2_pressure"]):.4e},"
     f"{float(data["linf_pressure"]):.4e},"
     f"{float(data["l2_saturation"]):.4e},"
     f"{float(data["linf_saturation"]):.4e},"
     f"{float(data["l2_relative_pressure"]):.4e},"
     f"{float(data["l2_relative_saturation"]):.4e},"
     f"{float(data["memory_usage"]):.2f}")' >> results/benchmark_results_cpu_200.csv
           done
       done
   done
done