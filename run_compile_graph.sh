#!/bin/bash
#SBATCH --job-name=compile_graph
#SBATCH --output=logs/compile_full_graph_%A.out
#SBATCH --error=logs/compile_full_graph_%A.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=email@email.com

source venv/bin/activate

python compile_graph.py