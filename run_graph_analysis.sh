#!/bin/bash
#SBATCH --job-name=graph_analysis
#SBATCH --output=logs/graph_analysis_%A.out
#SBATCH --error=logs/graph_analysis_%A.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=email@email.com

source venv/bin/activate
python graph_analysis.py