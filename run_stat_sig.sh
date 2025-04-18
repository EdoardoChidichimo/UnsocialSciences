#!/bin/bash
#SBATCH --job-name=graph_stat_sig
#SBATCH --output=logs/graph_stat_sig_%A.out
#SBATCH --error=logs/graph_stat_sig_%A.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=email@email.com

source venv/bin/activate
python stat_sig.py