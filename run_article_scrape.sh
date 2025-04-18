#!/bin/bash
#SBATCH --job-name=scrape_articles
#SBATCH --output=logs/article_scrape_%A.out
#SBATCH --error=logs/article_scrape_%A.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=email@email.com

mkdir -p logs
source venv/bin/activate

python article_scrape.py \
  --strict-book-filter