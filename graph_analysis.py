import igraph as ig
import pandas as pd
import os
import numpy as np
import statistics
from collections import defaultdict, Counter

from config import FIELD_NAME, MIN_YEAR, MAX_YEAR, ROLLING_WINDOW, MAX_TEAM_SIZE

RESULTS_DIR = f"results/{FIELD_NAME}"
YEARLY_SNA_CSV = f"{RESULTS_DIR}/yearly_sna_metrics.csv"
SNA_OUT = f"{RESULTS_DIR}/sna_results.csv"
GRAPH_FULL_PATH = f"graphs/full/{FIELD_NAME}_coauthorship.graphml"
GRAPH_YEARLY_DIR = "graphs/yearly"

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_graph(path):
    g = ig.Graph.Read_GraphML(path)
    print(f"Loaded graph {path}: {g.vcount()} nodes, {g.ecount()} edges")
    return g

def num_collaborative_papers():
    total_papers = 0
    collab_papers = 0
    coauthor_counts = []
    year_stats = defaultdict(lambda: {"total": 0, "collab": 0, "counts": []})
    team_size_counter = Counter()
    journal_summaries = []

    journal_files = [f for f in os.listdir(f"data/{FIELD_NAME}") if f.endswith('.csv') and f != f"{FIELD_NAME}_author_index.csv"]

    for journal_file in journal_files:
        path = f"data/{FIELD_NAME}/{journal_file}"

        try:
            df = pd.read_csv(path, header=0)
            df = df.dropna(subset=["year", "authors"])
            df["year"] = df["year"].astype(int)

            journal_total = 0
            journal_collab = 0
            journal_coauthor_counts = []

            for _, row in df.iterrows():
                year = row["year"]
                if not (MIN_YEAR <= year <= MAX_YEAR):
                    continue

                authors = row["authors"]
                if isinstance(authors, str):
                    count = authors.count(";") + 1
                    if count > MAX_TEAM_SIZE:
                        continue
                    is_collab = ";" in authors

                    year_stats[year]["total"] += 1
                    if is_collab:
                        year_stats[year]["collab"] += 1
                    year_stats[year]["counts"].append(count)

                    total_papers += 1
                    if is_collab:
                        collab_papers += 1
                    coauthor_counts.append(count)
                    team_size_counter[count] += 1

                    journal_total += 1
                    if is_collab:
                        journal_collab += 1
                    journal_coauthor_counts.append(count)

            
            if journal_total > 0:
                journal_metrics = {
                    "journal": journal_file.replace(".csv", ""),
                    "total_articles": journal_total,
                    "collaborative_articles": journal_collab,
                    "collab_percentage": round(journal_collab / journal_total * 100, 1),
                    "avg_team_size": round(np.mean(journal_coauthor_counts), 3),
                    "median_team_size": round(np.median(journal_coauthor_counts), 3),
                    "mode_team_size": statistics.mode(journal_coauthor_counts) if journal_coauthor_counts else 0
                }
                journal_summaries.append(journal_metrics)

            print(f"Journal {journal_file}: processed")

        except Exception as e:
            print(f"Error processing {journal_file}: {e}")

    # Save team size distribution
    team_size_df = pd.DataFrame.from_dict(team_size_counter, orient='index').reset_index()
    team_size_df.columns = ['team_size', 'count']
    team_size_df = team_size_df.sort_values(by='team_size')
    team_size_df.to_csv(f"{RESULTS_DIR}/team_size_distribution.csv", index=False)
    print(f"Saved team size distribution to {RESULTS_DIR}/team_size_distribution.csv")

    # Save per-journal collaboration metrics
    journal_summary_df = pd.DataFrame(journal_summaries)
    journal_summary_df = journal_summary_df.sort_values(by="total_articles", ascending=False)
    journal_summary_df.to_csv(f"{RESULTS_DIR}/per_journal_collab_metrics.csv", index=False)
    print(f"Saved per-journal collaboration metrics to {RESULTS_DIR}/per_journal_collab_metrics.csv")

    # Overall stats
    collab_proportion = collab_papers / total_papers if total_papers > 0 else 0
    avg_team_size = sum(coauthor_counts) / len(coauthor_counts) if coauthor_counts else 0
    median_team_size = np.median(coauthor_counts) if coauthor_counts else 0
    mode_team_size = statistics.mode(coauthor_counts) if coauthor_counts else 0

    overall_metrics = {
        "total_articles": total_papers,
        "collaborative_articles": collab_papers,
        "collab_percentage": round(collab_proportion * 100, 1),
        "avg_team_size": round(avg_team_size, 3),
        "median_team_size": round(median_team_size, 3),
        "mode_team_size": round(mode_team_size, 3)
    }

    # Yearly breakdown
    rows = []
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        stats = year_stats.get(year)
        if not stats or stats["total"] == 0:
            continue
        total = stats["total"]
        collab = stats["collab"]
        avg_team_size = np.mean(stats["counts"])
        median_team_size = np.median(stats["counts"])
        mode_team_size = statistics.mode(stats["counts"]) if stats["counts"] else 0

        rows.append({
            "year": year,
            "total_articles": total,
            "collaborative_articles": collab,
            "collab_percentage": round(collab / total * 100, 1),
            "avg_team_size": round(avg_team_size, 3),
            "median_team_size": round(median_team_size, 3),
            "mode_team_size": round(mode_team_size, 3)
        })

    yearly_df = pd.DataFrame(rows).sort_values("year")

    # Add rolling columns
    for col in ["avg_team_size", "median_team_size", "mode_team_size", "collab_percentage"]:
        yearly_df[f"{col}_rollavg_{ROLLING_WINDOW}"] = (
            yearly_df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean().round(3)
        )

    yearly_df.to_csv(f"{RESULTS_DIR}/yearly_collab_metrics.csv", index=False)
    print(f"Saved yearly collaboration metrics (with rolling) to {RESULTS_DIR}/yearly_collab_metrics.csv")

    return overall_metrics

def compute_sna_summary(g, full_time):
    g.simplify()  # Remove self-loops and multiple edges
    degrees = g.degree()

    if full_time:
        # Save degree distribution
        degree_counts = Counter(degrees)
        degree_df = pd.DataFrame.from_dict(degree_counts, orient='index').reset_index()
        degree_df.columns = ['degree', 'count']
        degree_df = degree_df.sort_values(by='degree')
        degree_df.to_csv(f"{RESULTS_DIR}/degree_distribution.csv", index=False)
        print(f"Saved degree distribution to {RESULTS_DIR}/degree_distribution.csv")

    return {
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "average_degree": round(np.mean(degrees), 7),
        "median_degree": round(np.median(degrees), 7),
        "mode_degree": int(statistics.mode(degrees)) if degrees else 0,
        "density": round(g.density(), 7),
        "clustering_coefficient": round(g.transitivity_avglocal_undirected(), 7),
        "connected_components": len(g.clusters()) if g.clusters() else 0,
        "largest_component_size": g.clusters().giant().vcount() if g.clusters() else 0
    }

def compute_yearly_metrics():
    sna_yearly_data = []

    for year in range(MIN_YEAR, MAX_YEAR + 1):
        path = os.path.join(GRAPH_YEARLY_DIR, f"{FIELD_NAME}_coauthorship_{year}.graphml")
        if not os.path.exists(path):
            continue

        g = load_graph(path)
        
        sna = compute_sna_summary(g, full_time=False)
        sna["year"] = year

        sna_yearly_data.append(sna)

    df = pd.DataFrame(sna_yearly_data).sort_values("year")

    rolling_cols = [
        "average_degree", "median_degree", "mode_degree", "density", "clustering_coefficient"
    ]

    for col in rolling_cols:
        df[f"{col}_rollavg_{ROLLING_WINDOW}"] = df[col].rolling(window=ROLLING_WINDOW, min_periods=1).mean().round(7)

    df.to_csv(YEARLY_SNA_CSV, index=False)
    print(f"Saved yearly SNA + rolling stats to {YEARLY_SNA_CSV}")

def main():
    # # Full graph SNA metrics
    # g = load_graph(GRAPH_FULL_PATH)
    # print("Computing SNA SUMMARY")
    # sna = compute_sna_summary(g, full_time=True)
    # pd.DataFrame([sna]).to_csv(SNA_OUT, index=False)
    # print(f"Saved full graph SNA metrics to {SNA_OUT}")

    # Yearly SNA metrics
    compute_yearly_metrics()

    # Number of collaborative papers
    collab_metrics = num_collaborative_papers()
    pd.DataFrame([collab_metrics]).to_csv(f"{RESULTS_DIR}/collab_metrics.csv", index=False)
    print(f"Saved overall collaboration metrics to {RESULTS_DIR}/collab_metrics.csv")

if __name__ == "__main__":
    main()
