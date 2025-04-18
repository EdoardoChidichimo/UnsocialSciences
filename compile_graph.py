import csv
import igraph as ig
import os
from collections import defaultdict

from config import SELECTED_JOURNALS, FIELD_NAME, MAX_TEAM_SIZE,MIN_YEAR, MAX_YEAR

FULL_GRAPH_PATH = f"graphs/full/{FIELD_NAME}_coauthorship.graphml"
YEARLY_GRAPH_DIR = "graphs/yearly"
os.makedirs("graphs/full", exist_ok=True)
os.makedirs(YEARLY_GRAPH_DIR, exist_ok=True)

def build_coauthorship_graph(file_list, year_filter=None):
    all_author_ids = set()
    edge_dict = defaultdict(lambda: {"weight": 0, "years": set()})

    for file in file_list:
        with open(f"data/{FIELD_NAME}/{file}.csv", "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue

                year = row[0].strip()
                if year_filter and year != str(year_filter):
                    continue

                author_ids = [int(a.strip()) for a in row[1].strip().split(";") if a.strip().isdigit()]
                if len(author_ids) > MAX_TEAM_SIZE:
                    continue
                    
                if len(author_ids) < 2:
                    all_author_ids.update(author_ids)
                    continue

                all_author_ids.update(author_ids)

                for i in range(len(author_ids)):
                    for j in range(i + 1, len(author_ids)):
                        a, b = sorted((author_ids[i], author_ids[j]))
                        edge_dict[(a, b)]["weight"] += 1
                        if year_filter is None:
                            edge_dict[(a, b)]["years"].add(year)

    author_id_list = sorted(all_author_ids)
    author_id_to_index = {aid: idx for idx, aid in enumerate(author_id_list)}

    g = ig.Graph()
    g.add_vertices(len(author_id_list))
    g.vs["author_id"] = author_id_list

    edges = []
    weights = []
    years = []

    for (a, b), data in edge_dict.items():
        edges.append((author_id_to_index[a], author_id_to_index[b]))
        weights.append(data["weight"])
        if year_filter is None:
            years.append(",".join(sorted(data["years"])))

    g.add_edges(edges)
    g.es["weight"] = weights
    if year_filter is None:
        g.es["years"] = years

    return g

def main():
    # # Full graph
    # full_graph = build_coauthorship_graph(SELECTED_JOURNALS)
    # print(f"Full graph: {full_graph.vcount()} nodes, {full_graph.ecount()} edges.")
    # full_graph.write_graphml(FULL_GRAPH_PATH)
    # print(f"Saved full graph to {FULL_GRAPH_PATH}")

    # Yearly graphs
    for year in range(MIN_YEAR, MAX_YEAR + 1):
        yearly_graph = build_coauthorship_graph(SELECTED_JOURNALS, year_filter=year)
        if yearly_graph.ecount() == 0:
            print(f"No data for {year}, skipping.")
            continue
        out_path = os.path.join(YEARLY_GRAPH_DIR, f"{FIELD_NAME}_coauthorship_{year}.graphml")
        yearly_graph.write_graphml(out_path)
        print(f"Saved graph for {year} to {out_path}")

if __name__ == "__main__":
    main()