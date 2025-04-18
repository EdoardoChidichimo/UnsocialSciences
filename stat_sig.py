import pandas as pd
import numpy as np
import os
import scipy.stats as stats
from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

DATA_BASE_PATH = "data"
ALL_FIELDS = ["socanth", 
            "sociology", 
            "psych", 
            "evoanth"
            ]
RESULTS_DIR = "results"

def load_articles_by_field():
    field_data = {}
    for field_name in ALL_FIELDS:
        path = os.path.join(DATA_BASE_PATH, field_name) 
        dfs = []
        for f in os.listdir(path):
            if f.endswith(".csv") and f != f"{field_name}_author_index.csv":
                df_path = os.path.join(path, f)
                df = pd.read_csv(df_path, dtype=str, low_memory=False)
                df = df.dropna(subset=["year", "authors"])
                df["year"] = pd.to_numeric(df["year"], errors="coerce")
                df = df.dropna(subset=["year"])
                df["year"] = df["year"].astype(int)
                df["team_size"] = df["authors"].apply(lambda x: x.count(";") + 1 if isinstance(x, str) else np.nan)
                df["is_collab"] = df["team_size"].apply(lambda x: 1 if x and x > 1 else 0)
                dfs.append(df)
        if dfs:
            field_data[field_name] = pd.concat(dfs, ignore_index=True)
    return field_data

with open(os.path.join(RESULTS_DIR, "field_comparison_stats.txt"), "w", encoding="utf-8") as out_file:
    with redirect_stdout(out_file):
        field_data = load_articles_by_field()

        print("\n--- TEAM SIZE COMPARISON (Kruskal-Wallis + Pairwise Mann-Whitney + Effect Sizes) ---")
        team_sizes = [df["team_size"].dropna().values for df in field_data.values()]
        labels = list(field_data.keys())
        stat, p = stats.kruskal(*team_sizes)
        print(f"Kruskal-Wallis: H = {stat:.3f}, p = {p:.4f}")

        # Calculate effect size for Kruskal-Wallis (eta-squared)
        n = sum(len(ts) for ts in team_sizes)
        eta_squared = (stat - len(team_sizes) + 1) / (n - len(team_sizes))
        print(f"Effect size (eta-squared): {eta_squared:.4f} ({'small' if eta_squared < 0.06 else 'medium' if eta_squared < 0.14 else 'large'})")

        # Collect p-values for multiple comparison correction
        pairwise_results = []
        for (f1, d1), (f2, d2) in combinations(field_data.items(), 2):
            stat, p = stats.mannwhitneyu(d1["team_size"], d2["team_size"], alternative='two-sided')
            n1, n2 = len(d1), len(d2)
            # Calculate effect size r (equivalent to Cohen's d for non-parametric)
            r = abs(stat - (n1 * n2 / 2)) / (n1 * n2 / 2) if p > 0 else 0
            pairwise_results.append((f1, f2, stat, p, r))
        
        # Apply Bonferroni correction
        p_values = [res[3] for res in pairwise_results]
        reject, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
        
        # Print results with corrected p-values
        for i, (f1, f2, stat, p, r) in enumerate(pairwise_results):
            effect_size_interp = 'negligible' if r < 0.1 else 'small' if r < 0.3 else 'medium' if r < 0.5 else 'large'
            print(f"{f1} vs {f2}: U = {stat:.0f}, p = {p_adjusted[i]:.4f} (corrected), effect size r = {r:.4f} ({effect_size_interp})")

        print("\n--- COLLABORATION RATE COMPARISON (Chi-square) ---")
        collab_table = pd.DataFrame({f: df["is_collab"].value_counts() for f, df in field_data.items()}).T.fillna(0)
        chi2, p, dof, _ = stats.chi2_contingency(collab_table)
        print(f"Chi-square: chi2 = {chi2:.2f}, p = {p:.4f}, dof = {dof}")
        
        # Calculate Cramer's V for effect size
        n = collab_table.sum().sum()
        cramer_v = np.sqrt(chi2 / (n * (min(collab_table.shape) - 1)))
        effect_size_interp = 'negligible' if cramer_v < 0.1 else 'small' if cramer_v < 0.3 else 'medium' if cramer_v < 0.5 else 'large'
        print(f"Effect size (Cramer's V): {cramer_v:.4f} ({effect_size_interp})")

        print("\n--- TEMPORAL TREND ANALYSIS: Is Social Anthropology increasing slower than other fields? ---")
        temporal_data = []
        for field, df in field_data.items():
            grouped = df.groupby("year")["is_collab"].mean().reset_index()
            grouped["field"] = field
            temporal_data.append(grouped)

        temp_df = pd.concat(temporal_data, ignore_index=True)

        # Fit the model
        model = ols("is_collab ~ year * field", data=temp_df).fit()
        print(model.summary())

        # ANOVA analysis
        anova_results = sm.stats.anova_lm(model, typ=2)
        print("\nANOVA on temporal trend model:")
        print(anova_results)
        
        # Calculate effect sizes for ANOVA (partial eta-squared)
        anova_results['partial_eta_sq'] = anova_results['sum_sq'] / (anova_results['sum_sq'] + anova_results['sum_sq'].sum())
        print("\nEffect sizes (partial eta-squared):")
        for idx, row in anova_results.iterrows():
            effect_size = row['partial_eta_sq']
            effect_size_interp = 'small' if effect_size < 0.06 else 'medium' if effect_size < 0.14 else 'large'
            print(f"{idx}: {effect_size:.4f} ({effect_size_interp})")

        # Save temporal data
        temp_df.to_csv(os.path.join(RESULTS_DIR, "temporal_collab_by_field.csv"), index=False)
        print(f"Saved temporal collaboration data to {os.path.join(RESULTS_DIR, 'temporal_collab_by_field.csv')}")
        
        # Create visualisation of temporal trends
        plt.figure(figsize=(12, 8))
        sns.lineplot(data=temp_df, x="year", y="is_collab", hue="field", marker="o")
        plt.title("Collaboration Rate by Field Over Time")
        plt.xlabel("Year")
        plt.ylabel("Collaboration Rate")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(RESULTS_DIR, "temporal_collab_trends.png"), dpi=300, bbox_inches='tight')
        print(f"Saved temporal trend visualisation to {os.path.join(RESULTS_DIR, 'temporal_collab_trends.png')}")