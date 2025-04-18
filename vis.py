import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import statistics
import seaborn as sns
from scipy import stats
import matplotlib as mpl
from matplotlib.gridspec import GridSpec


plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16

FIELD_NAME = "evoanth"
RESULTS_DIR = f"results/{FIELD_NAME}"
PLOT_DIR = f"{RESULTS_DIR}/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

RESULTS_BASE_DIR = f"results"
OVERALL_PLOT_DIR = f"{RESULTS_BASE_DIR}/plots"
os.makedirs(OVERALL_PLOT_DIR, exist_ok=True)
ALL_FIELDS = {"socanth":"Social Anthropology", "sociology":"Sociology", "psych":"Psychology", "evoanth":"Evolutionary Anthropology"} 

# File paths
COLLAB_CSV = f"{RESULTS_DIR}/collab_metrics.csv"
SNA_CSV = f"{RESULTS_DIR}/sna_results.csv"
YEARLY_COLLAB_CSV = f"{RESULTS_DIR}/yearly_collab_metrics.csv"
YEARLY_SNA_CSV = f"{RESULTS_DIR}/yearly_sna_metrics.csv"
TEAM_SIZE_DIST_CSV = f"{RESULTS_DIR}/team_size_distribution.csv"
DEGREE_DIST_CSV = f"{RESULTS_DIR}/degree_distribution.csv"

def plot_median_trends_across_fields(metric_column, ylabel, title, filename, source="collab"):
    fig, ax = plt.subplots(figsize=(8, 6))

    for field in os.listdir(RESULTS_BASE_DIR):
        field_dir = os.path.join(RESULTS_BASE_DIR, field)
        if not os.path.isdir(field_dir):
            continue
        yearly_file = os.path.join(field_dir, "yearly_collab_metrics.csv") if source == "collab" else os.path.join(field_dir, "yearly_sna_metrics.csv")
        if not os.path.isfile(yearly_file):
            continue

        try:
            df = pd.read_csv(yearly_file)
            if metric_column not in df.columns:
                continue
            label_name = ALL_FIELDS.get(field, field.capitalize())
            ax.plot(df["year"], df[metric_column], label=label_name, linewidth=2.5)
        except Exception as e:
            print(f"Failed to process {field}: {e}")
            continue

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks(np.arange(1925, 2026, 25))
    ax.set_xlim(1925, 2025)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OVERALL_PLOT_DIR, filename + ".pdf"), format='pdf')
    plt.close(fig)

def plot_distribution_with_lines(ax, data, xlabel, title, xlim=None):
    bins = range(0, max(data)+2)
    if xlim:
        bins = range(xlim[0], xlim[1] + 1)
    ax.hist(data, bins=bins, edgecolor='black', alpha=0.7, align='left', color='#4472C4')
    
    # Use distinct colors for mean, median, and mode
    ax.axvline(np.mean(data), color='#C00000', linestyle='--', linewidth=2, label=f"Mean: {np.mean(data):.1f}")
    ax.axvline(np.median(data), color='#548235', linestyle='-.', linewidth=2, label=f"Median: {int(np.median(data))}")
    ax.axvline(statistics.mode(data), color='#7030A0', linestyle=':', linewidth=3, label=f"Mode: {int(statistics.mode(data))}", zorder=10)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.3)
    if xlim:
        ax.set_xlim(0.5, xlim[1])

def gini_coefficient(array):
    sorted_arr = np.sort(array)
    n = len(array)
    cumulative = np.cumsum(sorted_arr)
    relative_mean_diff = (2 * np.sum((np.arange(1, n + 1) * sorted_arr))) / (n * np.sum(sorted_arr)) - (n + 1) / n
    return relative_mean_diff

def plot_lorenz_curve(ax, data, title, gini_loc='upper left'):
    sorted_vals = np.sort(data)
    cumvals = np.cumsum(sorted_vals)
    lorenz = np.insert(cumvals / cumvals[-1], 0, 0)
    x_vals = np.linspace(0, 1, len(lorenz))
    gini = gini_coefficient(data)

    ax.plot(x_vals, lorenz, label="Lorenz Curve", color='#4472C4', linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#7F7F7F', label="Line of Equality", linewidth=1.5)
    ax.set_title(title)
    ax.text(gini_loc[0], gini_loc[1], f"Gini: {gini:.3f}", transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#7F7F7F'))
    ax.set_xlabel("Cumulative Share of Authors")
    ax.set_ylabel("Cumulative Share of Coauthorships")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

def plot_power_law(ax, data, title, xlabel):
    values, counts = np.unique(data, return_counts=True)
    log_vals = np.log10(values[values > 0])
    log_counts = np.log10(counts[values > 0])

    slope, intercept, r_value, _, _ = stats.linregress(log_vals, log_counts)
    fit_line = slope * log_vals + intercept

    ax.loglog(values, counts, marker='o', linestyle='none', markersize=5, color='#4472C4', label="Empirical")
    ax.plot(10**log_vals, 10**fit_line, color='#C00000', linewidth=2, label=f"Fit: α={-slope:.2f}, $R^2$={r_value**2:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)

def plot_temporal_trends(fig, team_ax, collab_ax, degree_ax):
    collab_df = pd.read_csv(YEARLY_COLLAB_CSV)
    sna_df = pd.read_csv(YEARLY_SNA_CSV)
    merged_df = pd.merge(sna_df, collab_df, on="year", how="inner")

    # Plot collab percentage
    collab_ax.plot(merged_df["year"], merged_df["collab_percentage_rollavg_5"], 
                  label="% Collaborative Papers", color="#4472C4", linewidth=2.5)
    collab_ax.set_ylabel("% Collaborative Papers", color="#4472C4", fontweight='bold')
    collab_ax.tick_params(axis='y', labelcolor="#4472C4")
    collab_ax.set_xlabel("Year")
    
    collab_ax2 = collab_ax.twinx()
    collab_ax2.plot(merged_df["year"], merged_df["total_articles"], linestyle='dotted', 
                   color='#7F7F7F', linewidth=2, label="Total Articles")
    collab_ax2.set_ylabel("Total Articles", color="#7F7F7F", fontweight='bold')
    collab_ax2.tick_params(axis='y', labelcolor='#7F7F7F')
    collab_ax.set_title("Collaboration Percentage Over Time")
    collab_ax.grid(True, linestyle='--', alpha=0.3)
    
    # Add legend for both axes
    lines1, labels1 = collab_ax.get_legend_handles_labels()
    lines2, labels2 = collab_ax2.get_legend_handles_labels()
    collab_ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
                    facecolor='white', framealpha=0.9)

    # Plot degree trends
    degree_ax.plot(merged_df["year"], merged_df["average_degree_rollavg_5"], 
                  label="Average Degree", color="#4472C4", linewidth=2.5)
    degree_ax.plot(merged_df["year"], merged_df["median_degree_rollavg_5"], 
                  label="Median Degree", color="#ED7D31", linewidth=2.5)
    degree_ax.set_ylabel("Degree", fontweight='bold')
    degree_ax.set_xlabel("Year")
    degree_ax.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
    
    degree_ax2 = degree_ax.twinx()
    degree_ax2.plot(merged_df["year"], merged_df["total_articles"], linestyle='dotted', 
                   color='#7F7F7F', linewidth=2)
    degree_ax2.set_ylabel("Total Articles", color="#7F7F7F", fontweight='bold')
    degree_ax2.tick_params(axis='y', labelcolor='#7F7F7F')
    degree_ax.set_title("Degree Trends Over Time")
    degree_ax.grid(True, linestyle='--', alpha=0.3)

    # Plot team size trends
    team_ax.plot(merged_df["year"], merged_df["avg_team_size_rollavg_5"], 
                label="Average Authorship Size", color="#4472C4", linewidth=2.5)
    team_ax.plot(merged_df["year"], merged_df["median_team_size_rollavg_5"], 
                label="Median Authorship Size", color="#ED7D31", linewidth=2.5)
    team_ax.set_ylabel("Authorship Size", fontweight='bold')
    team_ax.set_xlabel("Year")
    team_ax.legend(loc="upper left", frameon=True, facecolor='white', framealpha=0.9)
    team_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    team_ax2 = team_ax.twinx()
    team_ax2.plot(merged_df["year"], merged_df["total_articles"], linestyle='dotted', 
                 color='#7F7F7F', linewidth=2)
    team_ax2.set_ylabel("Total Articles", color="#7F7F7F", fontweight='bold')
    team_ax2.tick_params(axis='y', labelcolor='#7F7F7F')
    team_ax.set_title("Authorship Size Trends Over Time")
    team_ax.grid(True, linestyle='--', alpha=0.3)

def main():
    print("Generating authorship size histogram with mean/median/mode...")
    team_df = pd.read_csv(TEAM_SIZE_DIST_CSV)
    all_teams = np.repeat(team_df["team_size"], team_df["count"]).astype(int)

    print("Generating degree histogram with mean/median/mode...")
    degree_df = pd.read_csv(DEGREE_DIST_CSV)
    all_degrees = np.repeat(degree_df["degree"], degree_df["count"]).astype(int)
    
    print("Generating individual high-quality figures...")
    
    # Team size distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_distribution_with_lines(
        ax,
        all_teams,
        xlabel="Authorship Size (Number of Authors)",
        title="Distribution of Authorship Sizes Across All Papers",
        xlim=(1, 50)
    )
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/team_size_distribution.pdf", format='pdf')
    plt.close(fig)
    
    # Degree distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_distribution_with_lines(
        ax,
        all_degrees,
        xlabel="Degree (Number of Collaborators)",
        title="Distribution of Author Degrees",
        xlim=(1, 50)
    )
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/degree_distribution.pdf", format='pdf')
    plt.close(fig)

    # Combined Lorenz curve for authorship sizes and degrees
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Calculate Lorenz curves and Gini coefficients
    # Authorship sizes
    sorted_teams = np.sort(all_teams)
    cum_teams = np.cumsum(sorted_teams)
    lorenz_teams = np.insert(cum_teams / cum_teams[-1], 0, 0)
    x_vals = np.linspace(0, 1, len(lorenz_teams))
    gini_teams = gini_coefficient(all_teams)
    
    # Degrees
    sorted_degrees = np.sort(all_degrees)
    cum_degrees = np.cumsum(sorted_degrees)
    lorenz_degrees = np.insert(cum_degrees / cum_degrees[-1], 0, 0)
    x_vals_degrees = np.linspace(0, 1, len(lorenz_degrees))
    gini_degrees = gini_coefficient(all_degrees)
    
    # Plot both curves
    ax.plot(x_vals, lorenz_teams, label="Authorship Sizes", color='#4472C4', linewidth=2.5)
    ax.plot(x_vals_degrees, lorenz_degrees, label="Author Degrees", color='#ED7D31', linewidth=2.5)
    ax.plot([0, 1], [0, 1], linestyle='--', color='#7F7F7F', label="Line of Equality", linewidth=1.5)
    
    # Add Gini coefficients as text
    ax.text(0.05, 0.85, f"Authorship Sizes Gini: {gini_teams:.3f}", transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#4472C4', boxstyle='round'), 
            color='#4472C4', fontweight='bold')
    ax.text(0.05, 0.78, f"Author Degrees Gini: {gini_degrees:.3f}", transform=ax.transAxes, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='#ED7D31', boxstyle='round'), 
            color='#ED7D31', fontweight='bold')
    
    ax.set_title("Inequality in Authorship Sizes and Author Degrees")
    ax.set_xlabel("Cumulative Share of Authors/Papers")
    ax.set_ylabel("Cumulative Share of Coauthorships/Authors")
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/combined_lorenz.pdf", format='pdf')
    plt.close(fig)

    # Power-law plot for authorship sizes
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_power_law(
        ax,
        all_teams,
        title="Power-Law Distribution of Authorship Sizes",
        xlabel="Authorship Size (Number of Authors)"
    )
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/team_size_powerlaw.pdf", format='pdf')
    plt.close(fig)

    # Power-law plot for degrees
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_power_law(
        ax,
        all_degrees,
        title="Power-Law Distribution of Author Degrees",
        xlabel="Degree (Number of Collaborators)"
    )
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/degree_powerlaw.pdf", format='pdf')
    plt.close(fig)

    # # Temporal trends in separate figures
    collab_df = pd.read_csv(YEARLY_COLLAB_CSV)
    sna_df = pd.read_csv(YEARLY_SNA_CSV)
    merged_df = pd.merge(sna_df, collab_df, on="year", how="inner")
    
    
    # Collaboration percentage trends
    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(merged_df["year"], merged_df["collab_percentage_rollavg_5"], 
            label="% Collaborative Papers", color="#4472C4", linewidth=2.5)
    ax1.set_ylabel("% Collaborative Papers", color="#4472C4", fontweight='bold')
    ax1.tick_params(axis='y', labelcolor="#4472C4")
    ax1.set_xlabel("Year")
    ax1.grid(True, linestyle='--', alpha=0.3)
    # ax1.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xticks(np.arange(1925, 2026, 25))
    ax1.set_xlim(1925, 2025)
    ax1.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.3)
    ax1.grid(axis='x', linestyle='')
    
    ax2 = ax1.twinx()
    ax2.plot(merged_df["year"], merged_df["total_articles"], linestyle='dotted', 
            color='#7F7F7F', linewidth=2, label="Total Articles")
    ax2.set_ylabel("Total Articles", color="#7F7F7F", fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#7F7F7F')
    ax2.grid(False)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, 
              facecolor='white', framealpha=0.9)
    
    plt.title("Collaboration Percentage Over Time")
    fig.tight_layout()
    fig.savefig(f"{PLOT_DIR}/collab_percentage_trends.pdf", format='pdf')
    plt.close(fig)


    print("Plotting median authorship size across fields...")
    plot_median_trends_across_fields(
        metric_column="median_team_size_rollavg_5",
        ylabel="Median Authorship Size",
        title="Median Authorship Size Over Time Across Fields",
        filename="median_team_size_across_fields",
        source="collab"
    )

    print("Plotting median author degree across fields...")
    plot_median_trends_across_fields(
        metric_column="median_degree_rollavg_5",
        ylabel="Median Degree (Coauthors)",
        title="Median Degree Over Time Across Fields",
        filename="median_degree_across_fields",
        source="sna"
    )

    print("Plotting average authorship size across fields...")
    plot_median_trends_across_fields(
        metric_column="avg_team_size_rollavg_5",
        ylabel="Average Authorship Size",
        title="Average Authorship Size Over Time Across Fields",
        filename="avg_team_size_across_fields",
        source="collab"
    )

    print("Plotting average author degree across fields...")
    plot_median_trends_across_fields(
        metric_column="avg_degree_rollavg_5",
        ylabel="Average Degree (Coauthors)",
        title="Average Degree Over Time Across Fields",
        filename="avg_degree_across_fields",
        source="sna"
    )

    print("Plotting collaboration percentage across fields...")
    plot_median_trends_across_fields(
        metric_column="collab_percentage_rollavg_5",
        ylabel="% Collaborative Articles",
        title="Collaboration Percentage Over Time Across Fields",
        filename="collab_percentage_across_fields",
        source="collab"
    )

if __name__ == "__main__":
    main()
