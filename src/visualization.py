"""Matplotlib visualizations used by the Streamlit dashboard."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


LIGHT_PALETTE = ["#9147d1", "#ec4fae", "#6f63d9", "#f28bc5", "#bb63d4", "#d93691"]
DARK_PALETTE = ["#c084fc", "#fb71bd", "#8f8aff", "#ff9bcf", "#d987f2", "#ff68a9"]


def _style_axes(ax: plt.Axes, theme: str) -> tuple[str, str, list[str]]:
    """Apply matching chart surfaces, labels, and cluster colors."""
    dark = theme == "dark"
    background = "#222640" if dark else "#ffffff"
    foreground = "#f5efff" if dark else "#30233e"
    grid = "#41415c" if dark else "#eee5f1"
    palette = DARK_PALETTE if dark else LIGHT_PALETTE
    ax.set_facecolor(background)
    ax.figure.set_facecolor(background)
    ax.tick_params(colors=foreground)
    ax.xaxis.label.set_color(foreground)
    ax.yaxis.label.set_color(foreground)
    ax.title.set_color(foreground)
    for spine in ax.spines.values():
        spine.set_color(grid)
    ax.grid(color=grid, linewidth=0.8, alpha=0.8)
    return background, foreground, palette


def plot_elbow(elbow_data: pd.DataFrame, theme: str = "light") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    _, foreground, _ = _style_axes(ax, theme)
    sns.lineplot(data=elbow_data, x="K", y="Inertia", marker="o", ax=ax, color="#ed55ae", linewidth=2.8)
    ax.lines[0].set_markerfacecolor("#8b42d2")
    ax.lines[0].set_markeredgecolor("white")
    ax.set(title="Elbow method", xlabel="Number of clusters (K)", ylabel="Inertia (within-cluster sum of squares)")
    ax.set_xticks(elbow_data["K"].tolist())
    ax.title.set_color(foreground)
    fig.tight_layout()
    return fig


def plot_clusters(data: pd.DataFrame, x_feature: str, y_feature: str, theme: str = "light") -> plt.Figure:
    if x_feature not in data.columns or y_feature not in data.columns or "Cluster" not in data:
        raise ValueError("Choose two available features and run clustering first.")
    fig, ax = plt.subplots(figsize=(7, 5))
    _, foreground, palette = _style_axes(ax, theme)
    cluster_ids = sorted(data["Cluster"].unique())
    cluster_colors = {cluster: palette[index % len(palette)] for index, cluster in enumerate(cluster_ids)}
    sns.scatterplot(
        data=data, x=x_feature, y=y_feature, hue="Cluster", palette=cluster_colors,
        s=65, alpha=0.85, ax=ax,
    )
    ax.set(title="Customer groups", xlabel=x_feature, ylabel=y_feature)
    legend = ax.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    legend.get_title().set_color(foreground)
    for label in legend.get_texts():
        label.set_color(foreground)
    fig.tight_layout()
    return fig


def plot_cluster_sizes(data: pd.DataFrame, theme: str = "light") -> plt.Figure:
    sizes = data["Cluster"].value_counts().sort_index().rename_axis("Cluster").reset_index(name="Customers")
    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    _, foreground, palette = _style_axes(ax, theme)
    cluster_colors = {cluster: palette[index % len(palette)] for index, cluster in enumerate(sizes["Cluster"])}
    sns.barplot(data=sizes, x="Cluster", y="Customers", hue="Cluster", legend=False, palette=cluster_colors, ax=ax)
    ax.set(title="Customers in each cluster", xlabel="Cluster label", ylabel="Number of customers")
    ax.title.set_color(foreground)
    fig.tight_layout()
    return fig


def cluster_profiles(data: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[int, str]]:
    """Return cluster counts/percentages/feature means and data-led summaries."""
    if not features or "Cluster" not in data.columns:
        raise ValueError("Run clustering and select features to build cluster profiles.")
    grouped = data.groupby("Cluster", sort=True)[features]
    means = grouped.mean(numeric_only=True)
    counts = grouped.size().rename("Customers")
    profile = means.copy()
    profile.insert(0, "Customers", counts)
    profile.insert(1, "Dataset share (%)", (counts / len(data) * 100).round(1))

    descriptions: dict[int, str] = {}
    for cluster, row in means.iterrows():
        above = [str(feature) for feature in features if row[feature] > means[feature].mean()]
        below = [str(feature) for feature in features if row[feature] < means[feature].mean()]
        if above and below:
            summary = f"Higher {', '.join(above)}; lower {', '.join(below)} than the overall cluster averages."
        elif above:
            summary = f"Higher {', '.join(above)} than the overall cluster averages."
        elif below:
            summary = f"Lower {', '.join(below)} than the overall cluster averages."
        else:
            summary = "Close to the overall cluster averages for the selected features."
        descriptions[int(cluster)] = summary
    profile["Interpretation"] = [descriptions[int(cluster)] for cluster in profile.index]
    return profile, descriptions
