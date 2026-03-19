"""
visualize.py
Reusable EDA visualization functions used by notebooks and the dashboard.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
PALETTE = px.colors.qualitative.Set2


def save_fig(fig, path: str, dpi: int = 150):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ── Static (Matplotlib/Seaborn) ──────────────────────────────────────────────

def plot_severity_distribution(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts = df["Severity"].value_counts().sort_index()
    axes[0].bar(counts.index, counts.values, color=sns.color_palette("muted", 4))
    axes[0].set(title="Accident Count by Severity", xlabel="Severity", ylabel="Count")
    for i, v in enumerate(counts.values):
        axes[0].text(counts.index[i], v + 5000, f"{v/len(df)*100:.1f}%", ha="center")
    axes[1].pie(counts.values, labels=[f"Severity {i}" for i in counts.index],
                autopct="%1.1f%%", colors=sns.color_palette("muted", 4))
    axes[1].set_title("Severity Distribution")
    plt.suptitle("US Accidents – Severity Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_temporal_trends(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    df["year"] = pd.to_datetime(df["Start_Time"]).dt.year
    df["month"] = pd.to_datetime(df["Start_Time"]).dt.month
    df["hour"] = pd.to_datetime(df["Start_Time"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["Start_Time"]).dt.day_name()

    df.groupby("year").size().plot(ax=axes[0, 0], marker="o", color="steelblue")
    axes[0, 0].set(title="Accidents per Year", xlabel="Year", ylabel="Count")

    df.groupby("month").size().plot(ax=axes[0, 1], kind="bar", color="coral")
    axes[0, 1].set(title="Accidents per Month", xlabel="Month", ylabel="Count")

    df.groupby("hour").size().plot(ax=axes[1, 0], marker="o", color="green")
    axes[1, 0].set(title="Accidents by Hour of Day", xlabel="Hour", ylabel="Count")

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_counts = df.groupby("day_of_week").size().reindex(day_order)
    day_counts.plot(ax=axes[1, 1], kind="bar", color="purple")
    axes[1, 1].set(title="Accidents by Day of Week", xlabel="Day", ylabel="Count")
    axes[1, 1].tick_params(axis="x", rotation=45)

    plt.suptitle("Temporal Patterns in US Accidents", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_weather_analysis(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if "Temperature(F)" in df.columns:
        df["Temperature(F)"].dropna().hist(ax=axes[0, 0], bins=50, color="tomato", edgecolor="white")
        axes[0, 0].set(title="Temperature Distribution", xlabel="°F")

    if "Visibility(mi)" in df.columns:
        df["Visibility(mi)"].clip(0, 15).dropna().hist(ax=axes[0, 1], bins=40, color="steelblue", edgecolor="white")
        axes[0, 1].set(title="Visibility Distribution", xlabel="Miles")

    if "Weather_Condition" in df.columns:
        top_weather = df["Weather_Condition"].value_counts().head(10)
        top_weather.plot(ax=axes[1, 0], kind="barh", color="mediumseagreen")
        axes[1, 0].set(title="Top 10 Weather Conditions", xlabel="Count")

    if "Wind_Speed(mph)" in df.columns:
        df["Wind_Speed(mph)"].clip(0, 60).dropna().hist(ax=axes[1, 1], bins=40, color="orchid", edgecolor="white")
        axes[1, 1].set(title="Wind Speed Distribution", xlabel="mph")

    plt.suptitle("Weather Conditions at Accident Sites", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, mask=mask, annot=False, cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


def plot_state_accidents(df: pd.DataFrame, save_path: str = None):
    state_counts = df["State"].value_counts().reset_index()
    state_counts.columns = ["State", "Count"]
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.barplot(data=state_counts.head(20), x="State", y="Count", palette="viridis", ax=ax)
    ax.set(title="Top 20 States by Accident Count", xlabel="State", ylabel="Count")
    plt.tight_layout()
    if save_path:
        save_fig(fig, save_path)
    return fig


# ── Interactive (Plotly) ─────────────────────────────────────────────────────

def plotly_accident_map(df: pd.DataFrame, sample_n: int = 50000) -> go.Figure:
    sample = df.dropna(subset=["Start_Lat", "Start_Lng"]).sample(
        min(sample_n, len(df)), random_state=42
    )
    fig = px.scatter_mapbox(
        sample, lat="Start_Lat", lon="Start_Lng",
        color="Severity", color_continuous_scale="RdYlGn_r",
        size_max=5, zoom=3, height=600,
        mapbox_style="carto-positron",
        title="US Accident Locations (sample)",
        hover_data=["State", "City", "Weather_Condition"] if "City" in sample.columns else ["State"],
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def plotly_severity_by_state(df: pd.DataFrame) -> go.Figure:
    state_sev = df.groupby("State")["Severity"].mean().reset_index()
    state_sev.columns = ["State", "Avg_Severity"]
    fig = px.choropleth(
        state_sev, locations="State", locationmode="USA-states",
        color="Avg_Severity", scope="usa",
        color_continuous_scale="RdYlGn_r",
        title="Average Accident Severity by State",
        labels={"Avg_Severity": "Avg Severity"},
    )
    return fig


def plotly_hourly_heatmap(df: pd.DataFrame) -> go.Figure:
    df = df.copy()
    df["hour"] = pd.to_datetime(df["Start_Time"]).dt.hour
    df["day_of_week"] = pd.to_datetime(df["Start_Time"]).dt.dayofweek
    pivot = df.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=list(range(24)), y=days,
        colorscale="YlOrRd", colorbar={"title": "Accidents"},
    ))
    fig.update_layout(
        title="Accident Frequency: Day × Hour",
        xaxis_title="Hour of Day", yaxis_title="Day of Week",
    )
    return fig


def plotly_weather_severity(df: pd.DataFrame) -> go.Figure:
    if "weather_category" not in df.columns:
        return go.Figure()
    wdf = df.groupby(["weather_category", "Severity"]).size().reset_index(name="count")
    fig = px.bar(wdf, x="weather_category", y="count", color="Severity",
                 barmode="group", title="Accidents by Weather Category and Severity",
                 color_continuous_scale="RdYlGn_r")
    return fig


def plotly_top_cities(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    city_counts = df.groupby(["City", "State"]).size().reset_index(name="count")
    city_counts["city_state"] = city_counts["City"] + ", " + city_counts["State"]
    top = city_counts.nlargest(top_n, "count")
    fig = px.bar(top, x="count", y="city_state", orientation="h",
                 title=f"Top {top_n} Cities by Accident Count",
                 labels={"count": "Accidents", "city_state": "City"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig

