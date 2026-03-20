from __future__ import annotations

import html
import json
import math
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.helpers import load_config
from src.visualization.visualize import (
    plot_severity_distribution,
    plot_state_accidents,
    plot_temporal_trends,
)

sns.set_theme(style="whitegrid", palette="muted")

DISPLAY_MODEL_NAMES = {
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def format_int(value: int) -> str:
    return f"{int(value):,}"


def format_float(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def format_pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}%"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def model_display_name(model_name: str) -> str:
    return DISPLAY_MODEL_NAMES.get(model_name, model_name.replace("_", " ").title())


def relative_href(from_dir: Path, target_path: Path) -> str:
    return os.path.relpath(target_path, start=from_dir).replace("\\", "/")


def df_to_html_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    display_df = df.copy()
    for column in display_df.columns:
        if pd.api.types.is_integer_dtype(display_df[column]):
            display_df[column] = display_df[column].map(format_int)
    return display_df.to_html(index=False, classes="styled-table", border=0)


def load_data_artifacts(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    clean_cols = [
        "Severity",
        "State",
        "City",
        "Start_Time",
        "Temperature(F)",
        "Visibility(mi)",
        "Weather_Condition",
        "Distance(mi)",
        "Sunrise_Sunset",
    ]
    processed_cols = [
        "target",
        "weather_severity_score",
        "road_feature_count",
        "hour",
        "day_of_week",
        "month",
        "year",
        "is_weekend",
        "is_rush_hour",
        "is_night",
        "duration_min",
        "weather_category",
    ]

    clean_df = pd.read_parquet(cfg["paths"]["interim_data"], columns=clean_cols)
    processed_df = pd.read_parquet(cfg["paths"]["processed_data"], columns=processed_cols)
    training_summary = load_json(Path(cfg["paths"]["models_dir"]) / "training_summary.json")
    test_metrics = load_json(Path(cfg["paths"]["models_dir"]) / "test_metrics.json")
    feature_metadata = load_json(Path(cfg["paths"]["models_dir"]) / "feature_metadata.json")
    return clean_df, processed_df, training_summary, test_metrics, feature_metadata


def plot_weather_insights(clean_df: pd.DataFrame, processed_df: pd.DataFrame, output_path: Path) -> None:
    top_weather = (
        clean_df["Weather_Condition"]
        .value_counts()
        .head(8)
        .sort_values(ascending=True)
    )
    severity_rate = (
        processed_df.groupby("weather_category", observed=False)["target"]
        .mean()
        .sort_values(ascending=True)
        .mul(100)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    axes[0].barh(top_weather.index, top_weather.values, color="steelblue")
    axes[0].set_title("Top Weather Conditions")
    axes[0].set_xlabel("Accident Count")

    axes[1].barh(severity_rate.index, severity_rate.values, color="darkorange")
    axes[1].set_title("High-Severity Rate by Weather Category")
    axes[1].set_xlabel("Rate (%)")
    axes[1].set_xlim(0, max(25, severity_rate.max() + 2))

    plt.suptitle("Weather Insights", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_eda_figures(clean_df: pd.DataFrame, processed_df: pd.DataFrame, fig_dir: Path) -> dict[str, Path]:
    ensure_dir(fig_dir)

    severity_path = fig_dir / "eda_severity_distribution.png"
    temporal_path = fig_dir / "eda_temporal_trends.png"
    geography_path = fig_dir / "eda_state_accidents.png"
    weather_path = fig_dir / "eda_weather_insights.png"

    plot_severity_distribution(clean_df[["Severity"]].copy(), str(severity_path))
    plot_temporal_trends(clean_df[["Start_Time"]].copy(), str(temporal_path))
    plot_state_accidents(clean_df[["State"]].copy(), str(geography_path))
    plot_weather_insights(clean_df, processed_df, weather_path)

    return {
        "severity": severity_path,
        "temporal": temporal_path,
        "geography": geography_path,
        "weather": weather_path,
    }


def compute_report_context(
    clean_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    training_summary: dict,
    test_metrics: dict,
    feature_metadata: dict,
) -> dict:
    clean_df = clean_df.copy()
    clean_df["Start_Time"] = pd.to_datetime(clean_df["Start_Time"], errors="coerce")
    clean_df["year"] = clean_df["Start_Time"].dt.year
    clean_df["month"] = clean_df["Start_Time"].dt.month
    clean_df["hour"] = clean_df["Start_Time"].dt.hour
    clean_df["weekday"] = clean_df["Start_Time"].dt.day_name()

    severity_counts = clean_df["Severity"].value_counts().sort_index()
    severity_pct = clean_df["Severity"].value_counts(normalize=True).sort_index().mul(100)
    top_states = clean_df["State"].value_counts().head(10)
    top_cities = (
        clean_df.groupby(["City", "State"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    top_weather = clean_df["Weather_Condition"].value_counts().head(10)
    year_counts = clean_df["year"].value_counts().sort_index()
    month_counts = clean_df["month"].value_counts()
    hour_counts = clean_df["hour"].value_counts()
    weekday_counts = clean_df["weekday"].value_counts()
    daylight_counts = clean_df["Sunrise_Sunset"].value_counts()
    weather_target_rate = (
        processed_df.groupby("weather_category", observed=False)["target"]
        .mean()
        .sort_values(ascending=False)
        .mul(100)
    )

    training_df = pd.DataFrame(training_summary["results"])
    metrics_df = pd.DataFrame(test_metrics["test_metrics"])
    merged_metrics = metrics_df.merge(
        training_df[["model_name", "val_auc", "train_time_sec"]],
        left_on="model",
        right_on="model_name",
        how="left",
    ).drop(columns=["model_name"])
    merged_metrics["display_model"] = merged_metrics["model"].map(model_display_name)
    merged_metrics = merged_metrics.sort_values(["test_auc", "test_ap"], ascending=False).reset_index(drop=True)

    best_model = merged_metrics.iloc[0].to_dict()
    visibility_stats = clean_df["Visibility(mi)"].describe()
    road_feature_stats = processed_df["road_feature_count"].describe()

    findings = [
        (
            f"Severity 2 dominates the dataset at {format_pct(severity_pct.loc[2])}, while "
            f"the modeling target (Severity 3-4) accounts for {format_pct(processed_df['target'].mean() * 100)}."
        ),
        (
            f"Accidents cluster around commute windows, with the highest hourly volumes at "
            f"{int(hour_counts.idxmax())}:00 and {int(hour_counts.sort_values(ascending=False).index[1])}:00."
        ),
        (
            f"{weekday_counts.idxmax()} has the highest accident count, and "
            f"{format_pct(daylight_counts.get('Day', 0) / daylight_counts.sum() * 100)} of incidents occur during daylight."
        ),
        (
            f"California, Florida, and Texas lead total volume, with {model_display_name(best_model['model'])} delivering "
            f"the strongest held-out performance (test AUC {format_float(best_model['test_auc'], 4)})."
        ),
    ]

    caveats = []
    if visibility_stats["std"] == 0:
        caveats.append(
            "Visibility is constant at 10.0 miles after cleaning, so it contributes little usable variation to downstream analysis."
        )
    if road_feature_stats["max"] == 0:
        caveats.append(
            "All road-infrastructure flags were removed before feature engineering, leaving `road_feature_count` constant at 0 for every row."
        )
    if clean_df["Start_Time"].max().year == 2023:
        caveats.append(
            "The 2023 slice only covers data through March 31, 2023, so year-over-year totals for 2023 are not directly comparable."
        )

    severity_table = pd.DataFrame(
        {
            "Severity": severity_counts.index.astype(int),
            "Accidents": severity_counts.values.astype(int),
            "Share": [format_pct(v) for v in severity_pct.values],
        }
    )
    state_table = pd.DataFrame(
        {
            "State": top_states.index,
            "Accidents": top_states.values.astype(int),
        }
    )
    city_table = pd.DataFrame(
        {
            "City": [f"{city}, {state}" for city, state in top_cities.index],
            "Accidents": top_cities.values.astype(int),
        }
    )
    weather_table = pd.DataFrame(
        {
            "Weather Condition": top_weather.index,
            "Accidents": top_weather.values.astype(int),
        }
    )
    model_table = merged_metrics[
        [
            "display_model",
            "test_auc",
            "test_ap",
            "test_f1",
            "test_accuracy",
            "val_auc",
            "train_time_sec",
        ]
    ].rename(
        columns={
            "display_model": "Model",
            "test_auc": "Test AUC",
            "test_ap": "Test AP",
            "test_f1": "Test F1",
            "test_accuracy": "Test Accuracy",
            "val_auc": "Validation AUC",
            "train_time_sec": "Train Time (s)",
        }
    )

    model_table["Test AUC"] = model_table["Test AUC"].map(lambda v: format_float(v, 4))
    model_table["Test AP"] = model_table["Test AP"].map(lambda v: format_float(v, 4))
    model_table["Test F1"] = model_table["Test F1"].map(lambda v: format_float(v, 4))
    model_table["Test Accuracy"] = model_table["Test Accuracy"].map(lambda v: format_float(v, 4))
    model_table["Validation AUC"] = model_table["Validation AUC"].map(lambda v: format_float(v, 4))
    model_table["Train Time (s)"] = model_table["Train Time (s)"].map(lambda v: format_float(v, 2))

    weather_rate_table = pd.DataFrame(
        {
            "Weather Category": weather_target_rate.index,
            "High-Severity Rate": [format_pct(v) for v in weather_target_rate.values],
        }
    )

    total_records = len(clean_df)
    train_count = int(training_summary["results"][0]["n_train"])
    val_count = int(training_summary["results"][0]["n_val"])
    test_count = int(total_records - train_count - val_count)

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "project_name": "US Accidents EDA Portfolio",
        "record_count": total_records,
        "clean_columns": 33,
        "processed_columns": 55,
        "feature_count": len(feature_metadata["features"]),
        "start_date": clean_df["Start_Time"].min(),
        "end_date": clean_df["Start_Time"].max(),
        "target_rate_pct": processed_df["target"].mean() * 100,
        "best_model": {
            "name": model_display_name(best_model["model"]),
            "test_auc": best_model["test_auc"],
            "test_ap": best_model["test_ap"],
            "test_f1": best_model["test_f1"],
        },
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "distance_median": clean_df["Distance(mi)"].median(),
        "duration_median": processed_df["duration_min"].median(),
        "winter_months_share": month_counts.loc[[12, 1, 2]].sum() / month_counts.sum() * 100,
        "findings": findings,
        "caveats": caveats,
        "tables": {
            "severity": severity_table,
            "states": state_table,
            "cities": city_table,
            "weather": weather_table,
            "weather_rates": weather_rate_table,
            "models": model_table,
            "year_counts": pd.DataFrame(
                {
                    "Year": year_counts.index.astype(int),
                    "Accidents": year_counts.values.astype(int),
                }
            ),
        },
    }


def build_html_report(report_path: Path, context: dict, figure_paths: dict[str, Path], fig_dir: Path) -> None:
    reports_dir = report_path.parent
    tables = context["tables"]
    hrefs = {name: relative_href(reports_dir, path) for name, path in figure_paths.items()}
    roc_href = relative_href(reports_dir, fig_dir / "roc_curves.png")
    pr_href = relative_href(reports_dir, fig_dir / "pr_curves.png")

    html_content = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>US Accidents EDA Report</title>
  <style>
    :root {{ --bg:#f5f7fb; --card:#fff; --ink:#14213d; --muted:#5c677d; --accent:#d97706; --line:#d6deeb; --shadow:0 14px 30px rgba(20,33,61,.08); }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:"Segoe UI", Tahoma, sans-serif; color:var(--ink); background:radial-gradient(circle at top right, rgba(217,119,6,.10), transparent 28%), linear-gradient(180deg, #eef3fb 0%, var(--bg) 38%, #eef2f7 100%); line-height:1.6; }}
    .page {{ max-width:1180px; margin:0 auto; padding:40px 24px 64px; }}
    .hero {{ background:linear-gradient(135deg, #102542 0%, #1d3557 55%, #8d5524 100%); color:#fff; padding:40px; border-radius:28px; box-shadow:var(--shadow); }}
    .hero h1 {{ margin:0 0 8px; font-size:2.6rem; line-height:1.15; }}
    .hero p {{ margin:8px 0 0; color:rgba(255,255,255,.88); max-width:760px; }}
    .meta {{ margin-top:18px; font-size:.95rem; color:rgba(255,255,255,.72); }}
    .card-grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:16px; margin:24px 0 0; }}
    .card {{ background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.15); border-radius:18px; padding:18px 20px; }}
    .card .label {{ display:block; font-size:.85rem; color:rgba(255,255,255,.75); text-transform:uppercase; letter-spacing:.08em; }}
    .card .value {{ display:block; margin-top:6px; font-size:1.55rem; font-weight:700; }}
    section {{ margin-top:28px; background:var(--card); border:1px solid rgba(214,222,235,.8); border-radius:24px; padding:28px; box-shadow:var(--shadow); }}
    h2 {{ margin:0 0 8px; font-size:1.65rem; }}
    h3 {{ margin:22px 0 10px; font-size:1.15rem; }}
    .lede {{ color:var(--muted); margin:0 0 18px; }}
    .split {{ display:grid; grid-template-columns:1.2fr 1fr; gap:22px; align-items:start; }}
    .gallery {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:18px; margin-top:18px; }}
    figure {{ margin:0; background:#fff; border:1px solid var(--line); border-radius:18px; padding:12px; }}
    figure img {{ width:100%; border-radius:12px; display:block; }}
    figcaption {{ font-size:.92rem; color:var(--muted); margin-top:10px; }}
    .styled-table {{ width:100%; border-collapse:collapse; border-radius:16px; overflow:hidden; font-size:.96rem; }}
    .styled-table th {{ background:#143d59; color:#fff; text-align:left; padding:12px 14px; }}
    .styled-table td {{ padding:11px 14px; border-bottom:1px solid #e7edf4; vertical-align:top; }}
    .styled-table tr:nth-child(even) td {{ background:#f9fbfd; }}
    .bullet-list {{ margin:0; padding-left:20px; }}
    .bullet-list li {{ margin-bottom:10px; }}
    .chip-row {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:14px; }}
    .chip {{ background:#fff3e6; color:#8a4b08; border:1px solid #f4d3ae; border-radius:999px; padding:8px 12px; font-size:.9rem; }}
    .note {{ margin-top:18px; padding:14px 16px; border-left:4px solid var(--accent); background:#fff7ed; color:#7c4a12; border-radius:12px; }}
    .footer {{ margin-top:24px; text-align:center; color:var(--muted); font-size:.9rem; }}
    @media (max-width:900px) {{ .split {{ grid-template-columns:1fr; }} .hero {{ padding:28px; }} .hero h1 {{ font-size:2rem; }} }}
  </style>
</head>
<body>
  <div class=\"page\">
    <div class=\"hero\">
      <h1>{html.escape(context["project_name"])} EDA Report</h1>
      <p>Portfolio-ready summary of the cleaned US Accidents dataset, engineered features, and the held-out model evaluation artifacts produced by this repository.</p>
      <div class=\"meta\">Generated on {html.escape(context["generated_at"])} from project artifacts in <code>data/</code>, <code>models/</code>, and <code>reports/figures/</code>.</div>
      <div class=\"card-grid\">
        <div class=\"card\"><span class=\"label\">Accident Records</span><span class=\"value\">{format_int(context["record_count"])}</span></div>
        <div class=\"card\"><span class=\"label\">Date Range</span><span class=\"value\">{context["start_date"].strftime("%b %Y")} to {context["end_date"].strftime("%b %Y")}</span></div>
        <div class=\"card\"><span class=\"label\">High Severity Share</span><span class=\"value\">{format_pct(context["target_rate_pct"])}</span></div>
        <div class=\"card\"><span class=\"label\">Best Model</span><span class=\"value\">{html.escape(context["best_model"]["name"])}</span></div>
      </div>
    </div>

    <section>
      <h2>Executive Snapshot</h2>
      <p class=\"lede\">The project analyzes {format_int(context["record_count"])} US traffic accidents spanning {context["start_date"].strftime("%B %d, %Y")} through {context["end_date"].strftime("%B %d, %Y")}. After cleaning, the repository works from 33 analytical columns, engineers {context["feature_count"]} modeling features, and evaluates four binary classifiers where the target is <code>Severity &gt;= 3</code>.</p>
      <ul class=\"bullet-list\">{''.join(f'<li>{html.escape(item)}</li>' for item in context['findings'])}</ul>
      <div class=\"chip-row\">
        <span class=\"chip\">Train / Val / Test: {format_int(context["train_count"])} / {format_int(context["val_count"])} / {format_int(context["test_count"])}</span>
        <span class=\"chip\">Median distance: {format_float(context["distance_median"], 2)} mi</span>
        <span class=\"chip\">Median duration: {format_float(context["duration_median"], 2)} min</span>
        <span class=\"chip\">Winter-month share: {format_pct(context["winter_months_share"])}</span>
      </div>
    </section>

    <section>
      <h2>Severity And Time Patterns</h2>
      <p class=\"lede\">Severity class 2 dominates the raw label distribution, while high-severity accidents remain a meaningful minority large enough to support supervised modeling. Temporal volume is strongest in winter months, on weekdays, and around morning and evening commute windows.</p>
      <div class=\"split\">
        <div>{df_to_html_table(tables["severity"])}</div>
        <div>{df_to_html_table(tables["year_counts"])}</div>
      </div>
      <div class=\"gallery\">
        <figure>
          <img src=\"{hrefs['severity']}\" alt=\"Severity distribution\">
          <figcaption>Severity volume is concentrated in class 2, with classes 3 and 4 defining the modeling target.</figcaption>
        </figure>
        <figure>
          <img src=\"{hrefs['temporal']}\" alt=\"Temporal trends\">
          <figcaption>Year, month, hour, and weekday views show a strong commute signature and seasonal peaks.</figcaption>
        </figure>
      </div>
    </section>

    <section>
      <h2>Geography And Weather</h2>
      <p class=\"lede\">Geographic concentration is led by large, high-traffic states and metro areas. Fair and cloudy conditions account for the largest share of incidents by volume, but the highest severe-accident rate appears in the repository's rain/storm bucket.</p>
      <div class=\"split\">
        <div>
          <h3>Top States</h3>
          {df_to_html_table(tables["states"])}
          <h3>Top Cities</h3>
          {df_to_html_table(tables["cities"])}
        </div>
        <div>
          <h3>Top Weather Conditions</h3>
          {df_to_html_table(tables["weather"])}
          <h3>High-Severity Rate By Weather Category</h3>
          {df_to_html_table(tables["weather_rates"])}
        </div>
      </div>
      <div class=\"gallery\">
        <figure>
          <img src=\"{hrefs['geography']}\" alt=\"Top states\">
          <figcaption>California is the largest concentration center, followed by Florida and Texas.</figcaption>
        </figure>
        <figure>
          <img src=\"{hrefs['weather']}\" alt=\"Weather insights\">
          <figcaption>Weather volume is dominated by fair conditions, but severe-event rates rise in rain/storm settings.</figcaption>
        </figure>
      </div>
    </section>

    <section>
      <h2>Feature Engineering And Modeling Snapshot</h2>
      <p class=\"lede\">The modeling pipeline engineers temporal, weather, and categorical encodings before training Logistic Regression, Random Forest, XGBoost, and LightGBM. XGBoost delivers the strongest held-out discrimination, while LightGBM remains a strong speed-performance tradeoff.</p>
      {df_to_html_table(tables["models"])}
      <div class=\"gallery\">
        <figure>
          <img src=\"{roc_href}\" alt=\"ROC curves\">
          <figcaption>Held-out ROC comparison across all trained models.</figcaption>
        </figure>
        <figure>
          <img src=\"{pr_href}\" alt=\"PR curves\">
          <figcaption>Precision-recall behavior is especially relevant given the class imbalance in severe accidents.</figcaption>
        </figure>
      </div>
      <div class=\"note\">Best test result: {html.escape(context["best_model"]["name"])} with test AUC {format_float(context["best_model"]["test_auc"], 4)}, average precision {format_float(context["best_model"]["test_ap"], 4)}, and weighted F1 {format_float(context["best_model"]["test_f1"], 4)}.</div>
    </section>

    <section>
      <h2>Data Caveats</h2>
      <p class=\"lede\">These are the most important constraints to keep in mind when interpreting this project’s results.</p>
      <ul class=\"bullet-list\">{''.join(f'<li>{html.escape(item)}</li>' for item in context['caveats'])}</ul>
    </section>

    <div class=\"footer\">Companion PDF: <code>reports/final_report.pdf</code></div>
  </div>
</body>
</html>
"""
    report_path.write_text(html_content, encoding="utf-8")


def add_wrapped_lines(fig: plt.Figure, lines: list[str], x: float, y: float, width: int = 95, size: int = 11, step: float = 0.05) -> None:
    current_y = y
    for line in lines:
        wrapped = textwrap.fill(line, width=width)
        fig.text(x, current_y, wrapped, fontsize=size, va="top", ha="left")
        current_y -= step * (wrapped.count("\n") + 1)


def draw_table_page(pdf: PdfPages, title: str, subtitle: str, tables: list[tuple[str, pd.DataFrame]]) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.06, 0.96, title, fontsize=22, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.925, subtitle, fontsize=11, color="#4a5568", ha="left", va="top")

    n_tables = len(tables)
    for idx, (table_title, df) in enumerate(tables):
        top = 0.82 - idx * (0.78 / max(1, n_tables))
        height = 0.22 if n_tables > 2 else 0.30
        ax = fig.add_axes([0.06, top - height, 0.88, height])
        ax.axis("off")
        ax.set_title(table_title, loc="left", fontsize=13, pad=8)
        table_df = df.copy()
        for column in table_df.columns:
            if pd.api.types.is_integer_dtype(table_df[column]):
                table_df[column] = table_df[column].map(format_int)
        table = ax.table(
            cellText=table_df.values,
            colLabels=table_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="upper left",
            bbox=[0, 0, 1, 0.9],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        for (row, _), cell in table.get_celld().items():
            cell.set_edgecolor("#d8dee9")
            if row == 0:
                cell.set_facecolor("#143d59")
                cell.set_text_props(color="white", weight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f7fafc")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_summary_page(pdf: PdfPages, context: dict) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")

    fig.text(0.06, 0.95, context["project_name"], fontsize=26, fontweight="bold", ha="left", va="top")
    fig.text(0.06, 0.915, "Final Report", fontsize=18, color="#8d5524", ha="left", va="top")
    fig.text(
        0.06,
        0.88,
        (
            f"Generated on {context['generated_at']} from the cleaned parquet, engineered features, "
            "saved model metadata, and report figures already present in this repository."
        ),
        fontsize=11,
        color="#4a5568",
        ha="left",
        va="top",
    )

    cards = [
        ("Records", format_int(context["record_count"])),
        ("Window", f"{context['start_date'].strftime('%b %Y')} - {context['end_date'].strftime('%b %Y')}"),
        ("Target Rate", format_pct(context["target_rate_pct"])),
        ("Best Model", context["best_model"]["name"]),
    ]
    y = 0.79
    for label, value in cards:
        rect = plt.Rectangle((0.06, y - 0.085), 0.88, 0.07, transform=fig.transFigure, color="#f7fafc", ec="#d8dee9")
        fig.add_artist(rect)
        fig.text(0.08, y - 0.025, label, fontsize=11, color="#4a5568", ha="left", va="center")
        fig.text(0.38, y - 0.025, value, fontsize=16, fontweight="bold", ha="left", va="center")
        y -= 0.09

    fig.text(0.06, 0.40, "Key Findings", fontsize=16, fontweight="bold", ha="left")
    add_wrapped_lines(fig, [f"- {item}" for item in context["findings"]], x=0.08, y=0.37, width=90, size=11, step=0.055)

    fig.text(0.06, 0.15, "Primary Caveats", fontsize=16, fontweight="bold", ha="left")
    add_wrapped_lines(fig, [f"- {item}" for item in context["caveats"]], x=0.08, y=0.12, width=90, size=11, step=0.055)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def draw_image_grid_page(pdf: PdfPages, title: str, image_paths: list[Path], ncols: int, figsize: tuple[float, float]) -> None:
    existing_paths = [path for path in image_paths if path.exists()]
    if not existing_paths:
        return

    nrows = math.ceil(len(existing_paths) / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if hasattr(axes, "flatten"):
        axes = axes.flatten()
    else:
        axes = [axes]
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.98)

    for ax, image_path in zip(axes, existing_paths):
        img = mpimg.imread(image_path)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(image_path.stem.replace("_", " ").title(), fontsize=11, pad=8)

    for ax in axes[len(existing_paths):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_pdf_report(report_path: Path, context: dict, figure_paths: dict[str, Path], fig_dir: Path) -> None:
    eval_curve_paths = [
        fig_dir / "roc_curves.png",
        fig_dir / "pr_curves.png",
        fig_dir / "calibration_curves.png",
    ]
    confusion_paths = [
        fig_dir / "cm_logistic_regression.png",
        fig_dir / "cm_random_forest.png",
        fig_dir / "cm_xgboost.png",
        fig_dir / "cm_lightgbm.png",
    ]
    feature_importance_paths = [
        fig_dir / "feat_imp_logistic_regression.png",
        fig_dir / "feat_imp_random_forest.png",
        fig_dir / "feat_imp_xgboost.png",
        fig_dir / "feat_imp_lightgbm.png",
    ]

    with PdfPages(report_path) as pdf:
        draw_summary_page(pdf, context)
        draw_table_page(
            pdf,
            "Dataset Summary",
            "Severity balance, year coverage, and top geographic concentrations from the cleaned dataset.",
            [
                ("Severity Distribution", context["tables"]["severity"]),
                ("Year Counts", context["tables"]["year_counts"]),
                ("Top States", context["tables"]["states"].head(8)),
            ],
        )
        draw_table_page(
            pdf,
            "Model Comparison",
            "Held-out metrics come from models/test_metrics.json, with training time and validation AUC from training_summary.json.",
            [
                ("Held-Out Metrics", context["tables"]["models"]),
                ("Top Cities", context["tables"]["cities"].head(8)),
                ("Weather Rates", context["tables"]["weather_rates"]),
            ],
        )
        draw_image_grid_page(
            pdf,
            "EDA Figures",
            [
                figure_paths["severity"],
                figure_paths["temporal"],
                figure_paths["geography"],
                figure_paths["weather"],
            ],
            ncols=2,
            figsize=(11.69, 8.27),
        )
        draw_image_grid_page(
            pdf,
            "Model Evaluation Curves",
            eval_curve_paths,
            ncols=1,
            figsize=(8.27, 11.69),
        )
        draw_image_grid_page(
            pdf,
            "Confusion Matrices",
            confusion_paths,
            ncols=2,
            figsize=(11.69, 8.27),
        )
        draw_image_grid_page(
            pdf,
            "Feature Importance",
            feature_importance_paths,
            ncols=2,
            figsize=(11.69, 8.27),
        )


def main() -> None:
    cfg = load_config()
    reports_dir = Path(cfg["paths"]["reports_dir"])
    fig_dir = Path(cfg["paths"]["figures_dir"])
    ensure_dir(reports_dir)
    ensure_dir(fig_dir)

    clean_df, processed_df, training_summary, test_metrics, feature_metadata = load_data_artifacts(cfg)
    figure_paths = build_eda_figures(clean_df, processed_df, fig_dir)
    context = compute_report_context(clean_df, processed_df, training_summary, test_metrics, feature_metadata)

    html_report_path = reports_dir / "eda_report.html"
    pdf_report_path = reports_dir / "final_report.pdf"

    build_html_report(html_report_path, context, figure_paths, fig_dir)
    build_pdf_report(pdf_report_path, context, figure_paths, fig_dir)

    print(f"Wrote {html_report_path}")
    print(f"Wrote {pdf_report_path}")


if __name__ == "__main__":
    main()
