import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dash_table, dcc, html

from src.utils.helpers import load_config

cfg = load_config()
API_BASE = cfg["dashboard"]["api_base_url"]
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
FONT_STACK = "Aptos, Segoe UI Variable Text, Trebuchet MS, sans-serif"
PLOT_GRID = "rgba(148, 163, 184, 0.14)"
CATEGORY_SEQUENCE = ["#67d2ff", "#7cf3bc", "#ffcf6b", "#f472b6", "#a78bfa", "#fb8a80"]
GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="US Accidents EDA Portfolio",
    assets_folder=ASSETS_DIR,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # expose Flask server for production

# -- Layout helpers ------------------------------------------------------------

def apply_figure_style(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT_STACK, "color": "#edf5ff"},
        margin={"l": 42, "r": 20, "t": 74, "b": 42},
        title={"x": 0.02, "xanchor": "left", "font": {"size": 22, "color": "#edf5ff"}},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(0,0,0,0)",
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor=PLOT_GRID, linecolor=PLOT_GRID, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=PLOT_GRID, linecolor=PLOT_GRID, zeroline=False)
    return fig


def empty_figure(title, message="Data is unavailable right now."):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16, "color": "#b7c5db"},
        xref="paper",
        yref="paper",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(showlegend=False, title=title)
    return apply_figure_style(fig)


def metric_chip(icon, text):
    return html.Span([html.I(className=f"fa {icon}"), html.Span(text)], className="hero-pill")


def page_hero(eyebrow, title, description, chips, metrics):
    return dbc.Card(
        dbc.CardBody(
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Span(eyebrow, className="hero-eyebrow"),
                            html.H1(title, className="hero-title"),
                            html.P(description, className="hero-description"),
                            html.Div(chips, className="hero-pill-row"),
                        ],
                        lg=8,
                    ),
                    dbc.Col(
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(label, className="hero-metric-label"),
                                        html.H4(value, className="hero-metric-value"),
                                    ],
                                    className="hero-metric",
                                )
                                for label, value in metrics
                            ],
                            className="hero-metric-grid",
                        ),
                        lg=4,
                    ),
                ],
                className="g-4 align-items-center",
            )
        ),
        className="page-hero border-0",
    )


def stat_card(title, value, icon, color="primary", detail=""):
    return dbc.Card(
        dbc.CardBody(
            html.Div(
                [
                    html.Span(html.I(className=f"fa {icon} fa-lg"), className="metric-icon"),
                    html.Div(
                        [
                            html.H3(value, className="mb-1 fw-bold"),
                            html.Span(title, className="metric-label"),
                            html.P(detail, className="metric-detail") if detail else None,
                        ]
                    ),
                ],
                className="d-flex align-items-start gap-3",
            )
        ),
        className=f"metric-card accent-{color} border-0 h-100",
    )


def graph_card(title, description, graph_id, height="360px", pill=None):
    intro = [
        html.Div(
            [
                html.H3(title, className="section-heading"),
                html.P(description, className="section-description"),
            ],
            className="section-topline",
        )
    ]
    if pill:
        intro.append(html.Span(pill, className="section-pill"))
    return dbc.Card(
        dbc.CardBody(
            intro + [
                dcc.Loading(
                    dcc.Graph(
                        id=graph_id,
                        config=GRAPH_CONFIG,
                        className="dashboard-graph",
                        style={"height": height},
                    ),
                    color="#67d2ff",
                )
            ]
        ),
        className="section-card border-0 h-100",
    )


def info_card(title, description, items, icon="fa-lightbulb"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Span(html.I(className=f"fa {icon}"), className="metric-icon me-3"),
                        html.Div(
                            [
                                html.H3(title, className="section-heading"),
                                html.P(description, className="section-description"),
                            ]
                        ),
                    ],
                    className="d-flex align-items-start",
                ),
                html.Hr(style={"borderColor": PLOT_GRID}),
                html.Ul([html.Li(item) for item in items], className="insight-list"),
            ]
        ),
        className="info-card border-0 h-100",
    )


def prediction_placeholder():
    return html.Div(
        [
            html.Span(html.I(className="fa fa-bolt"), className="prediction-placeholder-icon"),
            html.Div(
                [
                    html.H3("Ready for a scenario test", className="section-heading"),
                    html.P(
                        "Adjust the context on the left and run a prediction to see severity, confidence, and the model used for that estimate.",
                        className="section-description",
                    ),
                ]
            ),
            html.Div(
                [
                    html.Span("Time and season effects", className="result-chip"),
                    html.Span("Weather stress signals", className="result-chip"),
                    html.Span("Road complexity", className="result-chip"),
                ],
                className="result-chip-row",
            ),
        ],
        className="prediction-placeholder",
    )


NAV_STYLE = {
    "background": "rgba(5, 12, 22, 0.76)",
    "borderBottom": "1px solid rgba(143, 168, 201, 0.14)",
    "padding": "0.5rem 1.4rem",
    "boxShadow": "0 10px 30px rgba(2, 7, 18, 0.24)",
}
BRAND_STYLE = {
    "fontWeight": "700",
    "fontSize": "1.05rem",
    "letterSpacing": "0.03em",
    "color": "#edf5ff",
    "textDecoration": "none",
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
}
NAV_LINK_STYLE = {
    "color": "#b7c5db",
    "fontWeight": "600",
    "fontSize": "0.92rem",
    "padding": "0.65rem 1rem",
    "borderRadius": "999px",
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "textDecoration": "none",
}

navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            html.Div([
                html.I(className="fa fa-car-crash", style={"color": "#ff8a80", "fontSize": "1.3rem"}),
                html.Span("US Accidents", style={"color": "#edf5ff"}),
                html.Span(" Portfolio", style={"color": "#67d2ff"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            href="/", style=BRAND_STYLE,
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav(id="nav-links", navbar=True, className="ms-auto gap-2 align-items-center"),
            id="navbar-collapse", navbar=True,
        ),
    ], fluid=True),
    style=NAV_STYLE,
    className="portfolio-navbar mb-4 mb-lg-5",
    dark=True,
)


def overview_layout():
    return dbc.Container([
        page_hero(
            "Portfolio Dashboard",
            "A cleaner, sharper view of accident risk patterns across the dataset.",
            "Track severity, temporal movement, geographic hotspots, and API-backed model insights from one polished dashboard.",
            [
                metric_chip("fa-signal", "Live API powered"),
                metric_chip("fa-clock", "Refreshes every 60 seconds"),
                metric_chip("fa-layer-group", "Four portfolio views"),
            ],
            [("Views", "4 pages"), ("Refresh", "60 sec"), ("Coverage", "National"), ("Focus", "EDA + ML")],
        ),
        dbc.Row(id="stat-cards", className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(
                graph_card(
                    "Severity distribution",
                    "A fast read on how incidents are distributed across severity levels.",
                    "severity-dist-chart",
                    pill="Live summary",
                ),
                lg=5,
            ),
            dbc.Col(
                graph_card(
                    "Accident timeline",
                    "Spot volume shifts across time and watch for sustained changes in activity.",
                    "accidents-over-time-chart",
                    pill="Trend view",
                ),
                lg=7,
            ),
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(
                graph_card(
                    "Geographic risk snapshot",
                    "See how average severity varies by state across the United States.",
                    "severity-state-map",
                    height="520px",
                    pill="State severity map",
                ),
                lg=12,
            ),
        ], className="g-4"),
        dcc.Interval(id="overview-interval", interval=60_000, n_intervals=0),
    ], fluid=True, className="page-shell pb-5")


def eda_layout():
    return dbc.Container([
        page_hero(
            "Interactive Analysis",
            "Switch the lens and explore how time, weather, cities, and road complexity shape accident outcomes.",
            "Choose a view, scan the main chart, then compare the supporting charts for deeper context.",
            [
                metric_chip("fa-chart-line", "Responsive charts"),
                metric_chip("fa-sliders-h", "Interactive controls"),
                metric_chip("fa-map-marked-alt", "Multi-angle exploration"),
            ],
            [("Modes", "4 lenses"), ("Support", "2 charts"), ("Readout", "Instant"), ("Goal", "Story-led EDA")],
        ),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H3("Choose an analysis lens", className="section-heading"),
                        html.P(
                            "Use the selector to jump between common accident narratives.",
                            className="section-description",
                        ),
                        dcc.Dropdown(
                            id="eda-selector",
                            options=[
                                {"label": "Accidents by Hour of Day", "value": "hour"},
                                {"label": "Weather Impact on Severity", "value": "weather"},
                                {"label": "Top Cities by Accident Count", "value": "cities"},
                                {"label": "Road Features Impact", "value": "road"},
                            ],
                            value="hour",
                            clearable=False,
                            className="dash-dropdown mt-3",
                        ),
                        html.Div(id="eda-selection-copy", className="section-description mt-3"),
                    ]),
                ], className="section-card control-panel border-0 h-100"),
                lg=4,
            ),
            dbc.Col(
                info_card(
                    "What to look for",
                    "Each view is meant to answer a slightly different portfolio question.",
                    [
                        "Hour view shows when accident load builds and which periods carry more severe cases.",
                        "Weather view highlights whether harsher environmental conditions align with higher average severity.",
                        "Cities and road-feature views help translate raw counts into location and infrastructure stories.",
                    ],
                    icon="fa-binoculars",
                ),
                lg=8,
            ),
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(
                graph_card(
                    "Primary analysis",
                    "The main chart updates to match the selected storyline.",
                    "eda-main-chart",
                    height="430px",
                ),
                lg=12,
            )
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(
                graph_card(
                    "Supporting chart",
                    "Use this comparison to validate or challenge what the main chart suggests.",
                    "eda-secondary-chart",
                ),
                lg=6,
            ),
            dbc.Col(
                graph_card(
                    "Pattern cross-check",
                    "A second angle for density, mix, or state-level context depending on the selected analysis.",
                    "eda-heatmap-chart",
                ),
                lg=6,
            ),
        ], className="g-4"),
    ], fluid=True, className="page-shell pb-5")


def models_layout():
    return dbc.Container([
        page_hero(
            "Model Benchmarking",
            "Compare classification quality at a glance and communicate why one model stands out.",
            "A readable metrics table, grouped bars, and a radar chart make the portfolio walkthrough much easier to follow.",
            [
                metric_chip("fa-robot", "Model comparison"),
                metric_chip("fa-bullseye", "Metric focused"),
                metric_chip("fa-sync-alt", "Refreshes every 5 minutes"),
            ],
            [("Metrics", "AUC, F1, precision"), ("Views", "Table + 2 charts"), ("Refresh", "5 min"), ("Use case", "Model selection")],
        ),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.Div([
                            html.H3("Evaluation table", className="section-heading"),
                            html.P(
                                "Structured for quick scanning, with the top AUC result highlighted.",
                                className="section-description",
                            ),
                        ], className="section-topline"),
                        dcc.Loading(html.Div(id="metrics-table-container"), color="#67d2ff"),
                    ]),
                ], className="table-card border-0"),
                lg=12,
            )
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(
                graph_card(
                    "Metric comparison",
                    "Compare core holdout metrics across the shortlisted models.",
                    "metrics-bar-chart",
                    pill="Grouped bars",
                ),
                lg=7,
            ),
            dbc.Col(
                graph_card(
                    "Performance shape",
                    "The radar view makes tradeoffs easier to read than a raw table alone.",
                    "metrics-radar-chart",
                    pill="Radar profile",
                ),
                lg=5,
            ),
        ], className="g-4 mb-4"),
        dbc.Row([
            dbc.Col(
                info_card(
                    "Comparison notes",
                    "Use these takeaways to narrate the strengths of each model during a walkthrough.",
                    [
                        "XGBoost and LightGBM are usually the strongest options for structured tabular accident data.",
                        "Logistic regression remains a solid baseline when interpretability matters more than peak performance.",
                        "Random forest is typically resilient to noisy inputs and non-linear interactions.",
                        "All trained models account for class imbalance, which matters for rare high-severity outcomes.",
                    ],
                    icon="fa-notes-medical",
                ),
                lg=12,
            ),
        ], className="g-4"),
        dcc.Interval(id="models-interval", interval=300_000, n_intervals=0),
    ], fluid=True, className="page-shell pb-5")


def predict_layout():
    input_style = {"marginBottom": "0.9rem"}
    return dbc.Container([
        page_hero(
            "Scenario Predictor",
            "Test how timing, weather, and road context can shift the odds of a higher-severity accident.",
            "This panel is a presentation-friendly way to demonstrate the model in action without dropping into raw API requests.",
            [
                metric_chip("fa-flask", "What-if testing"),
                metric_chip("fa-tachometer-alt", "Probability gauge"),
                metric_chip("fa-cogs", "Best model or manual pick"),
            ],
            [("Inputs", "11 features"), ("Output", "Severity + confidence"), ("Flow", "Single click"), ("Mode", "Interactive demo")],
        ),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Input features"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Time and calendar", className="section-heading mb-3"),
                                dbc.Label("Hour of Day (0-23)"),
                                dbc.Input(id="p-hour", type="number", min=0, max=23, value=8, style=input_style),
                                dbc.Label("Day of Week (0 = Mon)"),
                                dbc.Input(id="p-dow", type="number", min=0, max=6, value=1, style=input_style),
                                dbc.Label("Month"),
                                dbc.Input(id="p-month", type="number", min=1, max=12, value=3, style=input_style),
                                dbc.Label("Year"),
                                dbc.Input(id="p-year", type="number", min=2016, max=2030, value=2023, style=input_style),
                                dbc.Label("Road Feature Count"),
                                dbc.Input(id="p-road", type="number", min=0, max=13, value=2, style=input_style),
                            ], md=6),
                            dbc.Col([
                                html.H6("Weather and exposure", className="section-heading mb-3"),
                                dbc.Label("Temperature (degF)"),
                                dbc.Input(id="p-temp", type="number", value=55.0, style=input_style),
                                dbc.Label("Humidity (%)"),
                                dbc.Input(id="p-humidity", type="number", min=0, max=100, value=70.0, style=input_style),
                                dbc.Label("Visibility (mi)"),
                                dbc.Input(id="p-visibility", type="number", min=0, value=10.0, style=input_style),
                                dbc.Label("Wind Speed (mph)"),
                                dbc.Input(id="p-wind", type="number", min=0, value=10.0, style=input_style),
                                dbc.Label("Precipitation (in)"),
                                dbc.Input(id="p-precip", type="number", min=0, value=0.0, style=input_style),
                                dbc.Label("Distance (mi)"),
                                dbc.Input(id="p-distance", type="number", min=0, value=0.5, style=input_style),
                            ], md=6),
                        ], className="g-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Model"),
                                dcc.Dropdown(
                                    id="p-model",
                                    options=[
                                        {"label": "Best (auto)", "value": ""},
                                        {"label": "XGBoost", "value": "xgboost"},
                                        {"label": "LightGBM", "value": "lightgbm"},
                                        {"label": "Random Forest", "value": "random_forest"},
                                        {"label": "Logistic Regression", "value": "logistic_regression"},
                                    ],
                                    value="",
                                    clearable=False,
                                    className="dash-dropdown",
                                ),
                            ], md=12),
                        ], className="mt-1"),
                        dbc.Button("Predict Severity", id="predict-btn", color="primary",
                                   className="mt-4 w-100", n_clicks=0),
                    ]),
                ], className="section-card border-0 h-100"),
            ], lg=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prediction result"),
                    dbc.CardBody(
                        dcc.Loading(
                            html.Div(id="prediction-result", children=prediction_placeholder(), className="prediction-panel"),
                            color="#67d2ff",
                        )
                    ),
                ], className="section-card border-0 h-100"),
            ], lg=6),
        ], className="g-4"),
    ], fluid=True, className="page-shell pb-5")


# -- App layout ----------------------------------------------------------------

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content"),
], className="app-shell")


@app.callback(
    Output("navbar-collapse", "is_open"),
    Input("navbar-toggler", "n_clicks"),
    State("navbar-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_navbar(n, is_open):
    return not is_open


@app.callback(
    Output("nav-links", "children"),
    Input("url", "pathname"),
)
def update_nav_active(pathname):
    pages = [
        ("/",        "fa-chart-bar", "Overview"),
        ("/eda",     "fa-search",    "EDA"),
        ("/models",  "fa-robot",     "ML Models"),
        ("/predict", "fa-bolt",      "Predict"),
    ]
    items = []
    for href, icon, label in pages:
        is_active = pathname == href
        style = {
            **NAV_LINK_STYLE,
            "color": "#edf5ff" if is_active else "#b7c5db",
            "background": "rgba(103,210,255,0.12)" if is_active else "transparent",
            "border": "1px solid rgba(103,210,255,0.24)" if is_active else "1px solid transparent",
        }
        items.append(dbc.NavItem(html.A(
            [html.I(className=f"fa {icon}"), f" {label}"],
            href=href, style=style, className="nav-link-custom",
        )))
    return items


# -- Routing -------------------------------------------------------------------

@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/eda":
        return eda_layout()
    elif pathname == "/models":
        return models_layout()
    elif pathname == "/predict":
        return predict_layout()
    return overview_layout()


@app.callback(Output("eda-selection-copy", "children"), Input("eda-selector", "value"))
def update_eda_copy(selection):
    copy = {
        "hour": "Follow how accident activity builds through the day and whether certain hours concentrate more severe cases.",
        "weather": "Compare environmental stress against both event volume and average severity to surface weather-linked risk.",
        "cities": "Use the city view to identify where accident concentration is highest and which states dominate the ranking.",
        "road": "Road-feature analysis helps connect infrastructure complexity to both accident volume and severity.",
    }
    return copy.get(selection, "")


# -- Overview callbacks --------------------------------------------------------

@app.callback(
    Output("stat-cards", "children"),
    Output("severity-dist-chart", "figure"),
    Output("accidents-over-time-chart", "figure"),
    Output("severity-state-map", "figure"),
    Input("overview-interval", "n_intervals"),
)
def update_overview(_):
    try:
        stats = requests.get(f"{API_BASE}/stats", timeout=30).json()
    except Exception:
        stats = {"total_accidents": 0, "by_severity": {}, "by_state": {},
                 "avg_temperature": 0, "avg_visibility": 0, "date_range": {}}

    date_range = stats.get("date_range", {})
    period_label = " | ".join(filter(None, [date_range.get("start"), date_range.get("end")])) or "Portfolio snapshot"
    state_count = len(stats.get("by_state", {})) or 0

    cards = [
        dbc.Col(
            stat_card("Total Accidents", f"{stats['total_accidents']:,}", "fa-car-crash", "danger", period_label),
            md=6,
            xl=3,
        ),
        dbc.Col(
            stat_card("Avg Temperature", f"{stats['avg_temperature']} degF", "fa-thermometer-half", "warning", "Environmental baseline"),
            md=6,
            xl=3,
        ),
        dbc.Col(
            stat_card("Avg Visibility", f"{stats['avg_visibility']} mi", "fa-eye", "info", "Exposure context"),
            md=6,
            xl=3,
        ),
        dbc.Col(
            stat_card("States in Summary", f"{state_count}", "fa-map", "success", "Top states from API stats"),
            md=6,
            xl=3,
        ),
    ]

    sev_data = stats.get("by_severity", {})
    sev_df = pd.DataFrame({"Severity": list(sev_data.keys()), "Count": list(sev_data.values())})
    if sev_df.empty:
        sev_fig = empty_figure("Accidents by Severity Level")
    else:
        sev_df["Severity"] = sev_df["Severity"].astype(str)
        sev_fig = px.bar(
            sev_df, x="Severity", y="Count", color="Severity",
            labels={"Severity": "Severity Level", "Count": "Accident Count"},
            title="Accidents by Severity Level",
            color_discrete_sequence=CATEGORY_SEQUENCE,
        )
        sev_fig.update_traces(marker_line_width=0, opacity=0.92)
        sev_fig = apply_figure_style(sev_fig)

    try:
        time_data = requests.get(f"{API_BASE}/eda/accidents-over-time", timeout=30).json()
        time_df = pd.DataFrame(time_data)
        if time_df.empty:
            time_fig = empty_figure("Accidents Over Time")
        else:
            time_fig = px.line(
                time_df,
                x="period",
                y="count",
                title="Accidents Over Time",
                markers=True,
                color_discrete_sequence=[CATEGORY_SEQUENCE[0]],
            )
            time_fig.update_traces(line={"width": 3}, marker={"size": 7})
            time_fig.update_xaxes(tickangle=45)
            time_fig = apply_figure_style(time_fig)
    except Exception:
        time_fig = empty_figure("Accidents Over Time")

    try:
        state_data = requests.get(f"{API_BASE}/eda/severity-by-state", timeout=30).json()
        state_df = pd.DataFrame(state_data)
        if state_df.empty:
            map_fig = empty_figure("Average Accident Severity by State")
        else:
            map_fig = px.choropleth(
                state_df, locations="state", locationmode="USA-states",
                color="avg_severity", scope="usa",
                color_continuous_scale="Tealgrn",
                title="Average Accident Severity by State",
                hover_data={"count": True, "avg_severity": ":.2f"},
            )
            map_fig.update_geos(
                bgcolor="rgba(0,0,0,0)",
                lakecolor="rgba(103, 210, 255, 0.08)",
                landcolor="rgba(15, 25, 42, 0.95)",
                subunitcolor="rgba(217, 242, 255, 0.1)",
                showlakes=True,
            )
            map_fig.update_layout(coloraxis_colorbar={"title": "Avg severity"})
            map_fig = apply_figure_style(map_fig, height=500)
    except Exception:
        map_fig = empty_figure("Average Accident Severity by State")

    return cards, sev_fig, time_fig, map_fig


# -- EDA callbacks -------------------------------------------------------------

@app.callback(
    Output("eda-main-chart", "figure"),
    Output("eda-secondary-chart", "figure"),
    Output("eda-heatmap-chart", "figure"),
    Input("eda-selector", "value"),
)
def update_eda(selection):
    empty = empty_figure("Analysis unavailable")

    if selection == "hour":
        try:
            data = requests.get(f"{API_BASE}/eda/severity-by-hour", timeout=30).json()
            df = pd.DataFrame(data)
            if df.empty:
                return empty, empty, empty
            main_fig = px.bar(
                df,
                x="hour",
                y="count",
                color="Severity" if "Severity" in df.columns else None,
                title="Accidents by Hour of Day",
                color_discrete_sequence=CATEGORY_SEQUENCE,
            )
            main_fig = apply_figure_style(main_fig, height=420)
            agg = df.groupby("hour", as_index=False)["count"].sum()
            sec_fig = px.area(
                agg,
                x="hour",
                y="count",
                title="Total Accidents by Hour",
                color_discrete_sequence=[CATEGORY_SEQUENCE[1]],
            )
            sec_fig = apply_figure_style(sec_fig)
            if "Severity" in df.columns:
                pivot = df.pivot_table(index="Severity", columns="hour", values="count", fill_value=0)
                heat_fig = px.imshow(
                    pivot,
                    title="Severity by Hour Heatmap",
                    color_continuous_scale="Tealgrn",
                    aspect="auto",
                )
                heat_fig = apply_figure_style(heat_fig)
            else:
                heat_fig = empty_figure("Severity by Hour Heatmap")
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    elif selection == "weather":
        try:
            data = requests.get(f"{API_BASE}/eda/weather-impact", timeout=30).json()
            df = pd.DataFrame(data)
            if df.empty:
                return empty, empty, empty
            main_fig = px.scatter(
                df,
                x="weather_score",
                y="avg_severity",
                size="count",
                title="Weather Score vs Average Severity",
                color="avg_severity",
                color_continuous_scale="Tealgrn",
            )
            main_fig.update_traces(marker={"line": {"width": 1, "color": "rgba(255,255,255,0.2)"}})
            main_fig = apply_figure_style(main_fig, height=420)
            sec_fig = px.bar(
                df,
                x="weather_score",
                y="count",
                title="Accident Count by Weather Score",
                color="weather_score",
                color_continuous_scale="Blues",
            )
            sec_fig = apply_figure_style(sec_fig)
            heat_fig = px.bar(
                df,
                x="weather_score",
                y="avg_severity",
                title="Average Severity by Weather Score",
                color="avg_severity",
                color_continuous_scale="Tealgrn",
            )
            heat_fig = apply_figure_style(heat_fig)
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    elif selection == "cities":
        try:
            data = requests.get(f"{API_BASE}/eda/top-cities?top_n=20", timeout=30).json()
            df = pd.DataFrame(data)
            if df.empty:
                return empty, empty, empty
            main_fig = px.bar(
                df,
                x="count",
                y="city_state",
                orientation="h",
                title="Top 20 Cities by Accident Count",
                color="count",
                color_continuous_scale="Blues",
            )
            main_fig.update_layout(yaxis={"categoryorder": "total ascending"})
            main_fig = apply_figure_style(main_fig, height=420)
            top10 = df.nlargest(10, "count")
            sec_fig = px.bar(
                top10,
                x="city_state",
                y="count",
                title="Top 10 Cities",
                color="count",
                color_continuous_scale="Tealgrn",
            )
            sec_fig.update_layout(xaxis_tickangle=30)
            sec_fig = apply_figure_style(sec_fig)
            df["state"] = df["city_state"].str.split(", ").str[-1]
            state_totals = df.groupby("state")["count"].sum().reset_index().nlargest(15, "count")
            heat_fig = px.bar(
                state_totals,
                x="state",
                y="count",
                title="Accidents by State from Top Cities",
                color="count",
                color_continuous_scale="Reds",
            )
            heat_fig = apply_figure_style(heat_fig)
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    elif selection == "road":
        try:
            data = requests.get(f"{API_BASE}/eda/road-features", timeout=30).json()
            df = pd.DataFrame(data)
            if df.empty:
                return empty, empty, empty
            df["feature_count"] = df["feature_count"].astype(str)
            main_fig = px.bar(
                df, x="feature_count", y="accident_count",
                title="Accidents by Road Feature Count",
                color="avg_severity", color_continuous_scale="Tealgrn",
                labels={"feature_count": "Road Features Present", "accident_count": "Accident Count"},
            )
            main_fig = apply_figure_style(main_fig, height=420)
            sec_fig = px.bar(
                df, x="feature_count", y="avg_severity",
                title="Average Severity by Road Feature Count",
                color="avg_severity", color_continuous_scale="Tealgrn",
                labels={"feature_count": "Road Features Present", "avg_severity": "Average Severity"},
            )
            sec_fig = apply_figure_style(sec_fig)
            heat_fig = px.scatter(
                df, x="accident_count", y="avg_severity",
                text="feature_count", size="accident_count",
                title="Accident Volume vs Average Severity",
                color="avg_severity", color_continuous_scale="Tealgrn",
                labels={"accident_count": "Accident Count", "avg_severity": "Average Severity"},
            )
            heat_fig.update_traces(textposition="top center")
            heat_fig = apply_figure_style(heat_fig)
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    return empty, empty, empty


# -- Models callbacks ----------------------------------------------------------

@app.callback(
    Output("metrics-table-container", "children"),
    Output("metrics-bar-chart", "figure"),
    Output("metrics-radar-chart", "figure"),
    Input("models-interval", "n_intervals"),
)
def update_models(_):
    try:
        data = requests.get(f"{API_BASE}/metrics", timeout=30).json()
        df = pd.DataFrame(data)
    except Exception:
        warning = dbc.Alert("Model metrics unavailable. Run evaluate_model.py first.", color="warning")
        return warning, empty_figure("Model Metrics Comparison"), empty_figure("Model Radar Chart")

    if df.empty:
        warning = dbc.Alert("No evaluation records were returned by the API.", color="warning")
        return warning, empty_figure("Model Metrics Comparison"), empty_figure("Model Radar Chart")

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
        style_table={"overflowX": "auto", "backgroundColor": "transparent"},
        style_cell={
            "backgroundColor": "rgba(8, 17, 30, 0.82)",
            "color": "#edf5ff",
            "textAlign": "center",
            "padding": "12px",
            "border": "1px solid rgba(143, 168, 201, 0.14)",
            "fontFamily": FONT_STACK,
        },
        style_header={
            "backgroundColor": "rgba(15, 27, 44, 0.96)",
            "fontWeight": "700",
            "color": "#edf5ff",
            "border": "1px solid rgba(143, 168, 201, 0.18)",
        },
        style_data_conditional=[
            {"if": {"filter_query": f"{{test_auc}} = {df['test_auc'].max()}"},
             "backgroundColor": "rgba(124, 243, 188, 0.12)", "color": "#edfef3"},
        ],
        page_size=10,
    )

    metric_cols = [col for col in ["test_auc", "test_f1", "test_precision", "test_recall"] if col in df.columns]
    bar_fig = px.bar(
        df.melt(id_vars="model_name", value_vars=metric_cols),
        x="model_name", y="value", color="variable", barmode="group",
        title="Model Metrics Comparison",
        color_discrete_sequence=CATEGORY_SEQUENCE,
    )
    bar_fig = apply_figure_style(bar_fig, height=380)

    metrics = [col for col in ["test_auc", "test_f1", "test_precision", "test_recall", "test_accuracy"] if col in df.columns]
    radar_fig = go.Figure()
    for idx, (_, row) in enumerate(df.iterrows()):
        radar_fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=[m.replace("test_", "").upper() for m in metrics],
            fill="toself",
            name=row["model_name"],
            line={"color": CATEGORY_SEQUENCE[idx % len(CATEGORY_SEQUENCE)]},
        ))
    radar_fig.update_layout(title="Model Radar Chart", polar={"radialaxis": {"range": [0, 1], "showgrid": True}})
    radar_fig = apply_figure_style(radar_fig, height=380)

    return table, bar_fig, radar_fig


# -- Prediction callback -------------------------------------------------------

@app.callback(
    Output("prediction-result", "children"),
    Input("predict-btn", "n_clicks"),
    State("p-hour", "value"), State("p-dow", "value"),
    State("p-month", "value"), State("p-year", "value"),
    State("p-temp", "value"), State("p-humidity", "value"),
    State("p-visibility", "value"), State("p-wind", "value"),
    State("p-precip", "value"), State("p-distance", "value"),
    State("p-road", "value"), State("p-model", "value"),
    prevent_initial_call=True,
)
def make_prediction(n_clicks, hour, dow, month, year, temp, humidity,
                    visibility, wind, precip, distance, road, model_name):
    is_weekend = 1 if dow >= 5 else 0
    is_rush = 1 if hour in [7, 8, 9, 16, 17, 18] else 0
    is_night = 1 if hour < 6 or hour >= 21 else 0
    season = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}.get(month, 1)

    weather_score = 0
    if temp and (temp < 32 or temp > 95):
        weather_score += 1
    if visibility and visibility < 1.0:
        weather_score += 2
    elif visibility and visibility < 3.0:
        weather_score += 1
    if wind and wind > 30:
        weather_score += 1
    if precip and precip > 0.1:
        weather_score += 1

    payload = {
        "hour": hour, "day_of_week": dow, "month": month, "year": year,
        "is_weekend": is_weekend, "is_rush_hour": is_rush, "is_night": is_night,
        "season": season, "duration_min": 30.0,
        "weather_severity_score": min(weather_score, 5),
        "road_feature_count": road or 0,
        "Distance(mi)": distance or 0.5,
        "Temperature(F)": temp or 55.0,
        "Humidity(%)": humidity or 70.0,
        "Pressure(in)": 29.9,
        "Visibility(mi)": visibility or 10.0,
        "Wind_Speed(mph)": wind or 10.0,
        "Precipitation(in)": precip or 0.0,
        "model_name": model_name or None,
    }

    try:
        resp = requests.post(f"{API_BASE}/predict", json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        color = "danger" if result["prediction"] == 1 else "success"
        label = result["severity_label"]
        prob = result["probability"]
        model_used = result["model"]

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "High Severity Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#ff8a80" if prob > 0.5 else "#7cf3bc"},
                "steps": [
                    {"range": [0, 35], "color": "rgba(124, 243, 188, 0.18)"},
                    {"range": [35, 65], "color": "rgba(255, 207, 107, 0.18)"},
                    {"range": [65, 100], "color": "rgba(255, 138, 128, 0.18)"},
                ],
                "threshold": {"line": {"color": "#edf5ff", "width": 4}, "value": 50},
            },
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=300, margin={"t": 55, "b": 10, "l": 15, "r": 15})

        return html.Div([
            dbc.Alert(f"Predicted Severity: {label}", color=color, className="fs-4 fw-bold mb-3"),
            html.Div([
                html.Span(f"Probability: {prob:.1%}", className="result-chip"),
                html.Span(f"Model used: {model_used}", className="result-chip"),
                html.Span(f"Weather score: {min(weather_score, 5)}", className="result-chip"),
            ], className="result-chip-row mb-3"),
            dcc.Graph(figure=gauge, config=GRAPH_CONFIG),
        ])
    except Exception as e:
        return dbc.Alert(f"Prediction failed: {e}. Ensure the API is running.", color="danger")


if __name__ == "__main__":
    app.run(
        host=cfg["dashboard"]["host"],
        port=cfg["dashboard"]["port"],
        debug=cfg["dashboard"]["debug"],
    )



