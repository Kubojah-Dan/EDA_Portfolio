"""
dashboard.py
Multi-page Dash dashboard for the US Accidents EDA Portfolio.
Pages: Overview, EDA, ML Models, Prediction Tool
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc

from src.utils.helpers import load_config

cfg = load_config()
API_BASE = cfg["dashboard"]["api_base_url"]

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, dbc.icons.FONT_AWESOME],
    suppress_callback_exceptions=True,
    title="US Accidents EDA Portfolio",
)
server = app.server  # expose Flask server for production

# -- Layout helpers ------------------------------------------------------------

def stat_card(title, value, icon, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"fa {icon} fa-2x me-3", style={"color": f"var(--bs-{color})"}),
                html.Div([
                    html.H4(value, className="mb-0 fw-bold"),
                    html.Small(title, className="text-muted"),
                ]),
            ], className="d-flex align-items-center"),
        ]),
        className="shadow-sm h-100",
    )


NAV_STYLE = {
    "background": "linear-gradient(90deg, #0d1117 0%, #161b22 60%, #1a1f2e 100%)",
    "borderBottom": "1px solid #30363d",
    "padding": "0 1.5rem",
    "boxShadow": "0 2px 12px rgba(0,0,0,0.5)",
}
BRAND_STYLE = {
    "fontWeight": "700",
    "fontSize": "1.1rem",
    "letterSpacing": "0.03em",
    "color": "#e6edf3",
    "textDecoration": "none",
    "display": "flex",
    "alignItems": "center",
    "gap": "10px",
}
NAV_LINK_STYLE = {
    "color": "#8b949e",
    "fontWeight": "500",
    "fontSize": "0.9rem",
    "padding": "0.6rem 1rem",
    "borderRadius": "6px",
    "transition": "all 0.2s",
    "display": "flex",
    "alignItems": "center",
    "gap": "6px",
    "textDecoration": "none",
}

navbar = dbc.Navbar(
    dbc.Container([
        html.A(
            html.Div([
                html.I(className="fa fa-car-crash", style={"color": "#f85149", "fontSize": "1.3rem"}),
                html.Span("US Accidents", style={"color": "#e6edf3"}),
                html.Span(" EDA Portfolio", style={"color": "#58a6ff"}),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            href="/", style=BRAND_STYLE,
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
        dbc.Collapse(
            dbc.Nav(id="nav-links", navbar=True, className="ms-auto gap-1"),
            id="navbar-collapse", navbar=True,
        ),
    ], fluid=True),
    style=NAV_STYLE,
    className="mb-4",
    dark=True,
)

# -- Pages ---------------------------------------------------------------------

def overview_layout():
    return dbc.Container([
        dbc.Row([
            dbc.Col(html.H2("Dataset Overview", className="mb-4")),
        ]),
        dbc.Row(id="stat-cards", className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="severity-dist-chart"), md=6),
            dbc.Col(dcc.Graph(id="accidents-over-time-chart"), md=6),
        ], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="severity-state-map"), md=12),
        ]),
        dcc.Interval(id="overview-interval", interval=60_000, n_intervals=0),
    ], fluid=True)


def eda_layout():
    return dbc.Container([
        dbc.Row([dbc.Col(html.H2("Exploratory Data Analysis", className="mb-4"))]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Select Analysis"),
                    dbc.CardBody(
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
                        )
                    ),
                ], className="mb-3"),
            ], md=4),
        ]),
        dbc.Row([dbc.Col(dcc.Graph(id="eda-main-chart"), md=12)]),
        dbc.Row([
            dbc.Col(dcc.Graph(id="eda-secondary-chart"), md=6),
            dbc.Col(dcc.Graph(id="eda-heatmap-chart"), md=6),
        ], className="mt-4"),
    ], fluid=True)


def models_layout():
    return dbc.Container([
        dbc.Row([dbc.Col(html.H2("ML Model Performance", className="mb-4"))]),
        dbc.Row([dbc.Col(html.Div(id="metrics-table-container"), md=12)], className="mb-4"),
        dbc.Row([
            dbc.Col(dcc.Graph(id="metrics-bar-chart"), md=6),
            dbc.Col(dcc.Graph(id="metrics-radar-chart"), md=6),
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Model Comparison Notes"),
                    dbc.CardBody([
                        html.P("- XGBoost & LightGBM typically achieve highest AUC on tabular data."),
                        html.P("- Logistic Regression provides a strong interpretable baseline."),
                        html.P("- Random Forest is robust to outliers and handles non-linearity well."),
                        html.P("- All models trained with class-imbalance handling (scale_pos_weight / class_weight)."),
                    ]),
                ]), md=12
            ),
        ], className="mt-4"),
        dcc.Interval(id="models-interval", interval=300_000, n_intervals=0),
    ], fluid=True)


def predict_layout():
    input_style = {"marginBottom": "8px"}
    return dbc.Container([
        dbc.Row([dbc.Col(html.H2("Accident Severity Predictor", className="mb-4"))]),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Input Features"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Hour of Day (0-23)"),
                                dbc.Input(id="p-hour", type="number", min=0, max=23, value=8, style=input_style),
                                dbc.Label("Day of Week (0=Mon)"),
                                dbc.Input(id="p-dow", type="number", min=0, max=6, value=1, style=input_style),
                                dbc.Label("Month"),
                                dbc.Input(id="p-month", type="number", min=1, max=12, value=3, style=input_style),
                                dbc.Label("Year"),
                                dbc.Input(id="p-year", type="number", min=2016, max=2030, value=2023, style=input_style),
                                dbc.Label("Temperature ( degF)"),
                                dbc.Input(id="p-temp", type="number", value=55.0, style=input_style),
                            ], md=6),
                            dbc.Col([
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
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Road Feature Count"),
                                dbc.Input(id="p-road", type="number", min=0, max=13, value=2, style=input_style),
                            ], md=6),
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
                                ),
                            ], md=6),
                        ], className="mt-2"),
                        dbc.Button("Predict Severity", id="predict-btn", color="primary",
                                   className="mt-3 w-100", n_clicks=0),
                    ]),
                ]),
            ], md=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prediction Result"),
                    dbc.CardBody(html.Div(id="prediction-result", className="text-center py-4")),
                ], className="h-100"),
            ], md=6),
        ]),
    ], fluid=True)


# -- App layout ----------------------------------------------------------------

app.index_string = app.index_string.replace(
    "</head>",
    "<style>.nav-link-custom:hover{color:#e6edf3 !important;background:rgba(177,186,196,0.12) !important;}</style></head>"
)

app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    navbar,
    html.Div(id="page-content"),
])


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
            "color": "#e6edf3" if is_active else "#8b949e",
            "background": "rgba(31,111,235,0.15)" if is_active else "transparent",
            "border": "1px solid #1f6feb" if is_active else "1px solid transparent",
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

    cards = dbc.Row([
        dbc.Col(stat_card("Total Accidents", f"{stats['total_accidents']:,}", "fa-car-crash", "danger"), md=3),
        dbc.Col(stat_card("Avg Temperature", f"{stats['avg_temperature']} degF", "fa-thermometer-half", "warning"), md=3),
        dbc.Col(stat_card("Avg Visibility", f"{stats['avg_visibility']} mi", "fa-eye", "info"), md=3),
        dbc.Col(stat_card("States Covered", "49", "fa-map", "success"), md=3),
    ], className="g-3")

    # Severity distribution — must use DataFrame, not raw lists
    sev_data = stats.get("by_severity", {})
    sev_df = pd.DataFrame({"Severity": list(sev_data.keys()), "Count": list(sev_data.values())})
    sev_df["Severity"] = sev_df["Severity"].astype(str)
    sev_fig = px.bar(
        sev_df, x="Severity", y="Count", color="Severity",
        labels={"Severity": "Severity Level", "Count": "Accident Count"},
        title="Accidents by Severity Level",
        color_discrete_sequence=px.colors.diverging.RdYlGn[::-1],
        template="plotly_dark",
    )

    # Accidents over time
    try:
        time_data = requests.get(f"{API_BASE}/eda/accidents-over-time", timeout=30).json()
        time_df = pd.DataFrame(time_data)
        time_fig = px.line(time_df, x="period", y="count", title="Accidents Over Time",
                           template="plotly_dark", markers=True)
        time_fig.update_xaxes(tickangle=45)
    except Exception:
        time_fig = go.Figure().update_layout(title="Accidents Over Time (data unavailable)",
                                             template="plotly_dark")

    # State severity map
    try:
        state_data = requests.get(f"{API_BASE}/eda/severity-by-state", timeout=30).json()
        state_df = pd.DataFrame(state_data)
        map_fig = px.choropleth(
            state_df, locations="state", locationmode="USA-states",
            color="avg_severity", scope="usa",
            color_continuous_scale="RdYlGn_r",
            title="Average Accident Severity by State",
            template="plotly_dark",
        )
    except Exception:
        map_fig = go.Figure().update_layout(title="State Map (data unavailable)", template="plotly_dark")

    return cards, sev_fig, time_fig, map_fig


# -- EDA callbacks -------------------------------------------------------------

@app.callback(
    Output("eda-main-chart", "figure"),
    Output("eda-secondary-chart", "figure"),
    Output("eda-heatmap-chart", "figure"),
    Input("eda-selector", "value"),
)
def update_eda(selection):
    empty = go.Figure().update_layout(template="plotly_dark")

    if selection == "hour":
        try:
            data = requests.get(f"{API_BASE}/eda/severity-by-hour", timeout=30).json()
            df = pd.DataFrame(data)
            main_fig = px.bar(df, x="hour", y="count",
                              color="Severity" if "Severity" in df.columns else None,
                              title="Accidents by Hour of Day", template="plotly_dark",
                              color_continuous_scale="RdYlGn_r")
            # Secondary: total accidents per hour (aggregated)
            agg = df.groupby("hour")["count"].sum().reset_index()
            sec_fig = px.area(agg, x="hour", y="count",
                              title="Total Accidents by Hour (All Severities)",
                              template="plotly_dark")
            # Heatmap: severity distribution across hours
            if "Severity" in df.columns:
                pivot = df.pivot_table(index="Severity", columns="hour", values="count", fill_value=0)
                heat_fig = px.imshow(pivot, title="Severity × Hour Heatmap",
                                     template="plotly_dark", color_continuous_scale="RdYlGn_r",
                                     aspect="auto")
            else:
                heat_fig = empty
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    elif selection == "weather":
        try:
            data = requests.get(f"{API_BASE}/eda/weather-impact", timeout=30).json()
            df = pd.DataFrame(data)
            main_fig = px.scatter(df, x="weather_score", y="avg_severity", size="count",
                                  title="Weather Score vs Avg Severity",
                                  template="plotly_dark", color="avg_severity",
                                  color_continuous_scale="RdYlGn_r")
            sec_fig = px.bar(df, x="weather_score", y="count",
                             title="Accident Count by Weather Score",
                             template="plotly_dark", color="weather_score",
                             color_continuous_scale="Blues")
            heat_fig = px.bar(df, x="weather_score", y="avg_severity",
                              title="Avg Severity by Weather Score",
                              template="plotly_dark", color="avg_severity",
                              color_continuous_scale="RdYlGn_r")
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    elif selection == "cities":
        try:
            data = requests.get(f"{API_BASE}/eda/top-cities?top_n=20", timeout=30).json()
            df = pd.DataFrame(data)
            main_fig = px.bar(df, x="count", y="city_state", orientation="h",
                              title="Top 20 Cities by Accident Count", template="plotly_dark")
            main_fig.update_layout(yaxis={"categoryorder": "total ascending"})
            # Secondary: top 10 only, vertical
            top10 = df.nlargest(10, "count")
            sec_fig = px.bar(top10, x="city_state", y="count",
                             title="Top 10 Cities", template="plotly_dark",
                             color="count", color_continuous_scale="Blues")
            sec_fig.update_layout(xaxis_tickangle=30)
            # Heatmap: extract state from city_state and show state totals
            df["state"] = df["city_state"].str.split(", ").str[-1]
            state_totals = df.groupby("state")["count"].sum().reset_index().nlargest(15, "count")
            heat_fig = px.bar(state_totals, x="state", y="count",
                              title="Accidents by State (Top Cities)",
                              template="plotly_dark", color="count",
                              color_continuous_scale="Reds")
        except Exception:
            main_fig, sec_fig, heat_fig = empty, empty, empty
        return main_fig, sec_fig, heat_fig

    elif selection == "road":
        try:
            data = requests.get(f"{API_BASE}/eda/road-features", timeout=30).json()
            df = pd.DataFrame(data)
            df["feature_count"] = df["feature_count"].astype(str)
            main_fig = px.bar(
                df, x="feature_count", y="accident_count",
                title="Accidents by Road Feature Count",
                template="plotly_dark",
                color="avg_severity", color_continuous_scale="RdYlGn_r",
                labels={"feature_count": "# Road Features Present", "accident_count": "Accident Count"},
            )
            sec_fig = px.bar(
                df, x="feature_count", y="avg_severity",
                title="Avg Severity by Road Feature Count",
                template="plotly_dark",
                color="avg_severity", color_continuous_scale="RdYlGn_r",
                labels={"feature_count": "# Road Features Present", "avg_severity": "Avg Severity"},
            )
            heat_fig = px.scatter(
                df, x="accident_count", y="avg_severity",
                text="feature_count", size="accident_count",
                title="Accident Volume vs Avg Severity",
                template="plotly_dark",
                color="avg_severity", color_continuous_scale="RdYlGn_r",
                labels={"accident_count": "Accident Count", "avg_severity": "Avg Severity"},
            )
            heat_fig.update_traces(textposition="top center")
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
        return html.P("Model metrics unavailable. Run evaluate_model.py first.", className="text-warning"), \
               go.Figure().update_layout(template="plotly_dark"), \
               go.Figure().update_layout(template="plotly_dark")

    table = dash_table.DataTable(
        data=df.to_dict("records"),
        columns=[{"name": c.replace("_", " ").title(), "id": c} for c in df.columns],
        style_table={"overflowX": "auto"},
        style_cell={"backgroundColor": "#2d2d2d", "color": "white", "textAlign": "center"},
        style_header={"backgroundColor": "#1a1a1a", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"filter_query": f"{{test_auc}} = {df['test_auc'].max()}"},
             "backgroundColor": "#1a472a", "color": "white"},
        ],
    )

    bar_fig = px.bar(
        df.melt(id_vars="model_name", value_vars=["test_auc", "test_f1", "test_precision", "test_recall"]),
        x="model_name", y="value", color="variable", barmode="group",
        title="Model Metrics Comparison", template="plotly_dark",
    )

    metrics = ["test_auc", "test_f1", "test_precision", "test_recall", "test_accuracy"]
    radar_fig = go.Figure()
    for _, row in df.iterrows():
        radar_fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=[m.replace("test_", "").upper() for m in metrics],
            fill="toself", name=row["model_name"],
        ))
    radar_fig.update_layout(title="Model Radar Chart", template="plotly_dark",
                            polar={"radialaxis": {"range": [0, 1]}})

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
                "bar": {"color": "red" if prob > 0.5 else "green"},
                "steps": [
                    {"range": [0, 50], "color": "#1a472a"},
                    {"range": [50, 100], "color": "#7b1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 4}, "value": 50},
            },
        ))
        gauge.update_layout(template="plotly_dark", height=300, margin={"t": 50, "b": 0})

        return html.Div([
            dbc.Alert(f"Predicted Severity: {label}", color=color, className="fs-4 fw-bold"),
            html.P(f"Probability: {prob:.1%}", className="fs-5"),
            html.P(f"Model: {model_used}", className="text-muted"),
            dcc.Graph(figure=gauge),
        ])
    except Exception as e:
        return dbc.Alert(f"Prediction failed: {e}. Ensure the API is running.", color="danger")


if __name__ == "__main__":
    app.run(
        host=cfg["dashboard"]["host"],
        port=cfg["dashboard"]["port"],
        debug=cfg["dashboard"]["debug"],
    )



