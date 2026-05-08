import dash_bootstrap_components as dbc
from dash import html, dcc

def landing_layout(stat_card_func):
    return html.Div([
        # Hero Section
        dbc.Container([
            html.Div([
                html.Div("EXPLORATORY DATA ANALYSIS PORTFOLIO", className="hero-eyebrow mb-3"),
                html.H1("US Accidents Portfolio", className="landing-hero-title"),
                html.P(
                    "Analyzing 7.7 million traffic incidents across the United States. "
                    "Uncover the hidden patterns behind road safety, weather impacts, and urban risk factors.",
                    className="landing-hero-subtitle"
                ),
                html.Div([
                    dbc.Button("Launch Dashboard", href="/login", color="primary", size="lg", className="px-5 py-3 me-3"),
                    dbc.Button("View Source Code", href="https://github.com/Kubojah-Dan/EDA_Portfolio", color="secondary", outline=True, size="lg", className="px-5 py-3"),
                ], className="d-flex justify-content-center flex-wrap gap-3"),
            ], className="landing-hero"),
        ], fluid=True),

        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(
    src="/assets/image.jpg",
    className="img-fluid rounded-4 shadow-lg border border-secondary me-3",
    style={"borderWidth": "3px", "opacity": "0.9"},
    alt="Dashboard Preview Placeholder",
),

html.Img(
    src="/assets/image2.jpg",
    className="img-fluid rounded-4 shadow-lg border border-secondary",
    style={"borderWidth": "3px", "opacity": "0.9"},
    alt="Dashboard Preview Placeholder",
),
                        html.Div([
                            html.H4("Real-time Visual Analytics", className="text-white mb-2"),
                            html.P("Our interactive choropleth maps and time-series charts provide instant geographic insights.", className="text-muted small"),
                        ], className="p-4 bg-dark rounded-bottom-4 border-top border-secondary")
                    ], className="landing-image-container position-relative overflow-hidden rounded-4")
                ], lg=10, className="mx-auto text-center")
            ], className="mb-5")
        ], fluid=True),

        # Features Section
        dbc.Container([
            html.Div([
                html.H2("System Capabilities", className="text-center mb-5 fw-bold"),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Div(html.I(className="fa fa-chart-line"), className="feature-icon"),
                            html.H3("Temporal Trends", className="section-heading"),
                            html.P("Identify high-risk time windows, from hourly rush periods to seasonal variations across years.", className="section-description"),
                        ], className="feature-card")
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.Div(html.I(className="fa fa-cloud-sun"), className="feature-icon"),
                            html.H3("Weather Influence", className="section-heading"),
                            html.P("Correlating atmospheric conditions—visibility, precipitation, and humidity—with incident frequency.", className="section-description"),
                        ], className="feature-card")
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.Div(html.I(className="fa fa-map-marked-alt"), className="feature-icon"),
                            html.H3("Geographic Hotspots", className="section-heading"),
                            html.P("State-level analysis highlighting areas with significant Class 3 and 4 severity outbreaks.", className="section-description"),
                        ], className="feature-card")
                    ], md=4),
                ], className="g-4"),
            ], className="landing-section"),
            
            # Impact Stats
            html.Div([
                dbc.Row([
                    dbc.Col([
                        stat_card_func("Total Incidents", "7,728,394", "fa-car-crash", "primary", "Verified Records")
                    ], md=4),
                    dbc.Col([
                        stat_card_func("Time Span", "2016 - 2023", "fa-calendar-alt", "success", "Continuous Monitoring")
                    ], md=4),
                    dbc.Col([
                        stat_card_func("ML Accuracy", "94%+", "fa-brain", "warning", "Severity Classification")
                    ], md=4),
                ], className="g-4 mb-5"),
            ]),
        ], fluid=True),

        # Footer
        html.Footer([
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.P("© 2026 US Accidents EDA Portfolio. Built with Dash, FastAPI, and XGBoost.", className="text-muted small mb-0")
                    ], md=6, className="text-center text-md-start"),
                    dbc.Col([
                        html.A("Project GitHub", href="#", className="text-muted small text-decoration-none me-3"),
                        html.A("API Documentation", href="/docs", className="text-muted small text-decoration-none"),
                    ], md=6, className="text-center text-md-end")
                ], className="py-4 border-top border-secondary mt-5")
            ], fluid=True)
        ])
    ], className="landing-shell")


def auth_layout():
    return html.Div([
        dbc.Container([
            html.Div([
                # Sign Up Container
                html.Div([
                    html.Div([
                        html.H2("Create Account", className="auth-form-title"),
                        dbc.Input(id="signup-user", placeholder="Username", type="text", className="mb-3"),
                        html.Div([
                            dbc.Input(id="signup-pass", placeholder="Password", type="password", className="pe-5"),
                            html.Button(html.I(id="signup-pass-icon", className="fa fa-eye"), 
                                        id="signup-pass-toggle", 
                                        className="btn btn-link position-absolute end-0 top-50 translate-middle-y text-muted text-decoration-none pe-3",
                                        style={"zIndex": 10})
                        ], className="position-relative mb-3"),
                        dbc.Button("Sign Up", id="signup-btn", color="primary", className="w-100 mt-2"),
                        html.Div(id="signup-alert", className="mt-3")
                    ], className="d-flex flex-column justify-content-center h-100")
                ], className="form-container sign-up-container"),
                
                # Sign In Container
                html.Div([
                    html.Div([
                        html.H2("Welcome Back", className="auth-form-title"),
                        dbc.Input(id="login-user", placeholder="Username", type="text", className="mb-3"),
                        html.Div([
                            dbc.Input(id="login-pass", placeholder="Password", type="password", className="pe-5"),
                            html.Button(html.I(id="login-pass-icon", className="fa fa-eye"), 
                                        id="login-pass-toggle", 
                                        className="btn btn-link position-absolute end-0 top-50 translate-middle-y text-muted text-decoration-none pe-3",
                                        style={"zIndex": 10})
                        ], className="position-relative mb-3"),
                        dbc.Button("Sign In", id="login-btn", color="primary", className="w-100 mt-2"),
                        html.Div(id="login-alert", className="mt-3")
                    ], className="d-flex flex-column justify-content-center h-100")
                ], className="form-container sign-in-container"),
                
                # Overlay Container
                html.Div([
                    html.Div([
                        html.Div([
                            html.H1("Welcome Back!"),
                            html.P("To keep connected with us please login with your personal info", className="mb-4"),
                            html.Button("Sign In", id="toggle-signin", className="auth-btn-ghost")
                        ], className="overlay-panel overlay-left"),
                        html.Div([
                            html.H1("Hello, Friend!"),
                            html.P("Enter your details and start your journey with us", className="mb-4"),
                            html.Button("Sign Up", id="toggle-signup", className="auth-btn-ghost")
                        ], className="overlay-panel overlay-right")
                    ], className="overlay")
                ], className="overlay-container")
            ], id="auth-container", className="auth-container")
        ], className="auth-shell")
    ])
