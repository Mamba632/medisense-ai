import streamlit as st


def inject_css():
    st.markdown(
        """
        <style>
        /* Healthcare color palette */
        :root {
            --primary: #0077B6;
            --secondary: #00B4D8;
            --accent: #90E0EF;
            --bg-light: #F8F9FA;
            --danger: #E63946;
            --success: #2D6A4F;
            --warning: #F4A261;
            --text-dark: #1B1B1B;
            --text-muted: #6C757D;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(135deg, #0077B6, #00B4D8);
            padding: 2.5rem 2rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
        }
        .hero-banner h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            color: white !important;
        }
        .hero-banner p {
            font-size: 1.1rem;
            opacity: 0.9;
            color: white !important;
        }

        /* Metric cards */
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 1.2rem;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 4px solid var(--primary);
            margin-bottom: 1rem;
        }
        .metric-card.success { border-left-color: var(--success); }
        .metric-card.warning { border-left-color: var(--warning); }
        .metric-card.danger { border-left-color: var(--danger); }

        .metric-card h3 {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.3rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric-card .value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-dark);
        }

        /* Feature cards */
        .feature-card {
            background: white;
            border-radius: 10px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-top: 3px solid var(--primary);
            height: 100%;
        }
        .feature-card .icon {
            font-size: 2.5rem;
            margin-bottom: 0.8rem;
        }
        .feature-card h4 {
            color: var(--primary);
            margin-bottom: 0.5rem;
        }
        .feature-card p {
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        /* Section headers */
        .section-header {
            color: var(--primary);
            font-size: 1.4rem;
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid var(--accent);
        }

        /* Risk badges */
        .risk-badge {
            display: inline-block;
            padding: 0.4rem 1.2rem;
            border-radius: 20px;
            font-weight: 600;
            font-size: 1.1rem;
        }
        .risk-badge.low { background: #D4EDDA; color: #155724; }
        .risk-badge.medium { background: #FFF3CD; color: #856404; }
        .risk-badge.high { background: #F8D7DA; color: #721C24; }

        /* Info box */
        .info-box {
            background: #E8F4FD;
            border-radius: 8px;
            padding: 1rem 1.2rem;
            border-left: 4px solid var(--primary);
            margin: 1rem 0;
        }

        /* Footer */
        .app-footer {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.8rem;
            padding: 1.5rem 0;
            margin-top: 2rem;
            border-top: 1px solid #E9ECEF;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F8F9FA, #FFFFFF);
        }

        /* Hide Streamlit default menu and footer */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title, value, card_type=""):
    cls = f"metric-card {card_type}" if card_type else "metric-card"
    return f"""
    <div class="{cls}">
        <h3>{title}</h3>
        <div class="value">{value}</div>
    </div>
    """


def feature_card(icon, title, description):
    return f"""
    <div class="feature-card">
        <div class="icon">{icon}</div>
        <h4>{title}</h4>
        <p>{description}</p>
    </div>
    """


def risk_badge(label):
    level = label.lower()
    return f'<span class="risk-badge {level}">{label} Risk</span>'


def section_header(text):
    return f'<div class="section-header">{text}</div>'
