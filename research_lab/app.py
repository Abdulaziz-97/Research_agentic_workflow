"""
Research Lab - Multi-Agent Academic Research Platform
Main Streamlit Application Entry Point
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ui.pages.home import render_home_page
from ui.pages.team_setup import render_team_setup_page
from ui.pages.research_session import render_research_session_page
from config.settings import settings
from config.key_manager import initialize_key_manager


def inject_custom_css():
    """Inject professional research lab CSS."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Crimson+Pro:wght@400;500;600&display=swap');
    
    /* ============================================
       ROOT VARIABLES - Research Lab Theme
    ============================================ */
    :root {
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-tertiary: #1a1a24;
        --bg-card: #16161f;
        --bg-card-hover: #1c1c28;
        
        --accent-primary: #00d4aa;
        --accent-secondary: #0ea5e9;
        --accent-tertiary: #8b5cf6;
        --accent-warm: #f59e0b;
        --accent-danger: #ef4444;
        
        --text-primary: #f0f0f5;
        --text-secondary: #a0a0b0;
        --text-muted: #606070;
        
        --border-subtle: rgba(255,255,255,0.06);
        --border-medium: rgba(255,255,255,0.1);
        
        --font-sans: 'IBM Plex Sans', -apple-system, sans-serif;
        --font-mono: 'IBM Plex Mono', 'Fira Code', monospace;
        --font-serif: 'Crimson Pro', Georgia, serif;
        
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
        --shadow-md: 0 4px 20px rgba(0,0,0,0.4);
        --shadow-lg: 0 8px 40px rgba(0,0,0,0.5);
        --shadow-glow: 0 0 40px rgba(0,212,170,0.15);
    }
    
    /* ============================================
       GLOBAL STYLES
    ============================================ */
    .stApp {
        background: var(--bg-primary);
        font-family: var(--font-sans);
    }
    
    .stApp > header {
        background: transparent !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, .stDeployButton {
        display: none !important;
    }
    
    /* Main container */
    .main .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* ============================================
       TYPOGRAPHY
    ============================================ */
    h1, h2, h3 {
        font-family: var(--font-sans);
        font-weight: 600;
        color: var(--text-primary);
    }
    
    h1 { font-size: 2.5rem; letter-spacing: -0.02em; }
    h2 { font-size: 1.75rem; letter-spacing: -0.01em; }
    h3 { font-size: 1.25rem; }
    
    p, li {
        color: var(--text-secondary);
        line-height: 1.7;
    }
    
    /* ============================================
       SIDEBAR STYLES
    ============================================ */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* ============================================
       CUSTOM CARDS & CONTAINERS
    ============================================ */
    .research-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }
    
    .research-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-medium);
        box-shadow: var(--shadow-md);
    }
    
    .research-card-accent {
        background: linear-gradient(135deg, var(--bg-card) 0%, rgba(0,212,170,0.05) 100%);
        border-left: 3px solid var(--accent-primary);
    }
    
    .stats-card {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    
    .stats-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent-primary);
        font-family: var(--font-mono);
    }
    
    .stats-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-muted);
        margin-top: 0.25rem;
    }
    
    /* ============================================
       HERO SECTION
    ============================================ */
    .hero-container {
        background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    
    .hero-container::before {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 50%;
        height: 100%;
        background: radial-gradient(ellipse at top right, rgba(0,212,170,0.08) 0%, transparent 60%);
        pointer-events: none;
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 50%, var(--accent-tertiary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        max-width: 600px;
    }
    
    /* ============================================
       DOMAIN CARDS
    ============================================ */
    .domain-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1.25rem;
        cursor: pointer;
        transition: all 0.25s ease;
        height: 100%;
    }
    
    .domain-card:hover {
        transform: translateY(-2px);
        border-color: var(--accent-primary);
        box-shadow: var(--shadow-glow);
    }
    
    .domain-card.selected {
        background: linear-gradient(135deg, rgba(0,212,170,0.1) 0%, rgba(0,212,170,0.02) 100%);
        border-color: var(--accent-primary);
    }
    
    .domain-icon {
        font-size: 2rem;
        margin-bottom: 0.75rem;
    }
    
    .domain-name {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
    }
    
    .domain-desc {
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    
    /* ============================================
       RESEARCH OUTPUT STYLES
    ============================================ */
    .research-output {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 2.5rem;
        font-family: var(--font-serif);
        line-height: 1.8;
    }
    
    .research-output h1 {
        font-family: var(--font-serif);
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        border-bottom: 2px solid var(--accent-primary);
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    .research-output h2 {
        font-family: var(--font-sans);
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--accent-primary);
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    .research-output h3 {
        font-family: var(--font-sans);
        font-size: 1.1rem;
        font-weight: 500;
        color: var(--text-primary);
        margin-top: 1.5rem;
    }
    
    .research-output p {
        color: var(--text-secondary);
        text-align: justify;
        margin-bottom: 1rem;
    }
    
    .research-output ul, .research-output ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .research-output li {
        margin-bottom: 0.5rem;
    }
    
    .research-output table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        font-family: var(--font-sans);
        font-size: 0.9rem;
    }
    
    .research-output th {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        border-bottom: 2px solid var(--accent-primary);
    }
    
    .research-output td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--border-subtle);
        color: var(--text-secondary);
    }
    
    .research-output tr:hover td {
        background: var(--bg-card);
    }
    
    .research-output blockquote {
        border-left: 3px solid var(--accent-primary);
        margin: 1.5rem 0;
        padding: 1rem 1.5rem;
        background: rgba(0,212,170,0.05);
        font-style: italic;
    }
    
    .research-output code {
        font-family: var(--font-mono);
        background: var(--bg-tertiary);
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.85em;
    }
    
    /* ============================================
       PROGRESS & STATUS INDICATORS
    ============================================ */
    .phase-indicator {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        font-family: var(--font-mono);
    }
    
    .phase-pending {
        background: var(--bg-tertiary);
        color: var(--text-muted);
    }
    
    .phase-active {
        background: rgba(0,212,170,0.15);
        color: var(--accent-primary);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .phase-complete {
        background: rgba(0,212,170,0.1);
        color: var(--accent-primary);
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    /* ============================================
       PAPER REFERENCE CARDS
    ============================================ */
    .paper-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease;
    }
    
    .paper-card:hover {
        border-color: var(--accent-secondary);
        background: var(--bg-card-hover);
    }
    
    .paper-title {
        font-weight: 500;
        color: var(--text-primary);
        margin-bottom: 0.25rem;
        font-size: 0.95rem;
    }
    
    .paper-meta {
        font-size: 0.8rem;
        color: var(--text-muted);
    }
    
    .paper-source {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        background: var(--bg-tertiary);
        border-radius: 4px;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--accent-secondary);
        margin-left: 0.5rem;
    }
    
    /* ============================================
       CHAT/MESSAGE STYLES
    ============================================ */
    .message-user {
        background: linear-gradient(135deg, rgba(139,92,246,0.15) 0%, rgba(139,92,246,0.05) 100%);
        border: 1px solid rgba(139,92,246,0.2);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    .message-assistant {
        background: var(--bg-secondary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    
    .message-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .message-label-user {
        color: var(--accent-tertiary);
    }
    
    .message-label-assistant {
        color: var(--accent-primary);
    }
    
    /* ============================================
       AGENT STATUS CARDS
    ============================================ */
    .agent-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .agent-card.active {
        border-color: var(--accent-primary);
        animation: pulse 2s ease-in-out infinite;
    }
    
    .agent-card.complete {
        border-left: 3px solid var(--accent-primary);
    }
    
    .agent-name {
        font-weight: 600;
        color: var(--text-primary);
        font-size: 0.9rem;
    }
    
    .agent-status {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-family: var(--font-mono);
    }
    
    /* ============================================
       BUTTONS
    ============================================ */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary) 0%, #00b894 100%);
        color: var(--bg-primary);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-family: var(--font-sans);
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,212,170,0.3);
    }
    
    .stButton > button[kind="secondary"] {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border-medium);
    }
    
    /* ============================================
       INPUT FIELDS
    ============================================ */
    .stTextArea textarea, .stTextInput input {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
        font-family: var(--font-sans) !important;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: var(--accent-primary) !important;
        box-shadow: 0 0 0 2px rgba(0,212,170,0.2) !important;
    }
    
    /* ============================================
       EXPANDERS
    ============================================ */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-subtle) !important;
        border-top: none !important;
    }
    
    /* ============================================
       TABS
    ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-tertiary);
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 6px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--accent-primary) !important;
    }
    
    /* ============================================
       CHECKBOX STYLING
    ============================================ */
    .stCheckbox label {
        color: var(--text-primary) !important;
    }
    
    .stCheckbox label span {
        color: var(--text-secondary) !important;
    }
    
    /* ============================================
       SPINNER / LOADING
    ============================================ */
    .stSpinner > div {
        border-top-color: var(--accent-primary) !important;
    }
    
    /* ============================================
       SCROLLBAR
    ============================================ */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--bg-tertiary);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* ============================================
       INFO/WARNING/ERROR BOXES
    ============================================ */
    .stAlert {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 8px !important;
    }
    
    /* ============================================
       DIVIDERS
    ============================================ */
    hr {
        border: none;
        border-top: 1px solid var(--border-subtle);
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


def init_key_manager():
    """Initialize the API key manager with multiple keys."""
    keys = settings.llm_api_keys
    if keys:
        key_manager = initialize_key_manager(
            keys=keys,
            base_url=settings.openai_base_url if settings.llm_provider == "openai" and settings.openai_base_url else None,
            model=settings.llm_model,
            provider=settings.llm_provider
        )
        return key_manager
    return None


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "page": "home",
        "selected_fields": [],
        "team_configured": False,
        "messages": [],
        "research_graph": None,
        "last_results": None,
        "research_history": [],
        "current_phase": "idle"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Research Lab - Academic Research Platform",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    inject_custom_css()
    
    # Initialize key manager first (before any agents are created)
    init_key_manager()
    
    init_session_state()
    
    # Route to appropriate page
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "team_setup":
        render_team_setup_page()
    elif st.session_state.page == "research":
        render_research_session_page()


if __name__ == "__main__":
    main()
