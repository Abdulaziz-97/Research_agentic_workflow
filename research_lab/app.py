"""
Research Lab - Multi-Agent Research Workflow
Main Streamlit Application Entry Point
"""

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
from ui.pages.home import render_home_page
from ui.pages.team_setup import render_team_setup_page
from ui.pages.research_session import render_research_session_page


def init_session_state():
    """Initialize Streamlit session state variables."""
    if "page" not in st.session_state:
        st.session_state.page = "home"
    if "selected_fields" not in st.session_state:
        st.session_state.selected_fields = []
    if "team_configured" not in st.session_state:
        st.session_state.team_configured = False
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "research_graph" not in st.session_state:
        st.session_state.research_graph = None


def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="Research Lab - AI Research Team",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Global theming & layout CSS for a modern, app-like experience
    st.markdown(
        """
        <style>
        /* Global background and typography */
        .stApp {
            background: radial-gradient(circle at top left, #1e293b 0, #020617 45%, #020617 100%);
            color: #e5e7eb;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                         "Segoe UI", Roboto, sans-serif;
        }

        /* Center the main content and add breathing room */
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1100px;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
            backdrop-filter: blur(18px);
        }

        /* Buttons */
        .stButton > button {
            border-radius: 999px;
            border: none;
            padding: 0.55rem 1.3rem;
            font-weight: 500;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            color: white;
            box-shadow: 0 12px 30px rgba(79, 70, 229, 0.35);
            transition: all 0.18s ease-out;
        }
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 18px 40px rgba(79, 70, 229, 0.55);
            filter: brightness(1.05);
        }

        /* Cards */
        .rl-card {
            background: radial-gradient(circle at top left, #1f2937 0, #020617 55%);
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.3);
            padding: 1.1rem 1.2rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.8);
        }
        .rl-card-soft {
            background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.65));
            border-radius: 1rem;
            border: 1px solid rgba(148, 163, 184, 0.25);
            padding: 1rem 1.1rem;
        }

        /* Chat bubbles */
        .rl-chat-user {
            background: linear-gradient(135deg, #22c55e, #16a34a);
            color: #ecfdf5;
            padding: 0.6rem 0.85rem;
            border-radius: 0.9rem;
            border-bottom-right-radius: 0.2rem;
            font-size: 0.94rem;
        }
        .rl-chat-assistant {
            background: linear-gradient(135deg, #0f172a, #020617);
            border-radius: 0.9rem;
            border-bottom-left-radius: 0.2rem;
            border: 1px solid rgba(148, 163, 184, 0.35);
            padding: 0.6rem 0.85rem;
            font-size: 0.94rem;
        }

        /* Subtle scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(148, 163, 184, 0.7);
            border-radius: 999px;
        }

        /* Hide Streamlit default menu & footer for app-like feel */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    init_session_state()
    
    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "team_setup":
        render_team_setup_page()
    elif st.session_state.page == "research":
        render_research_session_page()


if __name__ == "__main__":
    main()

