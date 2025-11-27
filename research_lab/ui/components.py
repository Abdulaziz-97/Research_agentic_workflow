"""Professional reusable UI components for Research Lab."""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime
from config.settings import FIELD_DISPLAY_NAMES, FIELD_DESCRIPTIONS, RESEARCH_FIELDS


# Domain icons mapping
DOMAIN_ICONS = {
    "ai_ml": "🤖",
    "physics": "⚛️",
    "biology": "🧬",
    "chemistry": "⚗️",
    "mathematics": "📐",
    "neuroscience": "🧠",
    "medicine": "💊",
    "computer_science": "💻"
}

SUPPORT_AGENT_INFO = {
    "literature_reviewer": {
        "icon": "📚",
        "name": "Literature Reviewer",
        "desc": "Systematically analyzes and summarizes relevant papers"
    },
    "methodology_critic": {
        "icon": "🔍",
        "name": "Methodology Critic",
        "desc": "Evaluates research methods and experimental designs"
    },
    "fact_checker": {
        "icon": "✓",
        "name": "Fact Checker",
        "desc": "Verifies claims against primary sources"
    },
    "writing_assistant": {
        "icon": "✍️",
        "name": "Writing Assistant",
        "desc": "Refines and structures research output"
    },
    "cross_domain_synthesizer": {
        "icon": "🔗",
        "name": "Cross-Domain Synthesizer",
        "desc": "Identifies connections between research fields"
    }
}


def render_hero_header(title: str = "Research Lab", subtitle: str = None):
    """Render the main hero header."""
    st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle or "Multi-Agent Academic Research Platform"}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the professional sidebar."""
    with st.sidebar:
        # Logo/Brand
        st.markdown("""
        <div style="text-align: center; padding: 1.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.06);">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: #00d4aa;">Research Lab</div>
            <div style="font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em;">Academic Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
        
        # Navigation
        st.markdown("<div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;'>Navigation</div>", unsafe_allow_html=True)
        
        current_page = st.session_state.get("page", "home")
        
        if st.button("🏠  Home", use_container_width=True, type="primary" if current_page == "home" else "secondary"):
            st.session_state.page = "home"
            st.rerun()
        
        if st.button("👥  Configure Team", use_container_width=True, type="primary" if current_page == "team_setup" else "secondary"):
            st.session_state.page = "team_setup"
            st.rerun()
        
        if st.session_state.get("team_configured"):
            if st.button("🔬  Research Session", use_container_width=True, type="primary" if current_page == "research" else "secondary"):
                st.session_state.page = "research"
                st.rerun()
        
        # Current Team Status
        if st.session_state.get("team_configured"):
            st.markdown("""<hr style='margin: 1.5rem 0; border-color: rgba(255,255,255,0.06);'>""", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;'>Active Team</div>", unsafe_allow_html=True)
            
            for field in st.session_state.get("selected_fields", []):
                icon = DOMAIN_ICONS.get(field, "📌")
                name = FIELD_DISPLAY_NAMES.get(field, field)
                st.markdown(f"""
                <div class="agent-card complete">
                    <div class="agent-name">{icon} {name}</div>
                    <div class="agent-status">Ready</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Session Stats
        if st.session_state.get("research_history"):
            st.markdown("""<hr style='margin: 1.5rem 0; border-color: rgba(255,255,255,0.06);'>""", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;'>Session Stats</div>", unsafe_allow_html=True)
            
            history_count = len(st.session_state.research_history)
            st.markdown(f"""
            <div style='font-size: 0.85rem; color: #a0a0b0;'>
                {history_count} research {"query" if history_count == 1 else "queries"} completed
            </div>
            """, unsafe_allow_html=True)


def render_domain_grid(selected_fields: List[str], max_selections: int = 3) -> List[str]:
    """Render interactive domain selection grid."""
    st.markdown("""
    <div style='margin-bottom: 1rem;'>
        <div style='font-size: 0.8rem; color: #606070;'>Select up to 3 research domains for your team</div>
    </div>
    """, unsafe_allow_html=True)
    
    new_selections = selected_fields.copy()
    
    # Create 4-column grid
    cols = st.columns(4)
    
    for i, field in enumerate(RESEARCH_FIELDS):
        col = cols[i % 4]
        
        icon = DOMAIN_ICONS.get(field, "📌")
        name = FIELD_DISPLAY_NAMES.get(field, field)
        desc = FIELD_DESCRIPTIONS.get(field, "")
        
        is_selected = field in new_selections
        can_select = is_selected or len(new_selections) < max_selections
        
        with col:
            # Custom styled checkbox
            selected_class = "selected" if is_selected else ""
            disabled_style = "opacity: 0.5; cursor: not-allowed;" if not can_select else ""
            
            if st.checkbox(
                f"{icon} {name}",
                value=is_selected,
                key=f"domain_{field}",
                disabled=not can_select and not is_selected
            ):
                if field not in new_selections:
                    new_selections.append(field)
            else:
                if field in new_selections:
                    new_selections.remove(field)
            
            st.caption(desc[:60] + "..." if len(desc) > 60 else desc)
    
    return new_selections


def render_support_agents_panel():
    """Render the support agents information panel."""
    st.markdown("""
    <div class="research-card">
        <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;'>Support Agents (Always Active)</div>
    """, unsafe_allow_html=True)
    
    for agent_id, info in SUPPORT_AGENT_INFO.items():
        st.markdown(f"""
        <div style='display: flex; align-items: flex-start; margin-bottom: 0.75rem;'>
            <div style='font-size: 1.2rem; margin-right: 0.75rem;'>{info['icon']}</div>
            <div>
                <div style='font-weight: 500; color: #f0f0f5; font-size: 0.9rem;'>{info['name']}</div>
                <div style='font-size: 0.75rem; color: #606070;'>{info['desc']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_team_summary(selected_fields: List[str]):
    """Render the team summary card."""
    if not selected_fields:
        st.info("Select at least one domain to configure your research team.")
        return
    
    st.markdown(f"""
    <div class="research-card research-card-accent">
        <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;'>Team Configuration</div>
        <div style='display: flex; gap: 2rem; flex-wrap: wrap;'>
            <div class="stats-card" style='flex: 1; min-width: 120px;'>
                <div class="stats-value">{len(selected_fields)}</div>
                <div class="stats-label">Domain Agents</div>
            </div>
            <div class="stats-card" style='flex: 1; min-width: 120px;'>
                <div class="stats-value">5</div>
                <div class="stats-label">Support Agents</div>
            </div>
            <div class="stats-card" style='flex: 1; min-width: 120px;'>
                <div class="stats-value">{len(selected_fields) + 5}</div>
                <div class="stats-label">Total Team</div>
            </div>
        </div>
        <div style='margin-top: 1.5rem;'>
            <div style='font-size: 0.8rem; color: #a0a0b0; margin-bottom: 0.5rem;'>Active Domains:</div>
            <div style='display: flex; gap: 0.5rem; flex-wrap: wrap;'>
    """, unsafe_allow_html=True)
    
    for field in selected_fields:
        icon = DOMAIN_ICONS.get(field, "📌")
        name = FIELD_DISPLAY_NAMES.get(field, field)
        st.markdown(f"""
                <span style='background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.3); border-radius: 20px; padding: 0.4rem 0.8rem; font-size: 0.85rem; color: #00d4aa;'>
                    {icon} {name}
                </span>
        """, unsafe_allow_html=True)
    
    st.markdown("""
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_research_output(response: str, stats: Dict = None):
    """Render the academic research output with proper formatting."""
    # Stats bar
    if stats:
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats.get('total_papers', 0)}</div>
                <div class="stats-label">Papers Found</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats.get('domains_consulted', 0)}</div>
                <div class="stats-label">Domains</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            confidence = stats.get('avg_confidence', 0) * 100
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{confidence:.0f}%</div>
                <div class="stats-label">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-value">{stats.get('execution_time', 'N/A')}</div>
                <div class="stats-label">Duration</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    # Research output in styled container
    st.markdown(f"""
    <div class="research-output">
        {response}
    </div>
    """, unsafe_allow_html=True)


def render_paper_list(papers: List, title: str = "Sources"):
    """Render a list of paper references."""
    if not papers:
        return
    
    st.markdown(f"""
    <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;'>{title}</div>
    """, unsafe_allow_html=True)
    
    for paper in papers[:10]:  # Limit to 10
        # Handle both dict and object access
        if isinstance(paper, dict):
            title = paper.get('title', 'Untitled')
            authors = paper.get('authors', [])
            source = paper.get('source', 'Unknown')
            url = paper.get('url', '')
        else:
            title = getattr(paper, 'title', 'Untitled')
            authors = getattr(paper, 'authors', [])
            source = getattr(paper, 'source', 'Unknown')
            url = getattr(paper, 'url', '')
        
        authors_str = ", ".join(authors[:2]) if authors else "Unknown"
        if len(authors) > 2:
            authors_str += " et al."
        
        st.markdown(f"""
        <div class="paper-card">
            <div class="paper-title">{title}</div>
            <div class="paper-meta">
                {authors_str}
                <span class="paper-source">{source}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_phase_tracker(current_phase: str, phases: Dict = None):
    """Render the research phase progress tracker."""
    phase_list = [
        ("init", "Initialize", "Setting up research parameters"),
        ("routing", "Routing", "Analyzing query and selecting agents"),
        ("domain_research", "Research", "Domain agents investigating"),
        ("support_review", "Review", "Support agents analyzing"),
        ("synthesis", "Synthesis", "Generating academic output"),
        ("complete", "Complete", "Research finalized")
    ]
    
    st.markdown("""
    <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;'>Research Progress</div>
    """, unsafe_allow_html=True)
    
    for phase_id, phase_name, phase_desc in phase_list:
        if phase_id == current_phase:
            status_class = "phase-active"
            icon = "⏳"
        elif phases and phases.get(phase_id, {}).get("status") == "complete":
            status_class = "phase-complete"
            icon = "✓"
        else:
            status_class = "phase-pending"
            icon = "○"
        
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
            <span class="phase-indicator {status_class}" style='min-width: 120px;'>
                {icon} {phase_name}
            </span>
            <span style='font-size: 0.75rem; color: #606070; margin-left: 0.75rem;'>{phase_desc}</span>
        </div>
        """, unsafe_allow_html=True)


def render_query_input():
    """Render the research query input area."""
    st.markdown("""
    <div class="research-card">
        <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;'>Research Query</div>
    """, unsafe_allow_html=True)
    
    query = st.text_area(
        "Enter your research question",
        placeholder="Enter a complex research question. For example:\n\n• How do transformer architectures compare to traditional RNNs for time-series forecasting in healthcare applications?\n\n• What are the molecular mechanisms underlying neuroplasticity, and how do they inform treatments for neurodegenerative diseases?",
        height=150,
        label_visibility="collapsed"
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    return query


def render_message(content: str, role: str = "assistant", label: str = None):
    """Render a chat message."""
    if role == "user":
        msg_class = "message-user"
        label_class = "message-label-user"
        default_label = "YOUR QUERY"
    else:
        msg_class = "message-assistant"
        label_class = "message-label-assistant"
        default_label = "RESEARCH TEAM"
    
    display_label = label or default_label
    
    st.markdown(f"""
    <div class="{msg_class}">
        <div class="message-label {label_class}">{display_label}</div>
        <div style='color: #a0a0b0;'>{content if role == "user" else ""}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # For assistant messages, render the markdown content with st.markdown for proper formatting
    if role == "assistant":
        st.markdown(content)
