"""Research session page - Main research command center."""

import streamlit as st
from typing import Optional, Dict, Any
from datetime import datetime

from ui.components import (
    render_sidebar,
    render_hero_header,
    render_research_output,
    render_paper_list,
    render_phase_tracker,
    render_query_input,
    render_message,
    DOMAIN_ICONS,
    SUPPORT_AGENT_INFO
)
from config.settings import FIELD_DISPLAY_NAMES
from graphs.research_graph import create_research_graph


def get_or_create_graph():
    """Get or create the research graph."""
    if "research_graph" not in st.session_state or st.session_state.research_graph is None:
        team_config = st.session_state.get("team_config")
        if team_config:
            st.session_state.research_graph = create_research_graph(team_config)
    return st.session_state.get("research_graph")


def render_active_team_bar():
    """Render the active team status bar."""
    team_config = st.session_state.get("team_config")
    if not team_config:
        return
    
    # Use columns for the team bar
    cols = st.columns([1, 4, 1])
    
    with cols[0]:
        st.markdown("<span style='font-size: 0.7rem; color: #606070; text-transform: uppercase;'>Active Team</span>", unsafe_allow_html=True)
    
    with cols[1]:
        team_names = []
        for field in team_config.domain_agents:
            icon = DOMAIN_ICONS.get(field, "📌")
            name = FIELD_DISPLAY_NAMES.get(field, field)
            team_names.append(f"{icon} {name}")
        st.markdown(" **|** ".join(team_names))
    
    with cols[2]:
        st.caption(f"Session: {team_config.team_id[:8]}...")


def render_research_session_page():
    """Render the main research session page."""
    render_sidebar()
    
    # Check for team configuration
    if not st.session_state.get("team_configured"):
        st.markdown("""
        <div class="hero-container" style='text-align: center;'>
            <div style='font-size: 3rem; margin-bottom: 1rem;'>⚠️</div>
            <h2 style='color: #f0f0f5;'>No Research Team Configured</h2>
            <p style='color: #a0a0b0;'>Please configure your research team before starting a session.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔧  Configure Team", type="primary"):
            st.session_state.page = "team_setup"
            st.rerun()
        return
    
    # Header
    st.markdown("""
    <div style='display: flex; align-items: center; margin-bottom: 1.5rem;'>
        <div>
            <h1 style='color: #f0f0f5; margin: 0; font-size: 2rem;'>🔬 Research Session</h1>
            <p style='color: #606070; margin: 0; font-size: 0.9rem;'>Academic Research Command Center</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Active team bar
    render_active_team_bar()
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    # Main layout
    main_col, side_col = st.columns([3, 1])
    
    with main_col:
        # Query input section
        query = render_query_input()
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            submit = st.button("🔬  Begin Research Investigation", type="primary", use_container_width=True)
        with col2:
            clear = st.button("🗑️  Clear Session", use_container_width=True)
        with col3:
            export = st.button("📥  Export", use_container_width=True)
        
        if clear:
            st.session_state.messages = []
            st.session_state.last_results = None
            st.session_state.research_history = []
            st.rerun()
        
        if export and st.session_state.get("last_results"):
            # Export functionality
            result = st.session_state.last_results
            response = result.get("final_response", "")
            st.download_button(
                "📄 Download as Markdown",
                response,
                file_name=f"research_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
        
        # Research execution
        if submit and query.strip():
            # Add to messages
            messages = st.session_state.get("messages", [])
            messages.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.messages = messages
            
            # Show progress
            with st.status("🔬 Research in progress...", expanded=True) as status:
                try:
                    st.write("Initializing research team...")
                    graph = get_or_create_graph()
                    
                    if graph:
                        st.write("Routing query to domain experts...")
                        st.write("Searching academic databases...")
                        
                        # Run the research
                        result = graph.run_sync(query, thread_id=f"session_{datetime.now().timestamp()}")
                        
                        st.write("Synthesizing findings...")
                        
                        # Safely extract response
                        final_response = ""
                        if isinstance(result, dict):
                            final_response = result.get("final_response", "") or ""
                        
                        # Safely extract domain results
                        domain_results = []
                        if isinstance(result, dict):
                            domain_results = result.get("domain_results", []) or []
                        
                        # Safely extract stats
                        research_stats = {}
                        if isinstance(result, dict):
                            research_stats = result.get("research_stats", {}) or {}
                        
                        # Collect all papers
                        all_papers = []
                        for dr in domain_results:
                            try:
                                if hasattr(dr, 'papers'):
                                    all_papers.extend(dr.papers)
                                elif isinstance(dr, dict) and 'papers' in dr:
                                    all_papers.extend(dr['papers'])
                            except:
                                pass
                        
                        if final_response:
                            messages.append({
                                "role": "assistant",
                                "content": final_response,
                                "timestamp": datetime.now().isoformat(),
                                "stats": research_stats,
                                "papers": all_papers
                            })
                            st.session_state.messages = messages
                            st.session_state.last_results = dict(result) if result else {}
                            
                            # Add to history
                            history = st.session_state.get("research_history", [])
                            history.append({
                                "query": query,
                                "timestamp": datetime.now().isoformat(),
                                "papers_count": len(all_papers)
                            })
                            st.session_state.research_history = history
                            
                            status.update(label="✅ Research complete!", state="complete", expanded=False)
                        else:
                            messages.append({
                                "role": "assistant",
                                "content": "Unable to generate research output. Please try again with a more specific query.",
                                "timestamp": datetime.now().isoformat()
                            })
                            st.session_state.messages = messages
                            status.update(label="⚠️ No results generated", state="complete", expanded=False)
                    else:
                        st.error("Failed to initialize research graph")
                    
                except Exception as e:
                    import traceback
                    error_details = traceback.format_exc()
                    st.error(f"Research Error: {str(e)}")
                    messages.append({
                        "role": "assistant",
                        "content": f"An error occurred during research: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.messages = messages
                    status.update(label=f"❌ Error: {str(e)[:50]}...", state="error", expanded=False)
            
            st.rerun()
        
        # Display conversation history
        messages = st.session_state.get("messages", [])
        
        if not messages:
            st.markdown("""
            <div class="research-card" style='text-align: center; padding: 3rem;'>
                <div style='font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;'>🔬</div>
                <h3 style='color: #a0a0b0; font-weight: 400;'>Ready for Research</h3>
                <p style='color: #606070; max-width: 500px; margin: 0 auto;'>
                    Enter a complex research question above. Your team of AI research agents will 
                    search academic databases, analyze findings, and synthesize a comprehensive 
                    research brief.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show messages in reverse order (newest first)
            for i, msg in enumerate(reversed(messages)):
                if msg["role"] == "user":
                    st.markdown(f"""
                    <div class="message-user">
                        <div class="message-label message-label-user">YOUR QUERY</div>
                        <div style='color: #f0f0f5;'>{msg["content"]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # For assistant messages, render with full academic formatting
                    st.markdown("""
                    <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem; margin-top: 1.5rem;'>Research Output</div>
                    """, unsafe_allow_html=True)
                    
                    # Stats if available
                    stats = msg.get("stats", {})
                    if stats:
                        render_research_output(msg["content"], stats)
                    else:
                        st.markdown(f"""
                        <div class="research-output">
                            {msg["content"]}
                        </div>
                        """, unsafe_allow_html=True)
    
    with side_col:
        # Research Info Panel
        st.markdown("""
        <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem;'>Research Pipeline</div>
        """, unsafe_allow_html=True)
        
        # Show active agents
        team_config = st.session_state.get("team_config")
        if team_config:
            st.markdown("""
            <div class="research-card" style='padding: 1rem;'>
                <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;'>Domain Experts</div>
            """, unsafe_allow_html=True)
            
            for field in team_config.domain_agents:
                icon = DOMAIN_ICONS.get(field, "📌")
                name = FIELD_DISPLAY_NAMES.get(field, field)
                st.markdown(f"""
                <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                    <span style='font-size: 1.2rem; margin-right: 0.5rem;'>{icon}</span>
                    <span style='color: #a0a0b0; font-size: 0.85rem;'>{name}</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Support agents
        st.markdown("""
        <div class="research-card" style='padding: 1rem; margin-top: 0.75rem;'>
            <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.75rem;'>Support Team</div>
        """, unsafe_allow_html=True)
        
        for agent_id, info in SUPPORT_AGENT_INFO.items():
            st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 0.4rem;'>
                <span style='font-size: 0.9rem; margin-right: 0.4rem;'>{info['icon']}</span>
                <span style='color: #606070; font-size: 0.75rem;'>{info['name']}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Papers from last result
        last_results = st.session_state.get("last_results")
        if last_results:
            all_papers = []
            domain_results = last_results.get("domain_results", [])
            for dr in domain_results:
                if hasattr(dr, 'papers'):
                    all_papers.extend(dr.papers)
                elif isinstance(dr, dict) and 'papers' in dr:
                    all_papers.extend(dr['papers'])
            
            if all_papers:
                st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                render_paper_list(all_papers, title="Referenced Papers")
        
        # Session history
        history = st.session_state.get("research_history", [])
        if history:
            st.markdown("""
            <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.75rem; margin-top: 1.5rem;'>Session History</div>
            """, unsafe_allow_html=True)
            
            for i, h in enumerate(reversed(history[-5:])):  # Last 5
                query_preview = h["query"][:50] + "..." if len(h["query"]) > 50 else h["query"]
                st.markdown(f"""
                <div class="paper-card" style='padding: 0.75rem;'>
                    <div style='font-size: 0.8rem; color: #a0a0b0;'>{query_preview}</div>
                    <div style='font-size: 0.7rem; color: #606070; margin-top: 0.25rem;'>
                        {h.get('papers_count', 0)} papers
                    </div>
                </div>
                """, unsafe_allow_html=True)
