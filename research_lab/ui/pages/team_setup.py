"""Team setup page for Research Lab - Professional Team Configuration."""

import streamlit as st
import uuid

from ui.components import (
    render_sidebar, 
    render_hero_header, 
    render_domain_grid,
    render_support_agents_panel,
    render_team_summary
)
from config.settings import RESEARCH_FIELDS, FIELD_DISPLAY_NAMES
from states.agent_state import TeamConfiguration


def render_team_setup_page():
    """Render the team configuration page."""
    render_sidebar()
    
    # Header
    render_hero_header(
        title="Configure Research Team",
        subtitle="Select up to 3 domain specialists to investigate your research questions. All support agents are automatically included."
    )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;'>Domain Specialists</div>
        """, unsafe_allow_html=True)
        
        current_selections = st.session_state.get("selected_fields", [])
        new_selections = render_domain_grid(current_selections, max_selections=3)
        st.session_state.selected_fields = new_selections
        
        # Selection counter
        remaining = 3 - len(new_selections)
        if remaining > 0:
            st.markdown(f"""
            <div style='margin-top: 1rem; padding: 0.75rem 1rem; background: rgba(14, 165, 233, 0.1); border: 1px solid rgba(14, 165, 233, 0.3); border-radius: 8px;'>
                <span style='color: #0ea5e9; font-weight: 500;'>ℹ️ You can select {remaining} more domain{"s" if remaining > 1 else ""}</span>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        render_support_agents_panel()
    
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    
    # Team Summary
    render_team_summary(new_selections)
    
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if new_selections:
            if st.button("✅  Confirm Team & Begin Research", type="primary", use_container_width=True):
                # Create team configuration
                team_config = TeamConfiguration(
                    team_id=str(uuid.uuid4()),
                    name="Research Team",
                    domain_agents=new_selections,
                    support_agents=[
                        "literature_reviewer",
                        "methodology_critic", 
                        "fact_checker",
                        "writing_assistant",
                        "cross_domain_synthesizer"
                    ]
                )
                
                # Save to session state
                st.session_state.team_config = team_config
                st.session_state.team_configured = True
                st.session_state.messages = []
                st.session_state.research_history = []
                st.session_state.page = "research"
                st.rerun()
        else:
            st.markdown("""
            <div style='text-align: center; padding: 1rem; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px;'>
                <span style='color: #f59e0b;'>⚠️ Select at least one domain to continue</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Tips section
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    
    with st.expander("💡 Tips for Team Selection"):
        st.markdown("""
        **Choosing the Right Team:**
        
        - **Single Domain**: Best for focused, deep investigations within one field
        - **Two Domains**: Ideal for questions at the intersection of fields (e.g., AI + Neuroscience)
        - **Three Domains**: Best for complex, multi-disciplinary research questions
        
        **Example Questions by Team Size:**
        
        | Team Size | Example Question |
        |-----------|-----------------|
        | 1 (AI/ML) | "What are the latest advances in vision transformers for medical imaging?" |
        | 2 (AI + Neuro) | "How are deep learning models being used to decode neural signals?" |
        | 3 (AI + Bio + Med) | "What role does AI play in drug discovery for neurodegenerative diseases?" |
        
        **Remember**: All support agents (Literature Reviewer, Methodology Critic, Fact Checker, Writing Assistant, Cross-Domain Synthesizer) are always active to ensure quality output.
        """)
