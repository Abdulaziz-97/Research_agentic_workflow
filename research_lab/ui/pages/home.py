"""Home page for Research Lab - Professional Landing Experience."""

import streamlit as st
from ui.components import render_sidebar, render_hero_header, DOMAIN_ICONS
from config.settings import FIELD_DISPLAY_NAMES, FIELD_DESCRIPTIONS, RESEARCH_FIELDS


def render_home_page():
    """Render the professional home page."""
    render_sidebar()
    
    # Hero Section
    render_hero_header(
        title="Research Lab",
        subtitle="A multi-agent AI system for rigorous academic research. Deploy specialized domain experts and support agents to investigate complex research questions with publication-quality output."
    )
    
    # Value Proposition
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="research-card research-card-accent">
            <h3 style='color: #f0f0f5; margin-top: 0;'>What Makes This Different</h3>
            <p style='color: #a0a0b0; line-height: 1.8;'>
                Unlike simple chatbots, Research Lab deploys a <strong style='color: #00d4aa;'>coordinated team of specialized AI agents</strong> 
                that work together like a real research group. Each agent has deep expertise in their domain, access to academic 
                databases, and the ability to critically analyze and synthesize findings.
            </p>
            <div style='margin-top: 1.5rem;'>
                <div style='display: flex; gap: 1.5rem; flex-wrap: wrap;'>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📚</div>
                        <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;'>Academic Rigor</div>
                        <div style='font-size: 0.85rem; color: #606070;'>Every claim backed by citations from peer-reviewed sources</div>
                    </div>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>🔬</div>
                        <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;'>Multi-Domain</div>
                        <div style='font-size: 0.85rem; color: #606070;'>Cross-disciplinary synthesis from multiple expert perspectives</div>
                    </div>
                    <div style='flex: 1; min-width: 200px;'>
                        <div style='font-size: 1.5rem; margin-bottom: 0.5rem;'>📄</div>
                        <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.25rem;'>Publication Quality</div>
                        <div style='font-size: 0.85rem; color: #606070;'>Structured output that reads like a research paper</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="research-card" style='height: 100%;'>
            <h3 style='color: #f0f0f5; margin-top: 0;'>Quick Start</h3>
            <div style='color: #a0a0b0;'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <span style='background: #00d4aa; color: #0a0a0f; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;'>1</span>
                    <span>Configure your research team</span>
                </div>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <span style='background: #0ea5e9; color: #0a0a0f; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;'>2</span>
                    <span>Ask a research question</span>
                </div>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <span style='background: #8b5cf6; color: white; width: 24px; height: 24px; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.8rem; margin-right: 0.75rem;'>3</span>
                    <span>Receive a research brief</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    
    # Research Domains Section
    st.markdown("""
    <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;'>Available Research Domains</div>
    """, unsafe_allow_html=True)
    
    # Domain cards in 4 columns
    cols = st.columns(4)
    for i, field in enumerate(RESEARCH_FIELDS):
        with cols[i % 4]:
            icon = DOMAIN_ICONS.get(field, "📌")
            name = FIELD_DISPLAY_NAMES.get(field, field)
            desc = FIELD_DESCRIPTIONS.get(field, "")
            
            st.markdown(f"""
            <div class="domain-card">
                <div class="domain-icon">{icon}</div>
                <div class="domain-name">{name}</div>
                <div class="domain-desc">{desc[:80]}...</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    
    # How It Works
    st.markdown("""
    <div style='font-size: 0.7rem; color: #606070; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 1rem;'>How The System Works</div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="research-card" style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>🎯</div>
            <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.5rem;'>Query Analysis</div>
            <div style='font-size: 0.8rem; color: #606070;'>
                The orchestrator analyzes your question and routes it to the most relevant domain experts
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="research-card" style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>🔍</div>
            <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.5rem;'>Literature Search</div>
            <div style='font-size: 0.8rem; color: #606070;'>
                Agents search arXiv, PubMed, Semantic Scholar, and more for relevant papers
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="research-card" style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>🧪</div>
            <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.5rem;'>Critical Analysis</div>
            <div style='font-size: 0.8rem; color: #606070;'>
                Support agents review methodology, verify facts, and synthesize across domains
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="research-card" style='text-align: center; padding: 1.5rem;'>
            <div style='font-size: 2rem; margin-bottom: 1rem;'>📝</div>
            <div style='font-weight: 600; color: #f0f0f5; margin-bottom: 0.5rem;'>Academic Output</div>
            <div style='font-size: 0.8rem; color: #606070;'>
                Receive a structured research brief with proper citations and references
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 2rem'></div>", unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀  Configure Your Research Team", type="primary", use_container_width=True):
            st.session_state.page = "team_setup"
            st.rerun()
