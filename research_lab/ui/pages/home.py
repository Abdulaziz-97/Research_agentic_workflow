"""Home page for Research Lab."""

import streamlit as st
from ui.components import render_header, render_sidebar


def render_home_page():
    """Render the home page."""
    render_sidebar()
    render_header()

    st.markdown(" ")

    # Hero section and quick start
    col_main, col_side = st.columns([2.2, 1.3])

    with col_main:
        st.markdown(
            """
            <div class="rl-card-soft">
                <h3 style="margin-top: 0; margin-bottom: 0.4rem;">Welcome to your AI Research Lab</h3>
                <p style="margin: 0; color: #9ca3af; font-size: 0.94rem;">
                    Assemble a cross-disciplinary team of AI experts, connect them to live scientific data sources,
                    and let them collaborate on your hardest research questions.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(" ")

        st.markdown("### 🔬 How it works")
        steps = [
            ("Build your team", "Select up to 3 domain specialists that match your research problem."),
            ("Ask a complex question", "Pose open-ended, multi-faceted research questions, not simple facts."),
            ("Let agents collaborate", "Domain and support agents search, critique, and cross-check each other."),
            ("Review synthesized insights", "Receive a structured answer with citations and cross-domain links."),
        ]

        for title, desc in steps:
            st.markdown(f"- **{title}**: {desc}")

    with col_side:
        st.markdown(
            """
            <div class="rl-card">
                <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.18em; color: #a5b4fc; margin-bottom: 0.3rem;">
                    Quick Start
                </div>
                <p style="margin: 0 0 0.6rem 0; font-size: 0.88rem; color: #e5e7eb;">
                    Configure a research team and start a live session in under a minute.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(" ")
        if st.button("Configure your team →", type="primary", use_container_width=True):
            st.session_state.page = "team_setup"
            st.rerun()

    st.markdown(" ")
    st.markdown("### Available Research Domains")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**🤖 AI / Machine Learning**")
        st.caption("Models, benchmarks, and learning theory.")
    with c2:
        st.markdown("**⚛️ Physics**")
        st.caption("Quantum, particle, astro, and more.")
    with c3:
        st.markdown("**🧬 Biology**")
        st.caption("Genomics, systems biology, life sciences.")
    with c4:
        st.markdown("**⚗️ Chemistry**")
        st.caption("Molecules, reactions, and materials.")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.markdown("**📐 Mathematics**")
        st.caption("Pure and applied mathematical research.")
    with c6:
        st.markdown("**🧠 Neuroscience**")
        st.caption("Brain, cognition, and neural systems.")
    with c7:
        st.markdown("**💊 Medicine**")
        st.caption("Clinical evidence and medical trials.")
    with c8:
        st.markdown("**💻 Computer Science**")
        st.caption("Systems, theory, and practical CS.")

