"""Team setup page for Research Lab."""

import streamlit as st
import uuid

from ui.components import render_header, render_sidebar, render_field_selector
from config.settings import RESEARCH_FIELDS, FIELD_DISPLAY_NAMES
from states.agent_state import TeamConfiguration


def render_team_setup_page():
    """Render the team setup page."""
    render_sidebar()
    render_header()

    st.markdown("## 👥 Configure Your Research Team")
    st.markdown(
        "Design a small, high-leverage research team. Choose up to **three** core domains that best match your question."
    )

    col_left, col_right = st.columns([1.7, 1.3])

    with col_left:
        current_selections = st.session_state.get("selected_fields", [])
        new_selections = render_field_selector(
            RESEARCH_FIELDS, current_selections, max_selections=3
        )
        st.session_state.selected_fields = new_selections

    with col_right:
        st.markdown("#### 🛠️ Support Agents (Always Available)")
        st.markdown(
            """
            <div class="rl-card-soft">
                <ul style="padding-left: 1rem; margin: 0; font-size: 0.9rem;">
                    <li><strong>📚 Literature Reviewer</strong> – synthesizes key papers.</li>
                    <li><strong>🔍 Methodology Critic</strong> – inspects methods and validity.</li>
                    <li><strong>✓ Fact Checker</strong> – verifies claims against sources.</li>
                    <li><strong>✍️ Writing Assistant</strong> – polishes explanations.</li>
                    <li><strong>🔗 Cross-Domain Synthesizer</strong> – connects ideas across fields.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(" ")

        if new_selections:
            st.markdown("#### 📋 Team Summary")
            st.markdown(f"**Domain Agents:** {len(new_selections)}")
            for field in new_selections:
                st.markdown(f"- {FIELD_DISPLAY_NAMES.get(field, field)}")

            st.markdown(" ")
            if st.button("✅ Confirm Team & Start Research", type="primary", use_container_width=True):
                team_config = TeamConfiguration(
                    team_id=str(uuid.uuid4()),
                    name="Research Team",
                    domain_agents=new_selections,
                    support_agents=[
                        "literature_reviewer",
                        "methodology_critic",
                        "fact_checker",
                        "writing_assistant",
                        "cross_domain_synthesizer",
                    ],
                )

                st.session_state.team_config = team_config
                st.session_state.team_configured = True
                st.session_state.messages = []
                st.session_state.page = "research"
                st.rerun()
        else:
            st.info("Select at least one domain to see a live team summary and continue.")

