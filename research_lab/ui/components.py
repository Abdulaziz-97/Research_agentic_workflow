"""Reusable Streamlit UI components."""

import streamlit as st
from typing import List, Dict, Any, Optional
from config.settings import FIELD_DISPLAY_NAMES, FIELD_DESCRIPTIONS


def render_header():
    """Render the main header."""
    st.markdown(
        """
        <div style="
            padding: 0.75rem 1.1rem 1.4rem 1.1rem;
            margin-bottom: 0.75rem;
        ">
            <div class="rl-card" style="padding: 1.25rem 1.4rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; gap: 1rem;">
                    <div>
                        <div style="font-size: 0.8rem; letter-spacing: 0.14em; text-transform: uppercase; color: #a5b4fc;">
                            Multi-Agent Research Environment
                        </div>
                        <h1 style="
                            margin: 0.25rem 0 0.4rem 0;
                            background: linear-gradient(120deg, #e5e7eb, #c4b5fd, #67e8f9);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            font-size: 2.35rem;
                            font-weight: 700;
                        ">
                            🔬 Research Lab
                        </h1>
                        <p style="margin: 0; color: #9ca3af; font-size: 0.95rem;">
                            Orchestrate a team of specialized AI researchers to explore complex scientific questions.
                        </p>
                    </div>
                    <div style="
                        min-width: 210px;
                        border-radius: 1rem;
                        padding: 0.9rem 1rem;
                        background: radial-gradient(circle at top left, rgba(94, 234, 212, 0.15), rgba(15, 23, 42, 0.95));
                        border: 1px solid rgba(148, 163, 184, 0.4);
                    ">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.18em; color: #a5b4fc; margin-bottom: 0.25rem;">
                            Live Session Snapshot
                        </div>
                        <div style="display: flex; justify-content: space-between; gap: 0.75rem; font-size: 0.8rem; color: #e5e7eb;">
                            <div>
                                <div style="opacity: 0.75;">Domains active</div>
                                <div style="font-weight: 600;">up to 3</div>
                            </div>
                            <div>
                                <div style="opacity: 0.75;">Support agents</div>
                                <div style="font-weight: 600;">5 always-on</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    """Render the sidebar with navigation."""
    with st.sidebar:
        current_page = st.session_state.get("page", "home")

        st.markdown("### 🧭 Workspace")

        def _nav_button(label: str, page_key: str, icon: str):
            is_active = current_page == page_key
            button_label = f"{icon} {label}"
            if st.button(
                button_label,
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = page_key
                st.rerun()

        _nav_button("Home", "home", "🏠")
        _nav_button("Team Setup", "team_setup", "👥")

        if st.session_state.get("team_configured"):
            _nav_button("Research Session", "research", "🔬")

        st.markdown("---")

        if st.session_state.get("team_configured"):
            st.markdown("#### 📋 Current Team")
            with st.container():
                for field in st.session_state.get("selected_fields", []):
                    name = FIELD_DISPLAY_NAMES.get(field, field)
                    st.markdown(f"- **{name}**")

        st.markdown("---")
        st.caption(
            "Tip: Configure your team first, then move to the Research Session to start a live analysis."
        )


def render_chat_message(content: str, role: str = "assistant", agent_id: Optional[str] = None):
    """Render a chat message with modern chat bubbles."""
    is_user = role == "user"
    avatar = "🧑‍💻" if is_user else "🔬"
    label = "You" if is_user else (agent_id or "Research Team")

    with st.chat_message("user" if is_user else "assistant", avatar=avatar):
        st.markdown(f"**{label}**")
        bubble_class = "rl-chat-user" if is_user else "rl-chat-assistant"
        st.markdown(
            f'<div class="{bubble_class}">{content}</div>',
            unsafe_allow_html=True,
        )


def render_agent_status(agent_state):
    """Render an agent's status."""
    st.markdown(f"**{agent_state.display_name}**: {agent_state.status.value}")


def render_paper_card(paper, expanded: bool = False):
    """Render a paper card."""
    authors_str = ", ".join(paper.authors[:3])
    if len(paper.authors) > 3:
        authors_str += " et al."

    st.markdown(
        f"""
        <div class="rl-card-soft" style="margin-bottom: 0.6rem;">
            <div style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.25rem;">
                {paper.title}
            </div>
            <div style="font-size: 0.78rem; color: #9ca3af;">
                {authors_str} &nbsp;·&nbsp; {paper.source}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if expanded and paper.abstract:
        with st.expander("Abstract"):
            st.write(paper.abstract)


def render_field_selector(available_fields: List[str], selected_fields: List[str], max_selections: int = 3) -> List[str]:
    """Render field selection interface."""
    st.markdown("### Select Your Research Team")
    st.markdown(f"Choose up to {max_selections} research domains:")

    new_selections = selected_fields.copy()

    cols = st.columns(2)
    for idx, field in enumerate(available_fields):
        display_name = FIELD_DISPLAY_NAMES.get(field, field)
        description = FIELD_DESCRIPTIONS.get(field, "")

        is_selected = field in selected_fields
        can_select = is_selected or len(new_selections) < max_selections

        with cols[idx % 2]:
            with st.container():
                checkbox = st.checkbox(
                    f"**{display_name}**",
                    value=is_selected,
                    key=f"field_{field}",
                    disabled=(not can_select and not is_selected),
                )
                st.caption(description)

            if checkbox:
                if field not in new_selections:
                    new_selections.append(field)
            else:
                if field in new_selections:
                    new_selections.remove(field)

    return new_selections

