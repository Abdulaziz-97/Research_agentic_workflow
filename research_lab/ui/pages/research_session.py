"""Research session page for Research Lab."""

import streamlit as st
from typing import Optional

from ui.components import render_header, render_sidebar, render_chat_message
from config.settings import FIELD_DISPLAY_NAMES
from graphs.research_graph import create_research_graph


def get_or_create_graph():
    """Get or create the research graph."""
    if "research_graph" not in st.session_state or st.session_state.research_graph is None:
        team_config = st.session_state.get("team_config")
        if team_config:
            st.session_state.research_graph = create_research_graph(team_config)
    return st.session_state.get("research_graph")


def render_research_session_page():
    """Render the research session page."""
    render_sidebar()
    render_header()

    if not st.session_state.get("team_configured"):
        st.warning("Please configure your team first!")
        if st.button("Go to Team Setup"):
            st.session_state.page = "team_setup"
            st.rerun()
        return

    st.markdown("## 🔬 Research Session")

    team_config = st.session_state.get("team_config")
    if team_config:
        team_display = " | ".join(
            [FIELD_DISPLAY_NAMES.get(f, f) for f in team_config.domain_agents]
        )
        st.info(f"**Active team:** {team_display}")

    col_chat, col_side = st.columns([2.1, 1.2])

    # Chat column
    with col_chat:
        messages = st.session_state.get("messages", [])

        chat_container = st.container()
        if not messages:
            with chat_container:
                st.markdown(
                    "*Start your research by asking a question in the chat input below.*"
                )
        else:
            with chat_container:
                for msg in messages:
                    render_chat_message(
                        content=msg.get("content", ""),
                        role=msg.get("role", "assistant"),
                        agent_id=msg.get("agent_id"),
                    )

        # Chat-style input
        user_query = st.chat_input("Ask a complex research question...")

        if user_query:
            messages = st.session_state.get("messages", [])
            messages.append({"role": "user", "content": user_query})
            st.session_state.messages = messages

            with st.spinner("🔬 Your research team is investigating..."):
                try:
                    graph = get_or_create_graph()
                    if graph:
                        result = graph.run_sync(user_query, thread_id="streamlit_session")
                        final_response = result.get("final_response", "")

                        if final_response:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": final_response,
                                    "agent_id": "Research Team",
                                }
                            )
                        else:
                            messages.append(
                                {
                                    "role": "assistant",
                                    "content": "Could not generate a response.",
                                    "agent_id": "System",
                                }
                            )

                        st.session_state.messages = messages
                        st.session_state.last_results = result
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Error: {str(e)}",
                            "agent_id": "System",
                        }
                    )
                    st.session_state.messages = messages

            st.rerun()

    # Side column: session context & sources
    with col_side:
        st.markdown("#### 📊 Session Overview")
        st.markdown(
            """
            <div class="rl-card-soft">
                <p style="margin: 0; font-size: 0.86rem; color: #d1d5db;">
                    Each question triggers a coordinated analysis from your domain agents and support agents.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(" ")

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_results = None
            st.rerun()

        last_results = st.session_state.get("last_results")
        if last_results:
            st.markdown("#### 📚 Latest Sources")
            papers = last_results.get("papers") or last_results.get("domain_papers")
            if papers:
                for paper in papers[:5]:
                    title = getattr(paper, "title", None) or paper.get("title", "")
                    url = getattr(paper, "url", None) or paper.get("url", "")
                    if url:
                        st.markdown(f"- [{title}]({url})")
                    else:
                        st.markdown(f"- {title}")

