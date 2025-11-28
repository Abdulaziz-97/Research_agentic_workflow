"""Research session page - Main research command center."""

import json
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
from ui.components_thinking import render_thinking_display, render_thinking_timeline
from config.settings import FIELD_DISPLAY_NAMES, settings
from graphs.research_graph import create_research_graph


def get_or_create_graph():
    """Get or create the research graph."""
    # Ensure key manager is initialized before creating graph
    from config.key_manager import get_key_manager, initialize_key_manager
    
    key_manager = get_key_manager()
    if not key_manager:
        # Initialize if not already done
        initialize_key_manager(
            keys=settings.llm_api_keys,
            base_url=settings.openai_base_url if settings.llm_provider == "openai" and settings.openai_base_url else None,
            model=settings.llm_model,
            provider=settings.llm_provider
        )
    
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
                        
                        # Extract thinking trail
                        thinking_trail = []
                        if isinstance(result, dict):
                            thinking_trail = result.get("thinking_trail", []) or []
                        
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
                                "papers": all_papers,
                                "thinking_trail": thinking_trail
                            })
                            st.session_state.messages = messages
                            # Store full result including thinking_trail
                            result_dict = {}
                            if isinstance(result, dict):
                                result_dict = dict(result)
                            elif hasattr(result, '__dict__'):
                                result_dict = result.__dict__.copy()
                            elif hasattr(result, 'keys'):
                                result_dict = dict(result)
                            
                            # Always ensure it's a dict and include thinking_trail
                            if not isinstance(result_dict, dict):
                                result_dict = {}
                            result_dict["thinking_trail"] = thinking_trail
                            st.session_state.last_results = result_dict
                            
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
                    import asyncio
                    error_str = str(e)
                    error_details = traceback.format_exc()
                    
                    # AUTOMATIC KEY ROTATION - Try next key immediately if this is a key error
                    from config.key_manager import get_key_manager
                    key_manager = get_key_manager()
                    
                    # Check if this is a key-related error
                    is_key_error = any(term in error_str.lower() for term in [
                        "insufficient", "budget", "rate limit", "429", "401", "authentication"
                    ])
                    
                    # AUTOMATIC RETRY: Try ALL available keys until one works
                    if key_manager and is_key_error:
                        available_keys = key_manager.get_available_keys()
                        max_auto_retries = min(6, len(available_keys))  # Try up to 6 keys automatically
                        
                        if len(available_keys) > 1:
                            # Mark first failed key
                            current_key = key_manager.get_current_key()
                            if current_key:
                                try:
                                    asyncio.run(key_manager.mark_key_failed(current_key, error_str))
                                except:
                                    pass
                            
                            # Try each available key automatically
                            for retry_num in range(max_auto_retries):
                                # Rotate to next key
                                key_manager._rotate_to_next_available()
                                new_key = key_manager.get_current_key()
                                
                                if not new_key:
                                    break
                                
                                # Try again with new key automatically
                                try:
                                    with status:
                                        st.write(f"🔄 Auto-retry {retry_num + 1}/{max_auto_retries}: Trying key {new_key[:30]}...")
                                    
                                    # Clear graph cache to force recreation with new key
                                    if "research_graph" in st.session_state:
                                        del st.session_state["research_graph"]
                                    
                                    graph = get_or_create_graph()
                                    if graph:
                                        result = graph.run_sync(query, thread_id=f"session_{datetime.now().timestamp()}")
                                        
                                        # Check if successful
                                        final_response = result.get("final_response", "") if isinstance(result, dict) else ""
                                        if final_response:
                                            # SUCCESS! Process result
                                            domain_results = result.get("domain_results", []) if isinstance(result, dict) else []
                                            research_stats = result.get("research_stats", {}) if isinstance(result, dict) else {}
                                            thinking_trail = result.get("thinking_trail", []) if isinstance(result, dict) else []
                                            
                                            all_papers = []
                                            for dr in domain_results:
                                                try:
                                                    if hasattr(dr, 'papers'):
                                                        all_papers.extend(dr.papers)
                                                    elif isinstance(dr, dict) and 'papers' in dr:
                                                        all_papers.extend(dr['papers'])
                                                except:
                                                    pass
                                            
                                            messages.append({
                                                "role": "assistant",
                                                "content": final_response,
                                                "timestamp": datetime.now().isoformat(),
                                                "stats": research_stats,
                                                "papers": all_papers,
                                                "thinking_trail": thinking_trail
                                            })
                                            st.session_state.messages = messages
                                            result_dict = dict(result) if hasattr(result, 'keys') else result
                                            if isinstance(result_dict, dict):
                                                result_dict["thinking_trail"] = thinking_trail
                                            st.session_state.last_results = result_dict if result_dict else {}
                                            
                                            history = st.session_state.get("research_history", [])
                                            history.append({
                                                "query": query,
                                                "timestamp": datetime.now().isoformat(),
                                                "papers_count": len(all_papers)
                                            })
                                            st.session_state.research_history = history
                                            
                                            status.update(label=f"✅ Success with key {retry_num + 1}!", state="complete", expanded=False)
                                            st.success(f"✅ Successfully completed with key {retry_num + 1}!")
                                            st.rerun()
                                            return
                                        else:
                                            # No response, mark as failed and try next key
                                            try:
                                                asyncio.run(key_manager.mark_key_failed(new_key, "No response generated"))
                                            except:
                                                pass
                                            continue
                                except Exception as retry_error:
                                    # This key also failed, mark it and try next
                                    error_str_retry = str(retry_error)
                                    is_still_key_error = any(term in error_str_retry.lower() for term in ["insufficient", "budget"])
                                    
                                    if is_still_key_error and new_key:
                                        try:
                                            asyncio.run(key_manager.mark_key_failed(new_key, error_str_retry))
                                        except:
                                            pass
                                        continue
                                    else:
                                        # Different error, stop retrying
                                        break
                            
                            # All auto-retries failed
                            error_str = f"{error_str} (Auto-tried {max_auto_retries} keys)"
                    
                    # Check key manager status for display
                    key_status_info = ""
                    if key_manager:
                        status = key_manager.get_status()
                        available = status.get("available_keys", 0)
                        total = status.get("total_keys", 0)
                        current = status.get("current_key", "Unknown")
                        
                        key_status_info = f"""
                        
                        **Key Rotation Status:**
                        - Total keys configured: {total}
                        - Available keys: {available}
                        - Current key: {current}
                        - Keys tried: {sum(1 for k in status['keys'] if k.get('failure_count', 0) > 0)} failed
                        - Total failures: {sum(k.get('failure_count', 0) for k in status['keys'])}
                        """
                        
                        if available == 0:
                            key_status_info += "\n⚠️ **All keys have been exhausted or disabled.**"
                            key_status_info += "\n\n**Options:**"
                            key_status_info += "\n1. Wait 1 hour for keys to auto-recover (budget errors disable for 1 hour)"
                            key_status_info += "\n2. Add credits to your keys"
                            key_status_info += "\n3. Click 'Reset Keys' below to force re-enable all keys (if you've added credits)"
                        elif available < total:
                            key_status_info += f"\n✅ **{available} key(s) still available - system will retry automatically.**"
                    
                    # Provider-specific metadata for support copy
                    provider_name = settings.provider_display_name
                    env_var_name = "GEMINI_API_KEY" if settings.llm_provider == "gemini" else "OPENAI_API_KEY"
                    billing_url = "https://aistudio.google.com/app/apikey" if settings.llm_provider == "gemini" else "https://platform.openai.com/account/billing"
                    keys_url = "https://aistudio.google.com/app/apikey" if settings.llm_provider == "gemini" else "https://platform.openai.com/api-keys"
                    limits_url = "https://ai.google.dev/gemini-api/docs/rate-limits" if settings.llm_provider == "gemini" else "https://platform.openai.com/account/limits"
                    env_tip_example = f"{env_var_name}=key1,key2,key3,key4"
                    alt_endpoint_note = ""
                    if settings.llm_provider == "openai":
                        alt_endpoint_note = "\n**Alternative:** If using a custom endpoint (Vocareum), check that endpoint's billing status."
                    
                    # Check for specific API errors
                    if "Insufficient budget" in error_str or "budget" in error_str.lower() or "insufficient" in error_str.lower():
                        # Check if we should retry with next key
                        should_retry = False
                        retry_message = ""
                        
                        if key_manager:
                            available = key_manager.get_available_keys()
                            if len(available) > 0:
                                should_retry = True
                                retry_message = f"\n\n**🔄 Automatic Retry Available:** {len(available)} key(s) still available. Click 'Retry with Next Key' below to continue."
                        
                        error_message = f"""
                        **⚠️ API Billing Error: Insufficient Credits**
                        
                        Your {provider_name} account has insufficient credits or has hit a spending limit.
                        {key_status_info}
                        
                        **To fix this:**
                        1. Check your billing dashboard: {billing_url}
                        2. Add credits or increase your spending limit
                        3. Verify your API key has sufficient permissions
                        4. **If you have multiple keys in .env**: The system will automatically try the next key on retry
                        
                        {alt_endpoint_note}
                        {retry_message}
                        
                        **💡 Tip:** Update your `.env` file with multiple keys (comma-separated) for automatic rotation:
                        ```
                        {env_tip_example}
                        ```
                        """
                        st.error(error_message)
                        
                        # Add retry/reset buttons
                        col1, col2, col3 = st.columns([1, 1, 3])
                        
                        if should_retry:
                            with col1:
                                if st.button("🔄 Retry with Next Key", type="primary", use_container_width=True):
                                    # Rotate to next key and retry
                                    if key_manager:
                                        # Mark current key as failed and rotate
                                        current_key = key_manager.get_current_key()
                                        if current_key:
                                            import asyncio
                                            asyncio.run(key_manager.mark_key_failed(current_key, "Manual retry - insufficient credits"))
                                        key_manager._rotate_to_next_available()
                                    
                                    # Clear the error message and retry
                                    st.session_state.last_error = None
                                    st.rerun()
                        
                        # Add reset button if all keys are disabled
                        if key_manager:
                            available = key_manager.get_available_keys()
                            if len(available) == 0:
                                with col2:
                                    if st.button("🔄 Reset All Keys", type="secondary", use_container_width=True):
                                        # Force reset all keys
                                        key_manager.force_reset_all_keys()
                                        st.success("✅ All keys reset! Try your query again.")
                                        st.rerun()
                        
                        messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "timestamp": datetime.now().isoformat()
                        })
                    elif "401" in error_str or "authentication" in error_str.lower():
                        error_message = f"""
                        **🔑 API Authentication Error**
                        
                        Your {provider_name} API key is invalid or expired.
                        
                        **To fix this:**
                        1. Check your `.env` file has the correct `{env_var_name}`
                        2. Verify the key is active at {keys_url}
                        """
                        if settings.llm_provider == "openai":
                            error_message += "\n3. If using a custom endpoint, verify `OPENAI_BASE_URL` is correct"
                        st.error(error_message)
                        messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "timestamp": datetime.now().isoformat()
                        })
                    elif "rate limit" in error_str.lower() or "429" in error_str:
                        error_message = f"""
                        **⏱️ Rate Limit Exceeded**
                        
                        You've hit the API rate limit. Please wait a moment and try again.
                        
                        **Tips:**
                        - Wait 1-2 minutes before retrying
                        - Consider using fewer domain agents (1-2 instead of 3)
                        - Review provider rate limits: {limits_url}
                        """
                        st.warning(error_message)
                        messages.append({
                            "role": "assistant",
                            "content": error_message,
                            "timestamp": datetime.now().isoformat()
                        })
                    else:
                        st.error(f"Research Error: {error_str}")
                        messages.append({
                            "role": "assistant",
                            "content": f"An error occurred during research: {error_str}",
                            "timestamp": datetime.now().isoformat()
                        })
                    
                    st.session_state.messages = messages
                    status.update(label=f"❌ Error occurred", state="error", expanded=False)
            
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
                    
                    # Display thinking/reasoning steps (Gemini-style)
                    # First check message for thinking_trail
                    thinking_trail = msg.get("thinking_trail", [])
                    
                    # Fallback to last_results (with None check)
                    if not thinking_trail:
                        last_results = st.session_state.get("last_results")
                        if last_results and isinstance(last_results, dict):
                            thinking_trail = last_results.get("thinking_trail", [])
                            
                            # Also check if thinking steps are in domain results
                            if not thinking_trail:
                                domain_results = last_results.get("domain_results", [])
                                if domain_results:
                                    for dr in domain_results:
                                        if isinstance(dr, dict) and "thinking_steps" in dr:
                                            thinking_trail.extend(dr.get("thinking_steps", []))
                                        elif hasattr(dr, 'thinking_steps'):
                                            thinking_trail.extend(dr.thinking_steps)
                    
                    if thinking_trail:
                        render_thinking_display(thinking_trail, "Research Team")
    
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

            # Knowledge Graph Context
            knowledge = last_results.get("knowledge_context", {})
            if knowledge and knowledge.get("nodes"):
                st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)
                with st.expander("📊 Knowledge Graph Path", expanded=False):
                    st.markdown(f"**Path Summary:** {knowledge.get('summary','')}")
                    if knowledge.get("path"):
                        path_str = " → ".join([n.get("label", n.get("id", "")) for n in knowledge.get("nodes", [])[:10]])
                        st.markdown(f"**Concepts:** {path_str}")
                    st.markdown(f"**Nodes:** {len(knowledge.get('nodes', []))} | **Edges:** {len(knowledge.get('edges', []))}")

            # Ontologist Output
            ontology = last_results.get("ontology_blueprint")
            if ontology:
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                with st.expander("🧠 Ontologist Hypothesis Blueprint", expanded=False):
                    if isinstance(ontology, dict) and "raw" not in ontology:
                        st.json(ontology, expanded=True)
                    else:
                        st.code(str(ontology)[:2000], language="json")

            # Scientist I Output
            scientist_one = last_results.get("scientist_proposal", {}).get("markdown")
            if scientist_one:
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                with st.expander("🔬 Scientist I: Research Proposal", expanded=False):
                    st.markdown(scientist_one, unsafe_allow_html=True)

            # Scientist II Output
            scientist_two = last_results.get("scientist_expansion", {}).get("markdown")
            if scientist_two:
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                with st.expander("⚗️ Scientist II: Quantitative Deep Dive", expanded=False):
                    st.markdown(scientist_two, unsafe_allow_html=True)

            # Critic Output
            critic = last_results.get("critic_feedback")
            if critic:
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                with st.expander("🔍 Critic: Critical Assessment", expanded=False):
                    st.markdown(critic, unsafe_allow_html=True)

            # Planner Output
            planner = last_results.get("planner_plan", {}).get("markdown")
            if planner:
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                with st.expander("📋 Planner: Actionable Roadmap", expanded=False):
                    st.markdown(planner, unsafe_allow_html=True)

            # Novelty Assessment
            novelty = last_results.get("novelty_report")
            if novelty:
                st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
                with st.expander("✨ Novelty Assessment", expanded=True):
                    if isinstance(novelty, dict):
                        if "novelty_score" in novelty:
                            score = novelty.get("novelty_score", 0.0)
                            st.metric("Novelty Score", f"{score:.2f}", delta=f"{score*100:.0f}%")
                        if "overlapping_papers" in novelty and novelty["overlapping_papers"]:
                            st.markdown("**Overlapping Papers:**")
                            for paper in novelty["overlapping_papers"][:5]:
                                st.markdown(f"- **{paper.get('title', 'Unknown')}** ({paper.get('year', 'N/A')})")
                                st.caption(f"Similarity: {paper.get('similarity', 'N/A')} - {paper.get('overlap_description', '')}")
                        if "summary" in novelty:
                            st.markdown(f"**Assessment:** {novelty['summary']}")
                        if "recommendations" in novelty:
                            st.markdown(f"**Recommendations:** {novelty['recommendations']}")
                    else:
                        st.json(novelty, expanded=False)
        
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
