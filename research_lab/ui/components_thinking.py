"""Gemini-style thinking/reasoning display components."""

import streamlit as st
from typing import List, Dict, Any
from datetime import datetime


def render_thinking_display(thinking_steps: List[Dict[str, Any]], agent_name: str = "Research Team"):
    """
    Render a Gemini-style thinking display showing step-by-step reasoning.
    
    Args:
        thinking_steps: List of thinking step dictionaries
        agent_name: Name of the agent/team
    """
    if not thinking_steps:
        return
    
    st.markdown("---")
    st.markdown("### 🧠 Research Team Thinking")
    
    with st.container():
        for i, step in enumerate(thinking_steps):
            _render_thinking_step(step, i, len(thinking_steps))


def _render_thinking_step(step: Dict[str, Any], index: int, total: int):
    """Render a single thinking step in Gemini style."""
    
    step_id = step.get("step_id", index + 1)
    agent_name = step.get("agent_name", "Agent")
    reasoning = step.get("reasoning", "")
    tool_calls = step.get("tool_calls", [])
    content = step.get("content", "")
    timestamp = step.get("timestamp", "")
    
    # Create expandable card
    with st.expander(
        f"**Step {step_id}** | {agent_name} | {'🔧 Using tools' if tool_calls else '💭 Reasoning'}",
        expanded=(index == 0)  # Expand first step by default
    ):
        # Step header with visual indicator
        col1, col2 = st.columns([3, 1])
        with col1:
            if tool_calls:
                st.markdown(f"**{agent_name}** is using tools to gather information...")
            elif reasoning:
                st.markdown(f"**{agent_name}** is thinking...")
            else:
                st.markdown(f"**{agent_name}** is processing...")
        
        with col2:
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.caption(dt.strftime("%H:%M:%S"))
                except:
                    pass
        
        # Reasoning section (Gemini-style)
        if reasoning:
            st.markdown("**💭 Reasoning:**")
            with st.container():
                st.markdown(f'<div style="background: rgba(66, 133, 244, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #4285F4; margin: 8px 0;">{reasoning}</div>', unsafe_allow_html=True)
        
        # Tool calls section
        if tool_calls:
            st.markdown("**🔧 Tool Usage:**")
            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "unknown")
                tool_args = tool_call.get("args", {})
                
                with st.container():
                    # Tool name badge
                    st.markdown(f'<div style="display: inline-block; background: #34A853; color: white; padding: 4px 12px; border-radius: 16px; font-size: 0.85em; margin: 4px 0;">{tool_name}</div>', unsafe_allow_html=True)
                    
                    # Tool arguments (collapsible)
                    if tool_args:
                        with st.expander("View tool parameters", expanded=False):
                            st.json(tool_args)
        
        # Content/output section
        if content and content.strip():
            st.markdown("**📝 Output:**")
            st.markdown(content)
        
        # Visual separator (except for last step)
        if index < total - 1:
            st.markdown('<div style="height: 1px; background: linear-gradient(to right, transparent, rgba(128,128,128,0.3), transparent); margin: 12px 0;"></div>', unsafe_allow_html=True)


def render_thinking_timeline(thinking_steps: List[Dict[str, Any]]):
    """
    Render thinking steps as a vertical timeline (alternative view).
    """
    if not thinking_steps:
        return
    
    st.markdown("---")
    st.markdown("### 🧠 Research Timeline")
    
    for i, step in enumerate(thinking_steps):
        step_id = step.get("step_id", i + 1)
        agent_name = step.get("agent_name", "Agent")
        reasoning = step.get("reasoning", "")
        tool_calls = step.get("tool_calls", [])
        
        # Timeline item
        col1, col2 = st.columns([0.1, 0.9])
        
        with col1:
            # Timeline dot
            if tool_calls:
                st.markdown("🔧")
            elif reasoning:
                st.markdown("💭")
            else:
                st.markdown("⚪")
        
        with col2:
            st.markdown(f"**Step {step_id}** | {agent_name}")
            if reasoning:
                st.caption(reasoning[:100] + "..." if len(reasoning) > 100 else reasoning)
            if tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                st.caption(f"Tools: {', '.join(tool_names)}")


def render_agent_thinking_summary(thinking_steps: List[Dict[str, Any]]):
    """
    Render a summary view of agent thinking (compact).
    """
    if not thinking_steps:
        return
    
    # Group by agent
    agent_steps = {}
    for step in thinking_steps:
        agent_name = step.get("agent_name", "Unknown")
        if agent_name not in agent_steps:
            agent_steps[agent_name] = []
        agent_steps[agent_name].append(step)
    
    st.markdown("---")
    st.markdown("### 🧠 Thinking Summary")
    
    for agent_name, steps in agent_steps.items():
        with st.expander(f"**{agent_name}** ({len(steps)} steps)", expanded=False):
            total_tools = sum(len(s.get("tool_calls", [])) for s in steps)
            has_reasoning = any(s.get("reasoning") for s in steps)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Steps", len(steps))
            with col2:
                st.metric("Tool Calls", total_tools)
            with col3:
                st.metric("Reasoning", "Yes" if has_reasoning else "No")
            
            # Show first step preview
            if steps:
                first_step = steps[0]
                if first_step.get("reasoning"):
                    st.markdown("**First Reasoning:**")
                    st.caption(first_step["reasoning"][:200] + "..." if len(first_step["reasoning"]) > 200 else first_step["reasoning"])

