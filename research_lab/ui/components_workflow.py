"""UI components for displaying workflow steps and node outputs."""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime


# Node display names and icons
NODE_INFO = {
    "init": {
        "name": "Initialization",
        "icon": "🚀",
        "description": "Workflow setup and initialization"
    },
    "routing": {
        "name": "Query Routing",
        "icon": "🧭",
        "description": "Analyzing query and selecting domain agents"
    },
    "domain_research": {
        "name": "Domain Research",
        "icon": "🔬",
        "description": "Domain experts conducting research"
    },
    "support_review": {
        "name": "Support Review",
        "icon": "🔍",
        "description": "Support agents reviewing findings"
    },
    "synthesis": {
        "name": "Synthesis",
        "icon": "📝",
        "description": "Synthesizing findings into research brief"
    },
    "complete": {
        "name": "Complete",
        "icon": "✅",
        "description": "Workflow completed"
    }
}


def render_workflow_steps(node_outputs: Dict[str, Dict[str, Any]]):
    """
    Render workflow steps showing each node's output.
    
    Args:
        node_outputs: Dictionary of node outputs from workflow state
    """
    if not node_outputs:
        return
    
    st.markdown("---")
    st.markdown("### 🔄 Workflow Steps")
    st.markdown("""
    <div style='font-size: 0.8rem; color: #606070; margin-bottom: 1rem;'>
        Step-by-step breakdown of the research workflow
    </div>
    """, unsafe_allow_html=True)
    
    # Define node order
    node_order = ["init", "routing", "domain_research", "support_review", "synthesis", "complete"]
    
    for i, node_name in enumerate(node_order):
        if node_name not in node_outputs:
            continue
        
        node_data = node_outputs[node_name]
        node_info = NODE_INFO.get(node_name, {"name": node_name, "icon": "📌", "description": ""})
        
        # Determine status color
        status = node_data.get("status", "unknown")
        if status == "complete":
            status_color = "#00d4aa"
            status_icon = "✅"
        elif status == "in_progress":
            status_color = "#0ea5e9"
            status_icon = "⏳"
        else:
            status_color = "#606070"
            status_icon = "⏸️"
        
        # Create expandable card for each step
        with st.expander(
            f"**{node_info['icon']} {node_info['name']}** | {status_icon} {status.title()}",
            expanded=(i == len(node_outputs) - 1)  # Expand last step
        ):
            # Step header
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{node_info['description']}**")
            with col2:
                timestamp = node_data.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        st.caption(dt.strftime("%H:%M:%S"))
                    except:
                        pass
            
            # Main output
            output = node_data.get("output", "")
            if output:
                st.markdown("**Output:**")
                st.info(output)
            
            # Details section
            details = node_data.get("details", {})
            if details:
                st.markdown("**Details:**")
                
                # Special formatting for different node types
                if node_name == "routing":
                    if "selected_agents" in details:
                        st.markdown("**Selected Agents:**")
                        for agent in details["selected_agents"]:
                            st.markdown(f"- {agent}")
                    if "reasoning" in details and details["reasoning"]:
                        st.markdown("**Routing Reasoning:**")
                        st.markdown(f'<div style="background: rgba(66, 133, 244, 0.1); padding: 12px; border-radius: 8px; border-left: 3px solid #4285F4; margin: 8px 0;">{details["reasoning"]}</div>', unsafe_allow_html=True)
                
                elif node_name == "domain_research":
                    if "agents" in details:
                        st.markdown("**Agent Results:**")
                        for agent_result in details["agents"]:
                            with st.container():
                                st.markdown(f"""
                                <div style='background: rgba(0, 212, 170, 0.05); padding: 12px; border-radius: 8px; margin: 8px 0; border-left: 3px solid #00d4aa;'>
                                    <strong>{agent_result.get('field', 'Unknown')}</strong><br>
                                    <span style='font-size: 0.85rem; color: #a0a0b0;'>
                                        Papers: {agent_result.get('papers_found', 0)} | 
                                        Confidence: {agent_result.get('confidence', 0):.0%} | 
                                        Insights: {agent_result.get('insights_count', 0)}
                                    </span>
                                    <div style='margin-top: 8px; font-size: 0.9rem; color: #a0a0b0;'>
                                        {agent_result.get('summary', '')}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                    if "total_papers" in details:
                        st.metric("Total Papers Found", details["total_papers"])
                
                elif node_name == "synthesis":
                    if "response_length" in details:
                        st.metric("Response Length", f"{details['response_length']:,} characters")
                    if "papers_referenced" in details:
                        st.metric("Papers Referenced", details["papers_referenced"])
                    if "domains_synthesized" in details:
                        st.markdown(f"**Domains:** {details['domains_synthesized']}")
                
                elif node_name == "complete":
                    if "total_papers" in details:
                        st.metric("Total Papers", details["total_papers"])
                    if "domains_consulted" in details:
                        st.metric("Domains Consulted", details["domains_consulted"])
                    if "avg_confidence" in details:
                        st.metric("Average Confidence", f"{details['avg_confidence']:.0%}")
                    if "execution_time" in details:
                        st.metric("Execution Time", details["execution_time"])
                
                else:
                    # Generic details display
                    st.json(details)
            
            # Visual separator
            if i < len([n for n in node_order if n in node_outputs]) - 1:
                st.markdown("---")


def render_workflow_timeline(node_outputs: Dict[str, Dict[str, Any]]):
    """
    Render a compact timeline view of workflow steps.
    
    Args:
        node_outputs: Dictionary of node outputs from workflow state
    """
    if not node_outputs:
        return
    
    st.markdown("### ⏱️ Workflow Timeline")
    
    node_order = ["init", "routing", "domain_research", "support_review", "synthesis", "complete"]
    
    for i, node_name in enumerate(node_order):
        if node_name not in node_outputs:
            continue
        
        node_data = node_outputs[node_name]
        node_info = NODE_INFO.get(node_name, {"name": node_name, "icon": "📌"})
        
        status = node_data.get("status", "unknown")
        timestamp = node_data.get("timestamp", "")
        
        # Timeline item
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            if i < len([n for n in node_order if n in node_outputs]) - 1:
                st.markdown("│")
                st.markdown("├─")
            else:
                st.markdown("│")
                st.markdown("└─")
        
        with col2:
            status_icon = "✅" if status == "complete" else "⏳" if status == "in_progress" else "⏸️"
            st.markdown(f"{node_info['icon']} **{node_info['name']}** {status_icon}")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    st.caption(dt.strftime("%H:%M:%S"))
                except:
                    pass
        
        with col3:
            output = node_data.get("output", "")
            if output:
                st.caption(output[:50] + "..." if len(output) > 50 else output)

