"""
Enhanced Streamlit Chat Application for RM Assistant
Features: Multi-agent support, conversation history, compliance indicators
"""

import logging
import os
import streamlit as st
from datetime import datetime
from typing import Optional
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RM Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .compliance-badge {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
    }
    .warning-badge {
        background-color: #F59E0B;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
    }
    .agent-indicator {
        font-size: 0.75rem;
        color: #9CA3AF;
        margin-top: 0.25rem;
    }
    .tool-call {
        background-color: #F3F4F6;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    }
    .stChatMessage {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def get_user_info() -> dict:
    """Get user information from headers"""
    headers = st.context.headers if hasattr(st.context, 'headers') else {}
    return {
        "user_name": headers.get("X-Forwarded-Preferred-Username", "User"),
        "user_email": headers.get("X-Forwarded-Email", ""),
        "user_id": headers.get("X-Forwarded-User", "anonymous"),
    }


def query_endpoint(endpoint_name: str, messages: list, max_tokens: int = 500) -> dict:
    """Query the model serving endpoint"""
    try:
        from mlflow.deployments import get_deploy_client
        
        client = get_deploy_client('databricks')
        response = client.predict(
            endpoint=endpoint_name,
            inputs={'messages': messages, "max_tokens": max_tokens},
        )
        
        if "messages" in response:
            return response["messages"][-1]
        elif "choices" in response:
            return response["choices"][0]["message"]
        
        return {"content": "No response received", "role": "assistant"}
        
    except Exception as e:
        logger.error(f"Error querying endpoint: {e}")
        return {"content": f"Error: {str(e)}", "role": "assistant"}


def render_sidebar():
    """Render the sidebar with settings and info"""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Agent mode selection
        agent_mode = st.selectbox(
            "Agent Mode",
            ["Standard", "Multi-Agent", "Compliance Enhanced"],
            help="Select the agent orchestration mode"
        )
        
        # Model selection
        model_endpoint = st.selectbox(
            "Model",
            [
                "databricks-meta-llama-3-3-70b-instruct",
                "databricks-meta-llama-3-1-8b-instruct",
            ],
            help="Select the LLM endpoint"
        )
        
        st.divider()
        
        # Quick actions
        st.markdown("### 🚀 Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Customer Lookup", use_container_width=True):
                st.session_state.quick_action = "customer_lookup"
        with col2:
            if st.button("📊 Product Search", use_container_width=True):
                st.session_state.quick_action = "product_search"
        
        col3, col4 = st.columns(2)
        with col3:
            if st.button("✅ Compliance Check", use_container_width=True):
                st.session_state.quick_action = "compliance_check"
        with col4:
            if st.button("💱 Currency Convert", use_container_width=True):
                st.session_state.quick_action = "currency_convert"
        
        st.divider()
        
        # Conversation management
        st.markdown("### 💬 Conversation")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📥 Export Chat", use_container_width=True):
            export_chat()
        
        st.divider()
        
        # User info
        user_info = get_user_info()
        st.markdown("### 👤 User Info")
        st.text(f"Name: {user_info['user_name']}")
        
        st.divider()
        
        # Compliance status
        st.markdown("### 🛡️ Compliance Status")
        st.markdown('<span class="compliance-badge">✓ Active</span>', unsafe_allow_html=True)
        st.caption("All interactions are logged for audit purposes")
        
        return agent_mode, model_endpoint


def export_chat():
    """Export chat history as JSON"""
    if st.session_state.messages:
        chat_data = {
            "exported_at": datetime.now().isoformat(),
            "messages": st.session_state.messages
        }
        st.download_button(
            label="Download Chat",
            data=json.dumps(chat_data, indent=2),
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def render_message(message: dict):
    """Render a chat message with enhanced formatting"""
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Show tool calls if present
        if "tool_calls" in message:
            with st.expander("🔧 Tool Calls", expanded=False):
                for tool_call in message["tool_calls"]:
                    st.markdown(f"""
                    <div class="tool-call">
                        <strong>{tool_call.get('name', 'Unknown Tool')}</strong><br>
                        {json.dumps(tool_call.get('arguments', {}), indent=2)}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Show agent info if available
        if "agent" in message:
            st.markdown(f'<div class="agent-indicator">🤖 {message["agent"]}</div>', 
                       unsafe_allow_html=True)


def handle_quick_action(action: str) -> Optional[str]:
    """Handle quick action buttons"""
    prompts = {
        "customer_lookup": "Please help me look up a customer. What information do you need?",
        "product_search": "I need to search for suitable investment products. What criteria should I consider?",
        "compliance_check": "I need to perform a compliance check. What type of check do you want to run?",
        "currency_convert": "I need to convert currencies. Please provide the amount, source currency, and target currency."
    }
    return prompts.get(action)


def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🏦 RM Assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-powered assistant for Relationship Managers</div>', 
                unsafe_allow_html=True)
    
    # Render sidebar and get settings
    agent_mode, model_endpoint = render_sidebar()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "quick_action" not in st.session_state:
        st.session_state.quick_action = None
    
    # Get endpoint from environment
    endpoint_name = os.getenv("SERVING_ENDPOINT")
    
    if not endpoint_name:
        st.warning("⚠️ SERVING_ENDPOINT environment variable not set. Using demo mode.")
        endpoint_name = "demo"
    
    # Main chat area
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            render_message(message)
    
    # Handle quick actions
    if st.session_state.quick_action:
        quick_prompt = handle_quick_action(st.session_state.quick_action)
        if quick_prompt:
            st.session_state.messages.append({"role": "user", "content": quick_prompt})
            st.session_state.quick_action = None
            st.rerun()
    
    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if endpoint_name == "demo":
                    # Demo mode response
                    response_content = f"""I understand you're asking about: **{prompt}**

In demo mode, I cannot access the actual backend services. However, I can help you with:

- 📋 **Customer Information**: Look up customer profiles, risk ratings, and account details
- 📊 **Product Recommendations**: Search and compare unit trusts based on risk profiles
- ✅ **Compliance Checks**: Verify KYC status, sanctions screening, and suitability
- 💱 **Currency Conversion**: Convert amounts between different currencies
- 📄 **Document Search**: Find relevant product documentation

Please configure the SERVING_ENDPOINT environment variable to connect to the actual agent backend."""
                else:
                    # Query actual endpoint
                    response = query_endpoint(
                        endpoint_name=endpoint_name,
                        messages=st.session_state.messages,
                        max_tokens=500
                    )
                    response_content = response.get("content", "No response received")
                
                st.markdown(response_content)
        
        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_content,
            "agent": agent_mode
        })
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🔒 Secure & Compliant")
    with col2:
        st.caption(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with col3:
        st.caption("Powered by AI")


if __name__ == "__main__":
    main()
