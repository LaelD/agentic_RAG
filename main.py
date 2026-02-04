from typing import Any, Dict

import streamlit as st

from backend.core import run_llm

# Page configuration
st.set_page_config(
    page_title="🌱 Smart Agriculture Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stChatMessage {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
    }
    h1 {
        color: #2d5016;
        text-align: center;
        padding: 20px 0;
        font-weight: 600;
    }
    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 18px;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100)
    st.title("🌱 About")
    st.markdown("""
    **Smart Agriculture Assistant** helps you find answers about:
    
    🌾 **Precision Agriculture**  
    📊 **Data Analytics**  
    🚜 **IoT & Sensors**  
    💧 **Irrigation Systems**  
    🤖 **AI in Farming**
    
    
    **Powered by:**  
    - 🧠 OpenAI GPT
    - 📚 LangChain
    - 🔍 Pinecone Vector DB
    """)
    
 

# Main content
st.title("🌱 Smart Agriculture Assistant")
st.markdown('<p class="subtitle">Your AI-powered guide to modern farming technology</p>', unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I'm your Smart Agriculture Assistant. Ask me anything about modern farming, precision agriculture, IoT sensors, or agricultural technology!"
        }
    ]

# Display chat messages
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

# Chat input
prompt = st.chat_input("💬 Ask a question about smart agriculture...")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("🔍 Retrieving relevant documentation and generating answer..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "❌ No answer was returned."

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            
        except Exception as e:
            error_msg = f"⚠️ **Error:** Failed to generate a response.\n\n```\n{str(e)}\n```"
            st.error("Failed to generate a response.")
            st.exception(e)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

