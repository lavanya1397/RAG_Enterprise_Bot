import streamlit as st
import requests
from src.rag_pipeline import generate_answer

API_URL = "http://127.0.0.1:8000/ask"

st.set_page_config(page_title="AI Chatbot", layout="wide")

st.title("🤖 AI Chatbot with RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat inputstreamlit 
query = st.chat_input("Ask something...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = requests.post(API_URL, json={
                "query": query,
                "chat_history": st.session_state.messages
            })

            answer = response.json()["answer"]
            st.write(answer)

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )