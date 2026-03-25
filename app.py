import streamlit as st
import requests

API_URL = "https://rag-enterprise-bot.onrender.com/ask"

st.set_page_config(page_title="AI Chatbot", layout="wide")

st.title("🤖 AI Chatbot with RAG")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
query = st.chat_input("Ask something...")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = requests.post(API_URL, json={
                    "query": query,
                    "chat_history": st.session_state.messages
                })

                if response.status_code == 200:
                    try:
                        data = response.json()
                        answer = data.get("answer", "No answer returned")
                        st.write(answer)
                    except Exception:
                        st.error("Invalid JSON response from API")
                        st.text(response.text)
                        answer = "Error occurred"
                else:
                    st.error(f"API Error {response.status_code}")
                    st.text(response.text)
                    answer = "Error occurred"

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")
                answer = "Error occurred"

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
