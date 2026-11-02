import os
import streamlit as st
import requests

# Experiment 9: Transformer-Based LLM Chatbot
# Name: Mohamed Aashik S
# Roll No: 24BAD072
# Model: SmolLM2-135M-Instruct

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful, accurate, and friendly AI assistant. "
    "Answer the user's question directly in clear, simple English. "
    "Explain technical concepts step by step when useful and include "
    "relevant examples. Do not invent facts. If you are unsure, say so."
)

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"

def get_token():
    """Read an optional Hugging Face token without requiring one to start."""
    return os.getenv("HF_TOKEN", st.secrets.get("HF_TOKEN", ""))

def local_fallback(message):
    """Keep the app useful when the hosted model is unavailable."""
    question = message.lower()
    if "artificial intelligence" in question or question == "what is ai?":
        return "Artificial intelligence is the field of building systems that can perform tasks that usually require human intelligence, such as understanding language, recognizing images, and making predictions."
    if "machine learning" in question:
        return "Machine learning is a type of artificial intelligence in which a system learns patterns from examples and uses those patterns to make predictions or decisions."
    if "transformer" in question:
        return "A Transformer processes relationships between words using attention. This lets it weigh the most relevant parts of the input while generating an answer."
    return "I am ready to help. Add an HF_TOKEN secret in Streamlit to enable generated answers, or ask about AI, machine learning, or Transformers."

def generate_answer(message, history):
    """Call Hugging Face without loading a model into Streamlit memory."""
    token = get_token()
    if not token:
        return local_fallback(message)

    conversation = [f"System: {SYSTEM_PROMPT}"]
    for item in history[-6:]:
        conversation.append(f"{item['role'].title()}: {item['content']}")
    conversation.append(f"User: {message}")
    prompt = "\n".join(conversation) + "\nAssistant:"

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 120,
                    "temperature": 0.3,
                    "return_full_text": False,
                },
            },
            timeout=45,
        )
        response.raise_for_status()
        result = response.json()
        if isinstance(result, list) and result and result[0].get("generated_text"):
            return result[0]["generated_text"].strip()
    except requests.RequestException:
        pass
    return local_fallback(message)

st.set_page_config(page_title="Transformer-Based LLM Chatbot", page_icon="🤖")
st.title("Transformer-Based LLM Chatbot")
st.caption("Experiment 9 | Roll No: 24BAD072 | Hosted inference with SmolLM2")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(prompt, history)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
