import os
import re

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
    return os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN", "")

def local_fallback(message):
    """Keep the app useful when the hosted model is unavailable."""
    question = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
    if question in {"hi", "hello", "hey", "hey there"}:
        return "Hello! Ask me about artificial intelligence, machine learning, Transformers, or Python."
    if "difference" in question and "ai" in question and ("machine learning" in question or " ml " in f" {question} "):
        return "AI is the broader goal of making machines perform intelligent tasks. Machine learning is one way to achieve AI by learning patterns from data instead of using only hand-written rules."
    if "artificial intelligence" in question or question in {"what is ai", "ai meaning", "define ai"}:
        return "Artificial intelligence is the field of building systems that can perform tasks that usually require human intelligence, such as understanding language, recognizing images, and making predictions."
    if "deep learning" in question or question in {"what is dl", "dl meaning", "define dl"}:
        return "Deep learning is a type of machine learning that uses neural networks with many layers. It is commonly used for images, speech, and language."
    if "machine learning" in question or question in {"what is ml", "what is an ml", "ml meaning", "define ml"}:
        return "Machine learning is a type of artificial intelligence in which a system learns patterns from examples and uses those patterns to make predictions or decisions."
    if "transformer" in question:
        return "A Transformer processes relationships between words using attention. This lets it weigh the most relevant parts of the input while generating an answer."
    if "python" in question:
        return "Python is a readable, general-purpose programming language used for web development, automation, data science, and artificial intelligence."
    if "what can you do" in question or "help" in question:
        return "I can explain AI and programming concepts, compare technologies, and answer questions about machine learning and Transformers."
    return f"You asked: {message.strip()}\n\nI do not have the hosted language model enabled yet. Add an HF_TOKEN secret in Streamlit for open-ended answers."

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
    except requests.RequestException as error:
        return f"I could not reach the hosted model ({error}).\n\n{local_fallback(message)}"
    return local_fallback(message)

st.set_page_config(page_title="Transformer-Based LLM Chatbot", page_icon="🤖")
st.title("Transformer-Based LLM Chatbot")
st.caption("Experiment 9 | Roll No: 24BAD072 | Hosted inference with SmolLM2")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question...")
if prompt:
    st.session_state.pending_prompt = prompt
    st.rerun()

pending_prompt = st.session_state.pop("pending_prompt", None)
if pending_prompt:
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": pending_prompt})
    with st.chat_message("user"):
        st.markdown(pending_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(pending_prompt, history)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
