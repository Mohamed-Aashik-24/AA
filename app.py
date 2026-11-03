import os
import re

import requests
import streamlit as st


MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
SYSTEM_PROMPT = (
    "You are a helpful, accurate, and friendly AI assistant. "
    "Answer the user's question directly in clear, simple English."
)


def get_token():
    token = os.getenv("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        return str(st.secrets.get("HF_TOKEN", "")).strip()
    except (FileNotFoundError, KeyError):
        return ""


def local_fallback(message):
    question = re.sub(r"[^a-z0-9]+", " ", message.lower()).strip()
    words = set(question.split())

    if question in {"hi", "hello", "hey", "hey there"}:
        return "Hello! Ask me about AI, machine learning, Transformers, or Python."
    if "difference" in words and "ai" in words and (
        "machine" in words or "ml" in words
    ):
        return (
            "AI is the broader goal of making machines perform intelligent tasks. "
            "Machine learning is one way to achieve AI by learning patterns from data."
        )
    if "artificial intelligence" in question or words.intersection({"ai", "artificial"}):
        return (
            "Artificial intelligence is the field of building systems that perform "
            "tasks that usually require human intelligence, such as understanding "
            "language, recognizing images, and making predictions."
        )
    if "deep learning" in question or "dl" in words:
        return (
            "Deep learning is a type of machine learning that uses neural networks "
            "with many layers. It is commonly used for images, speech, and language."
        )
    if "machine learning" in question or "ml" in words:
        return (
            "Machine learning is a type of artificial intelligence in which a system "
            "learns patterns from examples and uses them to make predictions."
        )
    if "transformer" in question:
        return (
            "A Transformer uses attention to identify relationships between words. "
            "This helps it understand and generate language."
        )
    if "python" in question:
        return (
            "Python is a readable, general-purpose programming language used for "
            "automation, data science, web development, and AI."
        )
    if "help" in words or "what can you do" in question:
        return (
            "I can explain AI and programming concepts, compare technologies, and "
            "answer questions about machine learning and Transformers."
        )
    return (
        f"You asked: {message.strip()}\n\n"
        "Add an HF_TOKEN secret in Streamlit for open-ended answers. "
        "I can answer questions about AI, ML, DL, Transformers, and Python now."
    )


def generate_answer(message, history):
    token = get_token()
    if not token:
        return local_fallback(message)

    conversation = [f"System: {SYSTEM_PROMPT}"]
    for item in history[-6:]:
        conversation.append(f"{item['role'].title()}: {item['content']}")
    conversation.append(f"User: {message}")

    try:
        response = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {token}"},
            json={
                "inputs": "\n".join(conversation) + "\nAssistant:",
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
        if isinstance(result, list) and result:
            answer = result[0].get("generated_text")
            if answer:
                return answer.strip()
    except (requests.RequestException, ValueError):
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

prompt = st.chat_input("Ask a question...")
if prompt:
    history = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = generate_answer(prompt, history)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
