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
@st.cache_resource(show_spinner="Loading the language model...")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True
    )
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def prepare_messages(message, history):
    """Prepare system prompt, recent conversation, and current user message."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Keep the latest six messages to limit prompt size.
    for item in (history or [])[-6:]:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in ("user", "assistant") and isinstance(content, str):
                if content.strip():
                    messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message.strip()})
    return messages


def respond(message, history, tokenizer, model):
    """Generate an answer using SmolLM2-135M-Instruct."""
    if not message or not message.strip():
        return "Please enter a question."

    messages = prepare_messages(message, history)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536
    ).to(DEVICE)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.3,
            top_p=0.85,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Decode only tokens generated after the input prompt.
    generated_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    if not answer:
        return "I couldn't generate a response. Please try rephrasing your question."

    return answer


st.set_page_config(page_title="Transformer-Based LLM Chatbot", page_icon="🤖")
st.title("Transformer-Based LLM Chatbot")
st.caption("Experiment 9 | Roll No: 24BAD072 | Powered by SmolLM2-135M-Instruct")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            history = st.session_state.messages[:-1]
            try:
                tokenizer, model = load_model()
                answer = respond(prompt, history, tokenizer, model)
            except Exception as error:
                answer = (
                    "The language model could not start on this deployment. "
                    f"Details: {error}"
                )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
