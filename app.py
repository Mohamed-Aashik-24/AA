import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

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

if torch.cuda.is_available():
    DEVICE = "cuda"
    MODEL_DTYPE = torch.float16
else:
    DEVICE = "cpu"
    MODEL_DTYPE = torch.float32

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

tokenizer, model = load_model()

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
            answer = respond(prompt, history, tokenizer, model)
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
