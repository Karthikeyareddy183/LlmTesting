import subprocess
import json
import requests
import openai
import streamlit as st
import os
from groq import Groq
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

# ——————— Environment & API Setup ———————
openai.api_key = os.getenv("OPENAI_API_KEY", "")
HF_API_KEY = os.getenv("HF_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"
MODEL1 = "mistral-saba-24b"
GEMINI_API2 = os.getenv("GEMINI_API2", "")

# ——————— Helper to merge instruction + prompt ———————


def apply_instruction(prompt: str, instruction: str) -> str:
    return f"{instruction.strip()}\n\n{prompt.strip()}"

# ——————— Your existing model-callers ———————


# def call_gpt(prompt: str, instruction: str) -> str:
#     # ... existing code ...
#     resp = openai.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7,
#         max_tokens=300
#     )
#     return resp.choices[0].message.content.strip()


def call_gpt(prompt: str, instruction: str) -> str:
    messages = [
        {"role": "system",  "content": instruction},
        {"role": "user",    "content": prompt}
    ]
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=200
    )
    return resp.choices[0].message.content.strip()


def call_llama_hf(prompt: str, instruction: str) -> str:
    """
    Calls Groq API if GROQ_API_KEY is set, otherwise falls back to Hugging Face for chat completions.
    """
    # 1) Groq inference path
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user",   "content": prompt}
        ]
        try:
            comp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=300
            )
            return comp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Groq API Error: {e}"


def call_gemini(prompt: str, instruction: str) -> str:
    # ... existing code ...
    if not GEMINI_API2:
        return "⚠️ Set GEMINI_API_KEY"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API2}"
    headers = {'Content-Type': 'application/json'}
    payload = {
        # system prompt goes here
        "system_instruction": {"parts": [{"text": instruction}]},
        # user prompt
        "contents": [{"parts": [{"text": prompt}]}],
        # enforce token limit
        "generationConfig": {"maxOutputTokens": 300}
    }
    try:
        response = requests.post(url, headers=headers,
                                 data=json.dumps(payload))
        response.raise_for_status()
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        return f"⚠️ Gemini error: {e}"
    except (KeyError, IndexError) as e:
        return f"⚠️ Gemini error: Unexpected response format: {e} - {response.text}"

# ——————— New: Ollama wrapper ———————


def call_llama_ollama(prompt: str, instruction: str, model: str = "llama3-with-lora") -> str:
    """
    Runs: echo PROMPT | ollama run MODEL -S "SYSTEM INSTRUCTION"
    Captures the stdout as the answer.
    """
    try:
        proc = subprocess.run(
            [
                "ollama", "run", model,
                "set system", instruction,
                "set max-tokens", "300",
                prompt
            ],
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            errors="ignore"
        )
        # Ollama echoes only the model reply to stdout
        return proc.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"❌ Ollama error:\n{e.stderr.strip()}"


def mistral(prompt: str, instruction: str) -> str:
    """
    Calls Groq API if GROQ_API_KEY is set, otherwise falls back to Hugging Face for chat completions.
    """
    # 1) Groq inference path
    if GROQ_API_KEY:
        client = Groq(api_key=GROQ_API_KEY)
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user",   "content": prompt}
        ]
        try:
            comp = client.chat.completions.create(
                model=MODEL1,
                messages=messages,
                max_tokens=300
            )
            return comp.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Groq API Error: {e}"


# ——————— Streamlit UI ———————
st.set_page_config(page_title="LLM Comparator", layout="wide")
st.sidebar.header("🔧 System Instruction")

instruction = st.sidebar.text_area(
    "Enter your system prompt here…",
    value=(
        "You are Cheeko, a friendly and playful storytelling toy for young children."
        "When telling stories or answering questions, you:"
        "1) Use vivid, child-friendly imagery with colorful descriptions "
        "2) Include familiar characters from previous conversations when possible"
        "3) Use simple language with occasional fun, new vocabulary words"
        "4) Create 2-3 short, engaging sentences that build excitement "
        "5) Add gentle sound effects in text when appropriate (like 'whoosh!' or 'splash!')"
        "6) Ask questions that spark imagination and continue the conversation"
        "7) Always keep stories positive, uplifting, and age-appropriate "
        "Your goal is to create magical moments that inspire wonder and joy."
    ),
    height=200,
    key="sysinst"
)

st.sidebar.divider()
st.title("🔍 Compare LLM Responses")

prompt = st.text_input("Enter your prompt:", "")

if st.button("Generate Responses") and prompt:
    with st.spinner("Generating…"):
        results = {
            # "OpenAI GPT-3.5":       call_gpt(prompt, instruction),
            "LLaMA ":       call_llama_hf(prompt, instruction),
            "Gemini ":   call_gemini(prompt, instruction),
            # "Fine Tuned Llama ": call_llama_ollama(prompt, instruction),
            "Mistral ": mistral(prompt, instruction),
        }

    st.divider()
    cols = st.columns(2)
    for i, (name, text) in enumerate(results.items()):
        with cols[i % 2]:
            st.subheader(name)
            st.write(text)
