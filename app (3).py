import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# ---- Page config ----
st.set_page_config(
    page_title="MediBot — AI Medical Assistant",
    page_icon="💊",
    layout="centered"
)

# ---- Custom CSS ----
st.markdown("""
<style>
    .main { max-width: 750px; }
    .stChatMessage { border-radius: 12px; }
    .emergency-box {
        background-color: #FCEBEB;
        border-left: 4px solid #E24B4A;
        padding: 12px 16px;
        border-radius: 4px;
        color: #501313;
        font-weight: 500;
    }
    .disclaimer {
        font-size: 12px;
        color: #888;
        border-top: 1px solid #eee;
        padding-top: 8px;
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("## 💊 MediBot")
st.markdown("Your AI-powered medical assistant. Ask about symptoms, conditions, or general health.")
st.markdown("---")

# ---- API Key from Streamlit Secrets ----
api_key = st.secrets.get("GOOGLE_API_KEY", None)

if not api_key:
    st.error("❌ Google API key not found. Please add it to Streamlit Secrets.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key
genai.configure(api_key=api_key)

# ---- Load LLM (cached so it loads only once) ----
@st.cache_resource
def load_llm():
    available = []
    for m in genai.list_models():
        if "generateContent" in m.supported_generation_methods:
            available.append(m.name.replace("models/", ""))

    PREFERRED = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-pro",
    ]

    selected = None
    for pref in PREFERRED:
        if pref in available:
            selected = pref
            break
    if selected is None and available:
        selected = available[0]

    if selected is None:
        return None, "No models found"

    try:
        llm = ChatGoogleGenerativeAI(
            model=selected,
            temperature=0.2,
            convert_system_message_to_human=True
        )
        llm.invoke("say hello")
        return llm, selected
    except Exception as e:
        return None, str(e)

llm, model_name = load_llm()

if llm is None:
    st.error(f"❌ Failed to load model: {model_name}")
    st.stop()

st.success(f"✅ Running on: `{model_name}`")

# ---- Emergency keywords ----
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe",
    "cannot breathe", "not breathing", "bleeding heavily",
    "unconscious", "overdose", "severe allergic", "anaphylaxis", "seizure"
]

def is_emergency(text):
    return any(kw in text.lower() for kw in EMERGENCY_KEYWORDS)

# ---- Session state for chat history ----
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm MediBot 👋 Ask me about any symptoms, conditions, or general health questions. For emergencies please call **102** immediately."
        }
    ]

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### 💊 MediBot")
    st.markdown("AI Medical Assistant")
    st.markdown("---")
    st.markdown("**Quick topics:**")

    quick_topics = [
        "What is diabetes?",
        "Symptoms of high blood pressure?",
        "What causes migraines?",
        "Cold vs flu differences?",
        "What is dengue fever?",
        "How to improve sleep?",
        "What is acne and treatment?",
        "Signs of vitamin D deficiency?",
    ]

    for topic in quick_topics:
        if st.button(topic, use_container_width=True):
            st.session_state.quick_input = topic

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Chat cleared! How can I help you today?"
            }
        ]
        st.rerun()

    st.markdown("---")
    st.markdown("⚠️ *For emergencies call 102*")
    st.markdown("*This app is for educational purposes only.*")

# ---- Display chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- MediBot response function ----
def get_response(user_input):
    if is_emergency(user_input):
        return """<div class="emergency-box">
        🚨 <strong>Emergency Detected!</strong><br><br>
        Please call <strong>102</strong> (ambulance) immediately or go to the nearest emergency room.<br>
        Do not wait — this could be life-threatening.
        </div>"""

    # Build history for context
    history_text = ""
    for m in st.session_state.messages[-6:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history_text += f"{role}: {m['content']}\n\n"

    prompt = f"""You are MediBot, a knowledgeable and compassionate AI medical assistant.

Previous conversation:
{history_text}

User question: {user_input}

Reply in this exact format:

**Overview**
Write 1-2 sentence summary here.

**Symptoms**
- Symptom 1
- Symptom 2
- Symptom 3

**Causes**
- Cause 1
- Cause 2
- Cause 3

**Treatment & Management**
- Treatment 1
- Treatment 2
- Treatment 3

**When to See a Doctor**
Write clear guidance here.

---
*Disclaimer: This is general health information only. Always consult a qualified doctor for diagnosis or treatment.*
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"⚠️ Error: {e}"

# ---- Handle quick topic buttons ----
if "quick_input" in st.session_state:
    user_msg = st.session_state.quick_input
    del st.session_state.quick_input

    st.session_state.messages.append({"role": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("MediBot is thinking..."):
            reply = get_response(user_msg)
            if "emergency-box" in reply:
                st.markdown(reply, unsafe_allow_html=True)
            else:
                st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()

# ---- Main chat input ----
user_input = st.chat_input("Ask a health question...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("MediBot is thinking..."):
            reply = get_response(user_input)
            if "emergency-box" in reply:
                st.markdown(reply, unsafe_allow_html=True)
            else:
                st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.rerun()