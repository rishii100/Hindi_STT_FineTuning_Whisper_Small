import streamlit as st
import requests
import io

# ─── Page Config ───
st.set_page_config(
    page_title="Hindi Speech-to-Text | Whisper Small",
    page_icon="🎙️",
    layout="centered"
)

# ─── Custom CSS ───
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #FF6B6B, #4ECDC4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}

.subtitle {
    text-align: center;
    color: #888;
    font-size: 1rem;
    margin-top: 0;
    margin-bottom: 2rem;
}

.result-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 24px;
    color: #e0e0e0;
    font-size: 1.2rem;
    line-height: 1.8;
    margin-top: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, #0f3460, #1a1a2e);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    border: 1px solid #1a3a5c;
}

.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #4ECDC4;
}

.metric-label {
    font-size: 0.8rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
}

div.stButton > button {
    width: 100%;
    border-radius: 12px;
    height: 3.2em;
    font-weight: 600;
    font-size: 1rem;
    background: linear-gradient(135deg, #FF6B6B, #ee5a24);
    color: white;
    border: none;
    transition: all 0.3s ease;
}

div.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───
st.markdown('<h1 class="main-title">🎙️ Hindi Speech-to-Text</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Fine-tuned Whisper Small — Powered by Hugging Face Inference API</p>', unsafe_allow_html=True)

# ─── Sidebar ───
with st.sidebar:
    st.markdown("### 📊 Model Performance")

    st.markdown("""
    <div class="metric-card" style="margin-bottom: 12px;">
        <div class="metric-label">Word Error Rate</div>
        <div class="metric-value">26.87%</div>
        <div style="color: #4ECDC4; font-size: 0.75rem;">↓ 60.8% improvement</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card">
        <div class="metric-label">Character Error Rate</div>
        <div class="metric-value">10.41%</div>
        <div style="color: #4ECDC4; font-size: 0.75rem;">↓ 69.8% improvement</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ℹ️ Model Info")
    st.markdown("""
    - **Model:** [whisper-small-hindi](https://huggingface.co/rishii100/whisper-small-hindi)
    - **Base:** OpenAI Whisper Small (244M)
    - **Language:** Hindi 🇮🇳
    - **Dataset:** Google FLEURS
    - **License:** MIT
    """)
    st.markdown("---")
    st.caption("Built by **Aneerban Saha**")


# ─── HF Inference API ───
HF_MODEL_ID = "rishii100/whisper-small-hindi"
API_URL = f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_ID}"


def transcribe_audio(audio_bytes: bytes) -> str:
    """Send audio to HuggingFace Inference API and get transcription."""
    headers = {"Content-Type": "audio/wav"}

    # Use HF token from secrets if available
    hf_token = st.secrets.get("HF_TOKEN", None)
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    response = requests.post(API_URL, headers=headers, data=audio_bytes, timeout=120)

    if response.status_code == 503:
        st.warning("⏳ Model is loading on the server. Please wait 30-60 seconds and try again.")
        return None
    elif response.status_code != 200:
        st.error(f"❌ API Error ({response.status_code}): {response.text}")
        return None

    result = response.json()
    if isinstance(result, dict) and "text" in result:
        return result["text"]
    elif isinstance(result, list) and len(result) > 0:
        return result[0].get("text", str(result))
    return str(result)


# ─── Main Content ───
st.markdown("### 🎧 Upload or Record Hindi Audio")

tab1, tab2 = st.tabs(["📁 Upload File", "🎤 Record Audio"])

with tab1:
    audio_file = st.file_uploader(
        "Upload a Hindi audio file",
        type=["wav", "mp3", "flac", "m4a", "ogg"],
        help="Supported formats: WAV, MP3, FLAC, M4A, OGG"
    )

with tab2:
    audio_recording = st.audio_input("Click to record from your microphone")

# Determine which audio source to use
final_audio = audio_file if audio_file is not None else audio_recording

if final_audio:
    st.audio(final_audio)
    st.markdown("---")

    if st.button("✨ Transcribe", use_container_width=True):
        audio_bytes = final_audio.read()

        with st.spinner("🔮 Transcribing Hindi speech..."):
            transcription = transcribe_audio(audio_bytes)

        if transcription:
            st.markdown("### 📋 Transcription Result")
            st.markdown(
                f'<div class="result-box">{transcription}</div>',
                unsafe_allow_html=True
            )

            # Copyable text area
            with st.expander("📑 Copy Text"):
                st.text_area("", value=transcription, height=100, label_visibility="collapsed")
else:
    st.info("💡 Upload a Hindi audio file or record your voice to get started.")

# ─── Footer ───
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "For best results, use clear Hindi speech in a quiet environment. "
    "Model may take ~30s to load on first request."
    "</p>",
    unsafe_allow_html=True
)
