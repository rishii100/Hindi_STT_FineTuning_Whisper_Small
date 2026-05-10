import streamlit as st
import torch
from transformers import pipeline
import librosa
import numpy as np

# Page Configuration
st.set_page_config(
    page_title="Hindi Speech-to-Text | Whisper-Small",
    page_icon="🎙️",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #FF4B4B;
        color: white;
    }
    .stAudio {
        width: 100%;
    }
    .title-text {
        text-align: center;
        color: #1E1E1E;
        font-family: 'Inter', sans-serif;
    }
    </style>
    """, unsafe_allow_name=True)

# App Title
st.markdown("<h1 class='title-text'>🎙️ Hindi Speech-to-Text</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Fine-tuned Whisper Small Model for Hindi ASR</p>", unsafe_allow_html=True)
st.divider()

# Sidebar
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.header("Model Information")
    st.info("""
    - **Model:** whisper-small-hindi
    - **Base:** OpenAI Whisper Small
    - **Language:** Hindi (Devanagari)
    - **Fine-tuned on:** Google FLEURS
    """)
    st.write("---")
    st.caption("Developed by Aneerban Saha")

# Model Loading (Cached)
@st.cache_resource
def load_asr_pipeline():
    model_id = "rishii100/whisper-small-hindi"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        chunk_length_s=30
    )
    return asr_pipe

try:
    with st.spinner("🚀 Initializing AI Model... (May take a moment)"):
        pipe = load_asr_pipeline()
    st.success("✨ Model Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Audio Input
st.subheader("1. Provide Audio Input")
audio_file = st.file_uploader("Upload an audio file (wav, mp3, flac)", type=["wav", "mp3", "flac"])

st.markdown("--- or ---")

# Optional: Audio recorder (requires streamlit-audio-recorder)
st.write("Record your voice directly:")
audio_value = st.audio_input("Record audio")

final_audio = None
if audio_file is not None:
    final_audio = audio_file
elif audio_value is not None:
    final_audio = audio_value

if final_audio:
    st.audio(final_audio, format='audio/wav')
    
    if st.button("✨ Transcribe Now"):
        try:
            with st.spinner("⏳ Processing Hindi Speech..."):
                # Load audio using librosa for consistency
                # We need to read the bytes first
                audio_bytes = final_audio.read()
                
                # Perform ASR
                # Pipeline handles the sampling rate conversion automatically usually, 
                # but we can pass it bytes directly.
                result = pipe(audio_bytes, generate_kwargs={"language": "hindi", "task": "transcribe"})
                
                st.divider()
                st.subheader("📋 Transcription")
                st.success(result["text"])
                
                # Copy to clipboard button (simulated with text area)
                st.text_area("Devanagari Output:", value=result["text"], height=100)
                
        except Exception as e:
            st.error(f"⚠️ Transcription failed: {e}")
else:
    st.info("💡 Please upload or record an audio file to start transcribing.")

# Footer
st.divider()
st.caption("Note: For best results, use a quiet environment and speak clearly in Hindi.")
