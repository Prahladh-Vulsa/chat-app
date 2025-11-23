import streamlit as st
from openai import OpenAI
import tempfile
import mimetypes

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🎤 GPT Voice Assistant with TTS")
st.write("Speak into the microphone and receive a spoken reply.")

# --- AUDIO INPUT ---
audio_bytes = st.audio_input("Record your voice:")


# --- SAFE & ERROR-PROOF AUDIO HANDLING ---
def detect_audio_suffix(audio_bytes):
    """Return correct file extension based on MIME type."""
    mime_type = mimetypes.guess_type("file", strict=False)[0]

    # Streamlit often sends audio/webm by default
    if audio_bytes[:4] == b"\x1A\x45\xDF\xA3":
        return ".webm"

    # WAV file signature (RIFF)
    if audio_bytes[:4] == b"RIFF":
        return ".wav"

    # Fallback
    return ".wav"


def transcribe_audio(file_bytes):
    """Convert mic audio → text using Whisper with full safety."""
    suffix = detect_audio_suffix(file_bytes)

    # Save as temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(file_bytes)
        temp_audio_path = temp_audio.name

    # Open temporary file for Whisper
    with open(temp_audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )

    return response.text


# --- GPT SHORT RESPONSE ---
def ask_gpt(question):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI voice assistant. "
                    "Always reply in short responses. "
                    "Limit replies to 2–3 concise sentences."
                )
            },
            {"role": "user", "content": question}
        ],
        max_tokens=150
    )
    return completion.choices[0].message.content


# --- TEXT TO SPEECH ---
def text_to_speech(text):
    """Convert GPT text → spoken audio."""
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        format="mp3"
    )
    return response.read()


# --- MAIN LOGIC ---
if audio_bytes:
    st.write("⏳ Processing... please wait.")

    try:
        # STEP 1 — Transcribe
        text_input = transcribe_audio(audio_bytes)
        st.write("🗣️ You said:", text_input)

        # STEP 2 — GPT reply
        reply = ask_gpt(text_input)
        st.write("🤖 Assistant:", reply)

        # STEP 3 — Play TTS
        tts_audio = text_to_speech(reply)
        st.audio(tts_audio, format="audio/mp3")

    except Exception as e:
        st.error("An error occurred while processing your audio. Please try again.")
        st.write("Error message:", str(e))
