import streamlit as st
from openai import OpenAI
import tempfile

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("🎤 GPT Voice Assistant with TTS")
st.write("Speak into the mic and receive a spoken reply.")

# Audio input
uploaded_audio = st.audio_input("Record your voice:")


# ---------- AUDIO FORMAT DETECTION ----------
def detect_audio_suffix(audio_bytes):
    """Detect audio type using file signature."""
    if audio_bytes.startswith(b"\x1A\x45\xDF\xA3"):   # WEBM header
        return ".webm"
    if audio_bytes.startswith(b"RIFF"):  # WAV header
        return ".wav"
    return ".wav"  # default


# ---------- TRANSCRIBE WITH WHISPER ----------
def transcribe_audio(uploaded_file):
    file_bytes = uploaded_file.getvalue()

    suffix = detect_audio_suffix(file_bytes)

    # Write audio to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
        temp_audio.write(file_bytes)
        temp_path = temp_audio.name

    # Send to Whisper
    with open(temp_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f
        )
    return response.text


# ---------- GPT RESPONSE (SHORT REPLIES) ----------
def ask_gpt(question):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI voice assistant. "
                    "Always reply in short responses, 2–3 sentences only."
                )
            },
            {"role": "user", "content": question},
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content


# ---------- TEXT TO SPEECH ----------
def text_to_speech(text):
    speech = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        format="mp3"
    )
    return speech.read()  # MP3 bytes


# ---------- MAIN APP LOGIC ----------
if uploaded_audio:
    try:
        st.write("⏳ Processing your voice...")

        # Step 1: Speech → Text
        transcript = transcribe_audio(uploaded_audio)
        st.write("🗣️ You said:", transcript)

        # Step 2: GPT → Reply
        reply = ask_gpt(transcript)
        st.write("🤖 Assistant:", reply)

        # Step 3: Reply → Speech
        tts_audio = text_to_speech(reply)
        st.audio(tts_audio, format="audio/mp3")

    except Exception as e:
        st.error("An error occurred while processing your audio. Please try again.")
        st.write("Error:", str(e))
