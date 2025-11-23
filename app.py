import streamlit as st
import openai
import base64

openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("🎤 GPT Voice Assistant with TTS")
st.write("Speak into the mic and get a short 2–3 sentence spoken reply.")

# ---- RECORD AUDIO ----
audio_bytes = st.audio_input("Record your voice:")


# ---- GPT TEXT RESPONSE ----
def ask_gpt(question):
    completion = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful AI voice assistant. "
                    "Always reply in short answers. "
                    "Limit replies to 2–3 concise sentences."
                )
            },
            {"role": "user", "content": question}
        ],
        max_tokens=150
    )
    return completion.choices[0].message.content


# ---- WHISPER TRANSCRIPTION ----
def transcribe_audio(file_bytes):
    response = openai.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=("audio.wav", file_bytes, "audio/wav")
    )
    return response.text


# ---- OPENAI TTS AUDIO ----
def text_to_speech(text):
    speech_response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        format="mp3"
    )
    return speech_response.read()  # MP3 bytes


# ---- MAIN FLOW ----
if audio_bytes:
    st.write("⏳ Processing your voice...")

    # Step 1: Convert voice → text
    text_input = transcribe_audio(audio_bytes)
    st.write("🗣️ You said:", text_input)

    # Step 2: Ask GPT
    reply = ask_gpt(text_input)
    st.write("🤖 Assistant:", reply)

    # Step 3: Convert GPT → Speech
    audio_output = text_to_speech(reply)
    st.audio(audio_output, format="audio/mp3")
