import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import pipeline
from io import BytesIO
import logging
import torch
from huggingface_hub import snapshot_download

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("transformers").setLevel(logging.DEBUG)

# Cache the model locally
@st.cache_resource
def download_model():
    model_path = snapshot_download("openai/whisper-tiny")
    return model_path

# Initialize the Whisper model
@st.cache_resource
def load_transcription_pipeline():
    model_path = download_model()
    logging.debug("Loading transcription pipeline from local cache...")
    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_path,
        device=device  # Use GPU if available, else CPU
    )

transcription_pipeline = load_transcription_pipeline()
logging.debug("Transcription pipeline loaded successfully.")

# Function to extract audio from video
def extract_audio_from_video(video_file):
    try:
        logging.debug("Extracting audio from video...")
        st.write("Extracting audio from video...")
        video = VideoFileClip(video_file.name)
        audio = video.audio
        audio_buffer = BytesIO()
        audio.write_audiofile(audio_buffer, format='mp3')
        audio_buffer.seek(0)
        video.close()
        logging.debug("Audio extracted successfully.")
        st.write("Audio extracted successfully.")
        return audio_buffer
    except Exception as e:
        error_message = f"Error extracting audio from video: {e}"
        logging.error(error_message)
        st.error(error_message)
        return None

# Function to handle audio file directly
def handle_audio_file(audio_file):
    try:
        logging.debug("Handling audio file...")
        st.write("Handling audio file...")
        audio_buffer = BytesIO()
        audio_buffer.write(audio_file.getvalue())
        audio_buffer.seek(0)
        logging.debug("Audio file handled successfully.")
        st.write("Audio file handled successfully.")
        return audio_buffer
    except Exception as e:
        error_message = f"Error handling audio file: {e}"
        logging.error(error_message)
        st.error(error_message)
        return None

# Function to transcribe audio and save to text file
def transcribe_audio(audio_buffer):
    try:
        logging.debug("Transcribing audio...")
        st.write("Transcribing audio...")
        audio_buffer.seek(0)
        result = transcription_pipeline(audio_buffer)
        text = result['text']
        text_buffer = BytesIO()
        text_buffer.write(text.encode('utf-8'))
        text_buffer.seek(0)
        logging.debug("Transcription completed successfully.")
        st.write("Transcription completed successfully.")
        return text_buffer, text
    except Exception as e:
        error_message = f"Error transcribing audio: {e}"
        logging.error(error_message)
        st.error(error_message)
        return None, None

st.title('Media to Transcript Converter')

# File uploader widget
uploaded_file = st.file_uploader("Choose a media file", type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "aac"])

if uploaded_file is not None:
    logging.debug(f"Uploaded file: {uploaded_file.name}, type: {uploaded_file.type}")
    file_type = uploaded_file.type.split('/')[0]
    if file_type == 'video':
        audio_buffer = extract_audio_from_video(uploaded_file)
    elif file_type == 'audio':
        audio_buffer = handle_audio_file(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a video or audio file.")
        audio_buffer = None

    if audio_buffer:
        st.audio(audio_buffer)
        text_buffer, text = transcribe_audio(audio_buffer)
        if text_buffer and text:
            st.write("Transcript:", text)
            st.download_button(
                label="Download Transcript as TXT",
                data=text_buffer,
                file_name="transcript.txt",
                mime="text/plain"
            )
        else:
            logging.debug("No transcription text available.")
    else:
        logging.debug("No audio buffer available.")
else:
    logging.debug("No file uploaded.")
