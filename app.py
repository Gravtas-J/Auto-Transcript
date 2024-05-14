import streamlit as st
from moviepy.editor import VideoFileClip
from transformers import pipeline
from io import BytesIO

# Initialize the Whisper model
@st.cache_resource
def load_transcription_pipeline():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v2",
        device=0  # Change to "cuda:0" if you have a GPU
    )

transcription_pipeline = load_transcription_pipeline()

# Function to extract audio from video
def extract_audio_from_video(video_file):
    try:
        video = VideoFileClip(video_file.name)
        audio = video.audio
        audio_buffer = BytesIO()
        audio.write_audiofile(audio_buffer, format='mp3')
        audio_buffer.seek(0)
        video.close()
        return audio_buffer
    except Exception as e:
        st.error(f"Error extracting audio from video: {e}")
        return None

# Function to handle audio file directly
def handle_audio_file(audio_file):
    try:
        audio_buffer = BytesIO()
        audio_buffer.write(audio_file.getvalue())
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        st.error(f"Error handling audio file: {e}")
        return None

# Function to transcribe audio and save to text file
def transcribe_audio(audio_buffer):
    try:
        audio_buffer.seek(0)
        result = transcription_pipeline(audio_buffer)
        text = result['text']
        text_buffer = BytesIO()
        text_buffer.write(text.encode('utf-8'))
        text_buffer.seek(0)
        return text_buffer, text
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None, None

st.title('Media to Transcript Converter')

# File uploader widget
uploaded_file = st.file_uploader("Choose a media file", type=["mp4", "avi", "mov", "mkv", "mp3", "wav", "aac"])

if uploaded_file is not None:
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
