import streamlit as st
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from io import BytesIO
from moviepy.editor import VideoFileClip
import tempfile

# Function to transcribe audio file
def transcribe_audio(audio_path):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",  # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
        torch_dtype=torch.float16,
        device="cuda:0",  # or mps for Mac devices
        model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
    )

    # Transcribe audio
    outputs = pipe(
        audio_path,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    
    return outputs

# Function to extract audio from video
def extract_audio_from_video(video_file):
    # Save video file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as temp_video_file:
        temp_video_file.write(video_file.read())
        temp_video_file_path = temp_video_file.name
    
    # Extract audio using moviepy
    with VideoFileClip(temp_video_file_path) as video:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            video.audio.write_audiofile(temp_audio_file.name)
            return temp_audio_file.name

# Function to format transcription to SRT
def format_srt(transcription):
    srt_content = []
    for idx, chunk in enumerate(transcription["chunks"]):
        start_time = chunk["timestamp"][0]
        end_time = chunk["timestamp"][1]
        text = chunk["text"]
        srt_content.append(f"{idx+1}")
        srt_content.append(f"{format_time(start_time)} --> {format_time(end_time)}")
        srt_content.append(f"{text}\n")
    return "\n".join(srt_content)

# Function to format time for SRT
def format_time(seconds):
    millisec = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02}:{minutes:02}:{seconds:02},{millisec:03}"

# Streamlit app
def main():
    st.title("Audio/Video Transcription with Whisper")

    # Sidebar for file upload and transcription button
    st.sidebar.header("Upload Audio or Video File")
    uploaded_file = st.sidebar.file_uploader("Choose an audio or video file...", type=["mp3", "wav", "m4a", "mp4", "mkv", "avi"])

    # Initialize session state for transcription and SRT data
    if "transcription_text" not in st.session_state:
        st.session_state.transcription_text = ""
        st.session_state.transcription_srt = ""

    if uploaded_file is not None:
        if st.sidebar.button("Transcribe"):
            with st.spinner("Transcribing..."):
                # Check if the file is an audio or video file
                file_type = uploaded_file.type.split('/')[0]
                
                if file_type == 'video':
                    # Extract audio from video file
                    audio_path = extract_audio_from_video(uploaded_file)
                else:
                    # Save audio file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
                        temp_audio_file.write(uploaded_file.read())
                        audio_path = temp_audio_file.name
                
                # Transcribe audio
                transcription = transcribe_audio(audio_path)
                st.session_state.transcription_text = transcription['text']
                st.session_state.transcription_srt = format_srt(transcription)

            st.success("Transcription complete!")
    st.text_area("Transcription", st.session_state.transcription_text, height=300)

    # Display download buttons if transcription data is available
    if st.session_state.transcription_text and st.session_state.transcription_srt:
        st.sidebar.download_button(
            label="Download Transcription",
            data=BytesIO(st.session_state.transcription_text.encode()),
            file_name="transcription.txt",
            mime="text/plain"
        )

        st.sidebar.download_button(
            label="Download SRT File",
            data=BytesIO(st.session_state.transcription_srt.encode()),
            file_name="transcription.srt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
