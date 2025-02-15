import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from io import BytesIO
import tempfile
import streamlit as st
from st_audiorec import st_audiorec
import ollama  # Make sure the Ollama library is installed and available

# Function to transcribe an audio file using Whisper
def transcribe_audio(audio_path):
    device = 0 if torch.cuda.is_available() else -1
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",  # See: https://huggingface.co/openai/whisper-large-v3
        torch_dtype=torch_dtype,
        device=device,
        model_kwargs={"attn_implementation": "flash_attention_2" if is_flash_attn_2_available() else "sdpa"},
    )

    outputs = pipe(
        audio_path,
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )
    
    return outputs


# Function to generate notes using Llama3 via Ollama
def gen_Notes(message, model='phi3'):  # Change model ID if needed
    try:
        # Read the system prompt from prompt.md
        with open('instruct.md', 'r') as f:
            system_prompt = f.read()
        
        # Prepare messages with the system prompt first, then the user transcript
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': message,
            }
        ]
        
        response = ollama.chat(model=model, messages=messages)
        return response['message']['content']
    except Exception as e:
        error_message = str(e).lower()
        if "not found" in error_message:
            return f"Model '{model}' not found. Please refer to the documentation at https://ollama.com/library."
        else:
            return f"An unexpected error occurred with model '{model}': {str(e)}"

# Main Streamlit app
def main():
    st.title("Live Audio Transcription & Llama3 Notes Generation")
    
    # Initialize session state variables if they don't exist yet
    if "recording" not in st.session_state:
        st.session_state.recording = False
    if "recorded_audio" not in st.session_state:
        st.session_state.recorded_audio = None
    if "transcription_text" not in st.session_state:
        st.session_state.transcription_text = ""
    if "notes" not in st.session_state:
        st.session_state.notes = ""

    # Sidebar toggle button for starting/ending transcription
    toggle_label = "End Transcript" if st.session_state.recording else "Start Transcript"
    if st.sidebar.button(toggle_label):
        st.session_state.recording = not st.session_state.recording
        # Clear previous data when starting a new transcript
        if st.session_state.recording:
            st.session_state.recorded_audio = None
            st.session_state.transcription_text = ""
            st.session_state.transcription_srt = ""
            st.session_state.notes = ""
    
    # If recording is active, display the audio recorder widget
    if st.session_state.recording:
        st.write("**Recording... Speak now!**")
        audio_bytes = st_audiorec()
        if audio_bytes is not None:
            st.session_state.recorded_audio = audio_bytes
            st.write("Recording captured. Click **End Transcript** to process transcription.")
    else:
        # When not recording, if audio is captured but not yet transcribed, process the transcription
        if st.session_state.recorded_audio is not None and st.session_state.transcription_text == "":
            st.info("Processing transcription, please wait...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
                temp_audio_file.write(st.session_state.recorded_audio)
                audio_path = temp_audio_file.name
            transcription = transcribe_audio(audio_path)
            st.session_state.transcription_text = transcription.get("text", "")
            st.success("Transcription complete!")
        
        # Display the transcription if available
        if st.session_state.transcription_text:
            st.text_area("Transcription", st.session_state.transcription_text, height=300)
            
            # Button to generate notes via Llama3 and Ollama
            if st.button("Generate Notes"):
                with st.spinner("Generating notes with '{model}': {str(e)}..."):
                    notes = gen_Notes(st.session_state.transcription_text)
                    st.session_state.notes = notes
                    st.success("Notes generated!")
            
            # Display generated notes if available
            if st.session_state.notes:
                st.text_area("Generated Notes", st.session_state.notes, height=300)
    
    # Download buttons for transcription and SRT files
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
    
    # Download button for generated notes
    if st.session_state.notes:
        st.sidebar.download_button(
            label="Download Notes",
            data=BytesIO(st.session_state.notes.encode()),
            file_name="notes.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()
