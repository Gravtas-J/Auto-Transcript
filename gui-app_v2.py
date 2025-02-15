import tkinter as tk
from tkinter import filedialog, messagebox
import torch
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from io import BytesIO
from moviepy.editor import VideoFileClip

import tempfile
import os

# Function to transcribe audio file
def transcribe_audio(audio_path):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3",
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device=device,
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
def extract_audio_from_video(video_path):
    # Extract audio using moviepy
    with VideoFileClip(video_path) as video:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
            video.audio.write_audiofile(temp_audio_file.name)
            return temp_audio_file.name

# Function to format transcription to SRT
def format_srt(transcription):
    srt_content = []
    for idx, chunk in enumerate(transcription["chunks"]):
        start_time = chunk["timestamp"][0] if chunk["timestamp"] and chunk["timestamp"][0] is not None else 0
        end_time = chunk["timestamp"][1] if chunk["timestamp"] and chunk["timestamp"][1] is not None else 0
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

# Main Tkinter app
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio/Video Transcription with Whisper")
        self.root.geometry("800x600")

        # Variables to store file paths and transcription data
        self.file_path = ""
        self.audio_path = ""
        self.transcription_text = ""
        self.transcription_srt = ""

        # Create GUI components
        self.create_widgets()

    def create_widgets(self):
        # Frame for file selection
        self.file_frame = tk.Frame(self.root)
        self.file_frame.pack(pady=10)

        self.file_label = tk.Label(self.file_frame, text="No file selected")
        self.file_label.pack(side=tk.LEFT, padx=5)

        self.select_button = tk.Button(self.file_frame, text="Select File", command=self.select_file)
        self.select_button.pack(side=tk.LEFT, padx=5)

        self.transcribe_button = tk.Button(self.file_frame, text="Transcribe", command=self.transcribe, state=tk.DISABLED)
        self.transcribe_button.pack(side=tk.LEFT, padx=5)

        # Text area for transcription
        self.text_area = tk.Text(self.root, wrap=tk.WORD)
        self.text_area.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Frame for save buttons
        self.save_frame = tk.Frame(self.root)
        self.save_frame.pack(pady=10)

        self.save_transcription_button = tk.Button(self.save_frame, text="Save Transcription", command=self.save_transcription, state=tk.DISABLED)
        self.save_transcription_button.pack(side=tk.LEFT, padx=5)

        self.save_srt_button = tk.Button(self.save_frame, text="Save SRT File", command=self.save_srt, state=tk.DISABLED)
        self.save_srt_button.pack(side=tk.LEFT, padx=5)

    def select_file(self):
        file_types = [
            ("Audio Files", "*.mp3 *.wav *.m4a"),
            ("Video Files", "*.mp4 *.mkv *.avi"),
            ("All Files", "*.*"),
        ]
        file_path = filedialog.askopenfilename(title="Choose an audio or video file...", filetypes=file_types)
        if file_path:
            self.file_path = file_path
            self.file_label.config(text=os.path.basename(file_path))
            self.transcribe_button.config(state=tk.NORMAL)

    def transcribe(self):
        # Disable the transcribe button
        self.transcribe_button.config(state=tk.DISABLED)
        self.save_transcription_button.config(state=tk.DISABLED)
        self.save_srt_button.config(state=tk.DISABLED)

        # Clear previous transcription
        self.transcription_text = ""
        self.transcription_srt = ""
        self.text_area.delete(1.0, tk.END)

        # Show a message that transcription is in progress
        messagebox.showinfo("Transcription", "Transcription is in progress. Please wait...")

        # Determine file type
        file_extension = os.path.splitext(self.file_path)[1].lower()
        video_extensions = ['.mp4', '.mkv', '.avi']
        audio_extensions = ['.mp3', '.wav', '.m4a']

        if file_extension in video_extensions:
            # Extract audio from video file
            self.audio_path = extract_audio_from_video(self.file_path)
        elif file_extension in audio_extensions:
            self.audio_path = self.file_path
        else:
            messagebox.showerror("Error", "Unsupported file type.")
            return

        # Transcribe audio
        try:
            transcription = transcribe_audio(self.audio_path)
            self.transcription_text = transcription['text']
            self.transcription_srt = format_srt(transcription)
            self.text_area.insert(tk.END, self.transcription_text)
            messagebox.showinfo("Success", "Transcription complete!")
            self.save_transcription_button.config(state=tk.NORMAL)
            self.save_srt_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during transcription:\n{str(e)}")
        finally:
            # Re-enable the transcribe button
            self.transcribe_button.config(state=tk.NORMAL)
            # Clean up temporary audio file if created
            if self.audio_path != self.file_path and os.path.exists(self.audio_path):
                os.remove(self.audio_path)

    def save_transcription(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", title="Save Transcription", filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.transcription_text)
            messagebox.showinfo("Success", "Transcription saved successfully!")

    def save_srt(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".srt", title="Save SRT File", filetypes=[("SRT Files", "*.srt")])
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.transcription_srt)
            messagebox.showinfo("Success", "SRT file saved successfully!")

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
