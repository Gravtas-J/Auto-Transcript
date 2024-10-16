import tkinter as tk
from tkinter import scrolledtext
import threading
import pyaudio
import numpy as np
from transformers import pipeline

# Initialize the transcription pipeline
transcriber = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v2",
    device="cuda:0"  # Change to "cpu" if you don't have a CUDA-compatible GPU
)

# Setup PyAudio with 16 kHz sample rate
RATE = 16000
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

def transcribe():
    """Function to handle audio stream and update the GUI with transcriptions."""
    buffer = b''
    try:
        while running:
            data = stream.read(CHUNK, exception_on_overflow=False)
            buffer += data
            # Accumulate enough data for 1 second of audio
            if len(buffer) >= RATE * 2:  # 2 bytes per sample for paInt16
                # Convert buffer to numpy array
                audio_data = np.frombuffer(buffer, dtype=np.int16).astype(np.float32) / 32768.0
                # Transcribe audio
                result = transcriber(audio_data, return_timestamps=False)
                transcript = result.get('text', '...')
                # Update GUI
                text_area.configure(state='normal')
                text_area.insert(tk.END, transcript.strip() + '\n')
                text_area.configure(state='disabled')
                text_area.yview(tk.END)
                buffer = b''  # Reset buffer
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def start_transcription():
    global running
    running = True
    threading.Thread(target=transcribe, daemon=True).start()

def stop_transcription():
    global running
    running = False

# Create the main window
root = tk.Tk()
root.title("Real-Time Audio Transcription")

# Add a text area
text_area = scrolledtext.ScrolledText(root, state='disabled', width=80, height=20)
text_area.pack(pady=20)

# Add start and stop buttons
start_button = tk.Button(root, text="Start Transcription", command=start_transcription)
start_button.pack(side='left', padx=20)

stop_button = tk.Button(root, text="Stop Transcription", command=stop_transcription)
stop_button.pack(side='right', padx=20)

# Start the Tkinter event loop
root.mainloop()
