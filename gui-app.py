import tkinter as tk
from tkinter import scrolledtext
import threading
import pyaudio
from transformers import pipeline

# Initialize the transcription pipeline
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", device="cuda:0")

# Setup PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

def transcribe():
    """Function to handle audio stream and update the GUI with transcriptions."""
    try:
        while running:
            data = stream.read(1024, exception_on_overflow=False)
            result = transcriber(data)
            transcript = result.get('text', '...')
            text_area.configure(state='normal')
            text_area.insert(tk.END, transcript + '\n')
            text_area.configure(state='disabled')
            text_area.yview(tk.END)
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
