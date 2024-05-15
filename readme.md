# Audio/Video Transcription with Whisper

This is a Streamlit-based application for transcribing audio and video files using OpenAI's Whisper model. The application supports a variety of file formats and outputs both plain text and SRT formatted transcriptions.

## Features

- **Audio/Video Upload:** Supports uploading audio and video files in formats like MP3, WAV, M4A, MP4, MKV, and AVI.
- **Automatic Speech Recognition (ASR):** Utilizes the `openai/whisper-large-v3` model to transcribe audio content.
- **Download Transcriptions:** Provides options to download the transcription in plain text.
- **SRT Formatting:** Generates SRT files with timestamped transcriptions for use in subtitles.


![Transcription with Whisper Screenshot](.images\Screenshot.png)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gravtas-J/Auto-Transcript.git
   cd Auto-Transcript
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload File:**
   - Use the sidebar to upload an audio or video file.
   
2. **Transcribe:**
   - Click the "Transcribe" button to start the transcription process.
   - The application will indicate that it is transcribing. This may take some time depending on the file size and content.
   
3. **View and Download Transcriptions:**
   - The transcription will be displayed in a text area once complete.
   - Download buttons will appear in the sidebar for downloading the transcription as a plain text file and an SRT file.

## Known Issues

- **Configuration File Issues:** Adding entries to `config.toml` may cause inconsistent importing of the `pipeline`.
- **Upload Limit:** The maximum upload file size is currently limited to 200MB, although in theory, it should support up to 100GB(by edditing `config.toml`).

## Future Improvements

- **Increase Upload Limit:** Work on reliably increasing the upload limit.
- **Download from URL:** Add functionality to download and transcribe files directly from URLs.
- **Multi-language Support:** Support transcription for multiple languages in both input and output.

## Potential Additions

- **Chat with Video using Local LLM:** Implement a feature to chat with video using a local Language Model (LM).

## Important Notes

- **Model Download:** On first use, the application will download the `openai/whisper-large-v3` model, which may take some time.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For issues, suggestions, or contributions, please create an issue or pull request on the [GitHub repository](https://github.com/Gravtas-J/Auto-Transcript).

