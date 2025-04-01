import whisper
import os
from moviepy import editor as mp 

# Define paths
input_video_path = "C:/Users/CDAC-WBLuser5/Desktop/multilingual-video-app/data/input/sample.mp4"   # Input video
transcript_path = "C:/Users/CDAC-WBLuser5/Desktop/multilingual-video-app/data/transcripts/transcript.txt"  # Output transcript file

# Step 1: Extract Audio from Video
def extract_audio(video_path, audio_path):
    """Extracts audio from the given video file and saves it as a WAV file."""
    print("Extracting audio from video...")
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec="pcm_s16le")  # Save as WAV format
    print(f"Audio extracted and saved at: {audio_path}")

# Step 2: Transcribe Audio to Text using Whisper
def transcribe_audio(audio_path, transcript_path):
    """Transcribes the extracted audio using Whisper and saves the transcript."""
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Load the Whisper model (base version)

    print("Transcribing audio...")
    result = model.transcribe(audio_path)

    # Save transcript to a file
    with open(transcript_path, "w", encoding="utf-8") as file:
        file.write(result["text"])

    print(f"Transcription completed. Transcript saved at: {transcript_path}")

# Main Execution
if __name__ == "__main__":
    audio_path = r"C:\Users\CDAC-WBLuser5\Desktop\multilingual-video-app\data\output\audio.wav" # Temporary audio file path

    # Ensure the output folder exists
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    
    # Step 1: Extract audio from video
    extract_audio(input_video_path, audio_path)

    # Step 2: Transcribe the extracted audio
    transcribe_audio(audio_path, transcript_path)

    print(" Speech-to-Text process completed successfully!")
