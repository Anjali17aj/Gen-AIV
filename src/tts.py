import os
import subprocess

# Path to the Indic-TTS directory
INDIC_TTS_DIR = r"C:\Users\CDAC-WBLuser5\Desktop\multilingual-video-app\Indic-TTS"  # 🔹 Change this to the actual path

# Input text file (Replace with translated text files)
input_text_file = "transcript_hindi.txt"  # Example: Hindi text file

# Output directory for generated speech
output_dir = r"C:\Users\CDAC-WBLuser5\Desktop\multilingual-video-app\data\output"
os.makedirs(output_dir, exist_ok=True)

# Extract language from file name (Assumes format: translated_<language>.txt)
language_code_mapping = {
    "hindi": "hin",
    "bengali": "ben",
    "marathi": "mar",
    "gujarati": "guj",
    "malayalam": "mal",
    "kannada": "kan"
}

lang = os.path.basename(input_text_file).split("_")[1].split(".")[0]
lang_code = language_code_mapping.get(lang.lower())

if not lang_code:
    raise ValueError(f"❌ Language code not found for {lang}")

# Output speech file
output_audio_file = os.path.join(output_dir, f"{lang}_tts_output.mp3")

# Run the Indic-TTS inference script
command = [
    "python",
    os.path.join(INDIC_TTS_DIR, "tts_infer.py"),
    "--lang", lang_code,
    "--text_path", input_text_file,
    "--output_path", output_audio_file
]

print(f"🎤 Converting {input_text_file} to speech...")
subprocess.run(command, check=True)
print(f"✅ Speech saved: {output_audio_file}")
