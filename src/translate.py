import torch
from transformers import MBartForConditionalGeneration, AutoModelForSeq2SeqLM, AutoTokenizer

# Load the pre-trained AI4Bharat IndicTrans model
model_name = "ai4bharat/indictrans2-en-indic-1B"  # English to Indian languages
tokenizer = AutoTokenizer.from_pretrained(model_name, token=True,trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, token=True,trust_remote_code=True)

# Move model to GPU if available (for faster processing)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate_text(text, target_lang="hi"):  # Default to Hindi
    """Translate English text to the target Indian language."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to GPU if available

    # Force the model to generate in the target language
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(f"<{target_lang}>")

    translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

# Read the transcribed text (English) from the file
input_file = "../data/transcripts/transcript.txt"
output_hindi_file = "../data/transcripts/transcript_hi1.txt"
output_telugu_file = "../data/transcripts/transcript_te2.txt"

with open(input_file, "r", encoding="utf-8") as f:
    english_text = f.read()

# Translate text
hindi_translation = translate_text(english_text, "hi")
telugu_translation = translate_text(english_text, "te")

# Print translations for debugging
print("🔹 Hindi Translation:", hindi_translation)
print("🔹 Telugu Translation:", telugu_translation)

# Save translations to files
with open(output_hindi_file, "w", encoding="utf-8") as f_hi:
    f_hi.write(hindi_translation)

with open(output_telugu_file, "w", encoding="utf-8") as f_te:
    f_te.write(telugu_translation)

print("✅ Translation completed successfully!")
print(f"🔹 Hindi Transcript saved at: {output_hindi_file}")
print(f"🔹 Telugu Transcript saved at: {output_telugu_file}")
