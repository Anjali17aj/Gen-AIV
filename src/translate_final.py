import os
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the M2M100 model (1.2B parameters)
model_name = "facebook/m2m100_1.2B"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
model = M2M100ForConditionalGeneration.from_pretrained(model_name)

def translate_text(text, target_lang):
    """
    Translates English text to any supported language using M2M100 (1.2B).

    :param text: str, Input text in English.
    :param target_lang: str, Target language code (e.g., 'fr' for French, 'te' for Telugu).
    :return: str, Translated text.
    """
    tokenizer.src_lang = "en"  # Source language is English
    encoded_text = tokenizer(text, return_tensors="pt")

    # Generate translation using the correct language token
    generated_tokens = model.generate(**encoded_text, forced_bos_token_id=tokenizer.get_lang_id(target_lang))

    # Decode translation
    translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated_text

if __name__ == "__main__":
    input_path = r"C:\Users\CDAC-WBLuser5\Desktop\multilingual-video-app\data\transcripts\transcript.txt"
    output_folder = r"C:\Users\CDAC-WBLuser5\Desktop\multilingual-video-app\data\translations"  # Folder to store translated files

  
    with open(input_path, "r", encoding="utf-8") as file:
        english_text = file.read().strip()

    
    target_languages = {
        "hi": "Hindi",
        "bn": "Bengali",
        # "te": "Telugu",
        "mr": "Marathi",
        "gu": "Gujarati",
        "ml": "Malayalam",
        "kn": "Kannada",
    }

    for lang_code, lang_name in target_languages.items():
        translated_text = translate_text(english_text, lang_code)

        # Save the translated text to a new file
        output_file = os.path.join(output_folder, f"transcript_{lang_code}.txt")
        with open(output_file, "w", encoding="utf-8") as out_file:
            out_file.write(translated_text)

        print(f"{lang_name} Translation saved: {output_file}")
