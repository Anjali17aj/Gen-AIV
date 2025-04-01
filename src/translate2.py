
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# Load the full M2M100 model (1.2B parameters)
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
    english_text = """ Stationery to protect your privacy information. Have you seen five blade scissors? It can shred paper into tiny pieces and have a brush-like cleaner to clean the debris between the blades. However, it takes some effort to completely shred a sheet of paper. Then, there's the privacy protection roller. It's dense text roller obscures information with a few rolls. The ink dries quickly and is refillable. Finally, there's this tool for thermal paper. Wipe it over the information and it disappears completely. It also has a small knife for opening packages. How do you protect your personal information?"""

    # Test multiple languages
    target_languages = {
        "hi": "Hindi",
        "ta": "Tamil",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        "zh": "Chinese",
        "ar": "Arabic",
        "ru": "Russian",
        "ja": "Japanese"
    }

    for lang_code, lang_name in target_languages.items():
        translated_text = translate_text(english_text, lang_code)
        print(f"{lang_name} Translation: {translated_text}")
