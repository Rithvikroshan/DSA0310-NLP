from transformers import pipeline

def translate_english_to_french(text):
    translator = pipeline(task="translation", model="Helsinki-NLP/opus-mt-en-fr")
    translated_text = translator(text, max_length=50)[0]['translation_text']
    return translated_text

# Example usage
english_text = "Hello, how are you?"
french_translation = translate_english_to_french(english_text)
print("English:", english_text)
print("French Translation:", french_translation)
