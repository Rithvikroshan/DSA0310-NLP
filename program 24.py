import nltk
from nltk import pos_tag, ne_chunk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def recognize_dialog_acts(utterance):
    tagged_words = pos_tag(word_tokenize(utterance))
    chunked_tree = ne_chunk(tagged_words)

    dialog_acts = []

    for subtree in chunked_tree:
        if isinstance(subtree, nltk.Tree):
            label, mention = subtree.label(), subtree.leaves()[0][0]
            response = {'PERSON': f'Hello! How can I help you, {mention}?',
                        'GPE': f'I see you mentioned {mention}. What can I assist you with related to that?'}.get(label, "I'm not sure how to respond.")
            dialog_acts.append((label, response))
        else:
            dialog_acts.append(('UNKNOWN', "I'm not sure how to respond."))

    return dialog_acts

# Example usage
conversation = ["Hi there!", "Planning a trip to Paris.", "What's the weather like there?",
                "Not sure about the weather, but I can help with travel tips.", "Tell me more about Paris."]

for utterance in conversation:
    acts = recognize_dialog_acts(utterance)
    print("Dialog Acts:", acts)
