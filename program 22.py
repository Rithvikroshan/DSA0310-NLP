import spacy

def resolve_references(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)

    resolved_text = []
    current_reference = None

    for token in doc:
        if token.text.lower() == 'it' and current_reference:
            resolved_text.append(current_reference.text)
        else:
            resolved_text.append(token.text)
            if token.dep_ == 'nsubj':
                current_reference = token

    return ' '.join(resolved_text)

# Example usage
text = "The cat is on the mat. It is sleeping."
resolved_text = resolve_references(text)
print("Resolved Text:", resolved_text)
