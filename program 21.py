import nltk
from nltk import pos_tag, ne_chunk
from nltk.chunk import tree2conlltags
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def get_meaning(word, pos='n'):
    synsets = wordnet.synsets(word, pos=pos)
    return synsets[0].definition() if synsets else None

def analyze_sentence(sentence):
    tagged_words = pos_tag(word_tokenize(sentence))
    chunked_tree = ne_chunk(tagged_words)
    iob_tags = tree2conlltags(chunked_tree)
    
    noun_phrases = [' '.join(tag[0] for tag in iob_tags if tag[2] == 'B-NP')]
    return {phrase: get_meaning(phrase) for phrase in noun_phrases}

# Example usage
sentence = "The quick brown fox jumps over the lazy dog."
results = analyze_sentence(sentence)
print("Noun Phrases and their Meanings:", results)
