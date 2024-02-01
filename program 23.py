from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

def evaluate_coherence(text):
    preprocessed_text = preprocess_text(text)
    dictionary = Dictionary([preprocessed_text])
    corpus = [dictionary.doc2bow(preprocessed_text)]

    lda_model = LdaModel(corpus, num_topics=1, id2word=dictionary)

    coherence_model = CoherenceModel(model=lda_model, texts=[preprocessed_text], dictionary=dictionary, coherence='c_v')
    coherence_score = coherence_model.get_coherence()

    return coherence_score

# Example usage
text = "Developing a coherent Python program requires careful planning and attention to detail. It's important to structure your code logically and use descriptive variable names. Additionally, commenting your code can improve readability and coherence."
coherence_score = evaluate_coherence(text)
print("Coherence Score:", coherence_score)
