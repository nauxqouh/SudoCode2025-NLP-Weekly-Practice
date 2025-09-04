from gensim.models.phrases import Phrases, Phraser

def build_bigrams(sentences, min_count=5, threshold=5):
    bigram = Phrases(sentences=sentences, min_count=min_count, threshold=threshold)
    bigram_phraser = Phraser(bigram)
    return [bigram_phraser[sent] for sent in sentences]
