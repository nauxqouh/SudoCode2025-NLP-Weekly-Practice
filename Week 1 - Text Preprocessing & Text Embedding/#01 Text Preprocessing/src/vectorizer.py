from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Vectorizer:
    def __init__(self, method="tfidf", ngram_range=(1, 1),  **kwargs):
        if method == "bow":
            self.vectorizer = CountVectorizer(ngram_range=ngram_range, **kwargs)
        elif method == "tfidf":
            self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, **kwargs)
        elif method == "unigram":
            self.vectorizer = CountVectorizer(ngram_range=(1, 1), **kwargs)
        elif method == "bigram":
            self.vectorizer = CountVectorizer(ngram_range=(2, 2), **kwargs)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names_out()

    def vocab_size(self):
        return len(self.vectorizer.vocabulary_)