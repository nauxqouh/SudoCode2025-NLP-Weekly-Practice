from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def get_vectorizer(method="tfidf"):
    if method == "bow":
        return CountVectorizer(ngram_range=(1,2))
    elif method == "tfidf":
        return TfidfVectorizer(ngram_range=(1,2))
    else:
        raise ValueError("Method must be 'bow' or 'tfidf'")