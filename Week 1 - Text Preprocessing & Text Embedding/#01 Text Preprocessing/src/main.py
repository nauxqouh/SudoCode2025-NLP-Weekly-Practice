import os
import json
from text_processor import TextProcessor, NewsProcessor
from vectorizer import Vectorizer

def main():
    # Load dataset
    DAT_PATH = "data/news_dataset.json"
    with open(DAT_PATH, "r", encoding="utf") as f:
        dat_raw = json.load(f)
    # Remove news that have empty content
    raw_news = [doc for doc in dat_raw if doc.get("content", "").strip()]
    
    # I: Preprocessing
    ## Init processor
    news_processor = NewsProcessor(stopwords_path="data/vietnamese-stopwords.txt", vectorizer_flag=False)
    
    ## Text preprocessing
    processed_news = news_processor.process(raw_news)
    os.makedirs("data/processed", exist_ok=True)
    news_processor.save(processed_news, "data/processed/processed_tokens.json")
    
    # II: Vectorizer (Bow, TF-IDF, Unigram, Bigram)
    texts = [doc["content"] for doc in processed_news]
    ## BoW
    bow_vec = Vectorizer(method="bow")
    X_bow = bow_vec.fit_transform(texts)
    print("BoW:", X_bow.shape)
    
    ## TF-IDF
    tfidf_vec = Vectorizer(method="tfidf")
    X_tfidf = tfidf_vec.fit_transform(texts)
    print("TF-IDF:", X_tfidf.shape)
    
    ## Unigram
    unigram_vec = Vectorizer(method="unigram")
    X_unigram = unigram_vec.fit_transform(texts)
    print("TF-IDF:", X_unigram.shape)
    
    ## Bigram
    bigram_vec = Vectorizer(method="bigram")
    X_bigram = bigram_vec.fit_transform(texts)
    print("TF-IDF:", X_bigram.shape)
    
if __name__ == "__main__":
    main()    
    
    
    