from src.preprocessing.processor import TextProcessor
from src.features.chunker import chunk_text
from src.features.phraser import build_bigrams
from src.models.train import train_word2vec
from src.data_loader import load_viwik18
import os
import json

def main():
    # Load data
    dataset_path = load_viwik18("data/raw/viwik18.txt")
    with open("data/raw/viwik18.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Preprocess
    with open("data/vietnamese-stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    processor = TextProcessor(stopwords=stopwords, vectorizer_flag=False)
    tokens = processor.process(text)

    # Chunk
    sentences = list(chunk_text(tokens, chunk_size=15))
    os.makedirs(os.path.dirname("data/processed"), exist_ok=True)
    with open("data/processed/processed_text.json", "w", encoding="utf-8") as f:
        json.dump(tokens, f, ensure_ascii=False, indent=2)

    # Bigram
    bigram_sentences = build_bigrams(sentences)

    # Train Word2Vec
    model = train_word2vec(bigram_sentences, model_path="models/word2vec_viwik18.model")

    # Test
    print(model.wv.most_similar(positive=["việt nam"], topn=5))

if __name__ == "__main__":
    main()
