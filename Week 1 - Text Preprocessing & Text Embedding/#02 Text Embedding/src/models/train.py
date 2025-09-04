import time
from gensim.models import Word2Vec

def train_word2vec(sentences, model_path="models/word2vec_viwik18.model"):
    model = Word2Vec(
        vector_size=100,
        window=5,
        min_count=3,
        workers=2,
        sg=1,
        sample=1e-3,
        negative=10,
        epochs=5
    )

    start_time = time.time()
    model.build_vocab(sentences, progress_per=10000)
    print(f"Build vocab done in {time.time() - start_time:.2f}s")

    start_time = time.time()
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    print(f"Training done in {time.time() - start_time:.2f}s")

    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model
