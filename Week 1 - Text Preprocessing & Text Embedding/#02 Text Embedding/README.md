# #02 Word2Vec Embedding on Vietnamese Wikipedia (viwik18)

This folder implements **Word2Vec** model on the [**viwik18**](https://github.com/NTT123/viwik18) dataset.

Goals:
- Train word embeddings for Vietnamese words/phrases.
- Explore semantic relationships between words.
- Visualize the embedding space.

## Folder Structure

```
#02 Text Embedding/
├── src/                               # Source code
│   ├── preprocessing/                 # Text preprocessing modules
│   │   └── processor.py               # Preprocessing logic
│   │
│   ├── features/                      # Feature engineering utilities
│   │   ├── chunker.py                 # Text chunking
│   │   └── phraser.py                 # Phrase detection
│   │
│   ├── models/                        # Model training scripts
│   │   ├── train.py                   # Training pipeline
│   │   └── word2vec_viwik18_bigram.py # Word2Vec with bigram support
│   │
│   ├── main.py                        # Entry point for pipeline
│   └── data_loader.py                 # Data loading utilities
│
├── data/                              # Data directory
│   ├── processed/                     # Processed data storage
│   │   └── processed_text.json        # Preprocessed text data
│   └── vietnamese-stopwords.txt       # Stopwords list
│
├── notebooks/                         # Jupyter notebooks
│   └── experiments.ipynb              # Experimental workflows
│
├── README.md                          # Project documentation
└── requirements.txt                   # Dependencies list
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 1 - Text Preprocessing & Text Embedding/#02 Text Embedding"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation
4. 
## Notebook Workflow

- **Data preprocessing:**
  - Lowercasing.
  - Remove digits, punctuation, special symbols.
  - Tokenize text into words.
  - *Split tokens into pseudo-sentences of 40 words.*

- **Phrase Detection:**

  Use `gensim.models.Phrases` to detect bigrams/trigrams:

  ```
  "phi chính phủ"
  "việt nam"
  "liên hợp quốc"
  ```

- **Train Word2Vec:**
  - Initialize skip-gram Word2Vec model.
  - Training progress follows 2 key steps - build vocab and train.

- **Visualization:**
  - Top 20 most frequency words
    <img width="897" height="545" alt="image" src="https://github.com/user-attachments/assets/2c26624e-bf83-4af5-b103-5338c608f3a2" />
  - Visualizing similar words to `'việt nam'` using PCA and TSNE:
    <img width="904" height="526" alt="image" src="https://github.com/user-attachments/assets/18fe9218-0ef1-4f56-b8b9-296aa5c2a0b7" />
    <img width="863" height="526" alt="image" src="https://github.com/user-attachments/assets/7c515503-1229-41ad-ab70-9f660920e4f0" />
  - Embedding heatmap
    <img width="1411" height="1011" alt="image" src="https://github.com/user-attachments/assets/deaa66e8-9a97-4aca-b5dd-cf48192d84dd" />

## References

1. [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
2. [Gensim Word2Vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html)
