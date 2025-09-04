# 📰 #01 Vietnamese Online News Text Preprocessing

This folder contains a **complete text preprocessing pipeline and applied feature extraction** for the [Vietnamese Online News Dataset](https://www.kaggle.com/datasets/haitranquangofficial/vietnamese-online-news-dataset).  
The practice is divided into 2 main parts:

- **Part I: Text Preprocessing** - Normalize, clean, tokenize, and remove stopwords from raw news articles.  
- **Part II: Text Feature Extraction** - Represent preprocessed texts using multiple vectorization techniques for downstream NLP tasks.

## Folder Structure

```
#01 Text Preprocessing/
├── src/                           # Source code
│   ├── main.py                    # Main pipeline script
│   ├── text_processor.py          # Custom preprocessing module (TextProcessor & NewsProcessor)
│   ├── vectorizer.py              # Text vectorization utilities
│
├── data/                          # Data directory
│   ├── processed/                 # Storage for processed outputs
│   │   ├── processed_tokens.json  # Tokens after preprocessing
│   │── vietnamese-stopwords.txt   # Stopwords list used for filtering
│   
└── notebooks/                     # Jupyter notebooks
    ├── experiments.ipynb
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 1 - Text Preprocessing & Text Embedding/#01 Text Preprocessing"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation

## Key stages

### Part I: Text Preprocessing

Implemented in `text_processor.py`:

- **TextProcessor class**
  - Normalize text: lowercase, remove email/URLs/punctuation/numbers, clean whitespaces.
  - Tokenize using `underthesea` library.
  - Remove Vietnamese stopwords.
  - Output tokens (list) or preprocessed string (for vectorizer).

- **NewsProcessor class**
  - Apply `TextProcessor` across the dataset.
  - Support early stopping (for quick debugging).
  - Save results in `.json` (tokens) or `.txt` (vectorizer input).

### Part II: Text Feature Extraction

Applied **Scikit-learn** vectorizers on the cleaned text from Part I:

1. **Bag of Words (BoW)**  
2. **TF-IDF**  
3. **Unigram**  
4. **Bigram**

## References

1. [Text preprocessing with NLTK](https://www.nltk.org/book/ch03.html)
2. [Scikit-learn: Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html)
