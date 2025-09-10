# #03 Product Review Sentiment Analysis

This folder contains a Sentiment Analysis project in Python for the [vietnamese-text-classification](https://www.kaggle.com/datasets/tuannguyenvananh/vietnamese-text-classification-dataset). The project applies traditional **Machine Learning algorithms (Naive Bayes and SVM)** for Natural Language Processing (NLP).

## Folder Structure

```
#03 ML for NLP/
├── src/                           # Source code
│   ├── main.py                    # Main pipeline script
│   ├── models                     # 
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── evaluate.py
│   ├── preprocessing
│   │   ├── cleaner.py             # Custom preprocessing module
│   │   └── vectorizer.py
│
├── data/                          # Data directory
│   ├── datasets.py                 # Storage for processed outputs
│   ├── train.csv          
│   │── vietnamese-stopwords.txt   # Stopwords list used for filtering
│   
├── notebooks/                     # Jupyter notebooks
│   ├── experiments.ipynb
│
├── config.py
├── README.md
└── requirements.txt
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 2 - ML, DL for NLP/#03 ML for NLP"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation
4. Adjust parameters or paths in `config.py` before running `main.py`

## Key insights

- Naive Bayes (NB):
  - Works under the conditional independence assumption.
  - Performs best with Bag-of-Words (CountVectorizer) representation, as it leverages raw frequency counts.
- Support Vector Machines (SVM):
  - A margin-based classifier that benefits from TF-IDF features, which emphasize important terms while down-weighting very common words.
  - SVM is robust to high-dimensional sparse vectors, making it a strong baseline for text classification.
- Comparison
  - NB is fast, simple, and interpretable.
  - SVM generally achieves higher accuracy, especially on balanced datasets, but is more computationally expensive.

## References

1. [Scikit-learn: Classification](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
