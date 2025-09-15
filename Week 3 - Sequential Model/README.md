# #05 Text Generation using LSTM in PyTorch

This folder contains a Text Generation project in Python using the [10000 Vietnamese Books](https://www.kaggle.com/datasets/iambestfeeder/10000-vietnamese-books) dataset. The project applies **LSTM Model** both character-level and word-level.

## Folder Structure

```
Week 3 - Sequential Model/
├── data/                          
│   └── dat_book_chunk1.json           # Preprocessing data for the first 100 books in raw dataset
├── src/                          
│   ├── models/
│   │   ├── single-char.pth            # Best model state, char_to_int dict for character-level
│   │   └── single-word.pth            # Best model state, word_to_int dict for word-level
├── notebooks/                         # Jupyter notebooks
│   └── 250101-SequentialModel.ipynb   # Experiments notebook
├── README.md
└── requirements.txt
```

(Comming soon for updating full pipeline)

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 3 - Sequential Model"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation

## Model Usage

1. Load checkpoints:
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("../src/models/single-word.pth", map_location=device)
best_model_state = checkpoint["model_state"]
word_to_int = checkpoint["word_to_int"]
int_to_word = {i: w for w, i in word_to_int.items()}
vocab_size = len(word_to_int)
```

2. Load weights:
```
model = WordLevelGenerationLSTM(vocab_size=vocab_size, embed_dim=128, hidden_dim=256, num_layers=1)
model.load_state_dict(best_model_state)
```
*Note: Define `WordLevelGenerationLSTM` followed bellow.*

3. Example usage:
```
prompt = "cậu bé"
print(generate(model, prompt, 100, temperature=0.8, seq_len=50))
```

## Model Training
- Train an **TextGenerationLSTM** end-to-end using Pytorch in both **Character-level** and **Word-level**.
  - Character-Level
    ```
    class CharModel(nn.Module): 
        def __init__(self, vocab_size, hidden_size=256, num_layers=2, dropout=0.2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 128)
            self.lstm = nn.LSTM(
                input_size=128, hidden_size=hidden_size, 
                num_layers=num_layers, batch_first=True, dropout=dropout
            )
            self.dropout = nn.Dropout(dropout)
            self.linear = nn.Linear(hidden_size, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)  # (batch, seq, embed_dim)
            x, _ = self.lstm(x)
            x = x[:, -1, :]    # take only the last output
            x = self.linear(self.dropout(x))  # produce output
            return x
    ```
  - Word-level
    ```
    class WordLevelGenerationLSTM(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=1):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim, vocab_size)
    
        def forward(self, x, hidden=None):
            x = self.embedding(x)
            out, hidden = self.lstm(x, hidden)
            out = self.fc(out)
            return out, hidden
    ```

#### Evaluation

- Visualize learning curves (loss vs. epochs) to monitor training progress of word-level model.
  <img width="691" height="470" alt="image" src="https://github.com/user-attachments/assets/3e8b1ebd-ba4b-4f2e-97de-8ca2fcf0e024" />

## References

1. [Sequence Model](https://www.coursera.org/learn/nlp-sequence-models)
2. [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. [Text Generation with LSTM in PyTorch](https://machinelearningmastery.com/text-generation-with-lstm-in-pytorch/)
4. [Word-Level Text Generation using LSTM](https://debuggercafe.com/word-level-text-generation-using-lstm/)
