# #07 Text Translation - Transformer

This folder contains a Text Translation project in Python using the [EVBCorpus](https://github.com/qhungngo/EVBCorpus) dataset. The project applies **Transformer Model**.

## Folder Structure

```
Week 5 - Transformer/
├── src/                          
│   ├── models/
│   │   ├── 
│   │   └── 
├── notebooks/                         # Jupyter notebooks
│   └── 250101-Transformer.ipynb       # Experiments notebook
├── README.md
└── requirements.txt
```

_(Comming soon for updating full pipeline)_

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 5 - Transformer"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation

## Model Usage

1. Load checkpoints:
```
from tensorflow import keras
transformer = keras.models.load_model(
    './transformer.keras',
    custom_objects={
        'PositionalEncoding': PositionalEncoding,
        'EncoderLayer': EncoderLayer,
        'DecoderLayer': DecoderLayer,
        'Transformer': Transformer
    }
)
transformer.summary()
```

2. Example usage:
```
translator = Translator(
    transformer,
    en_tokenizer,
    vi_tokenizer,
    max_len_en=max_len_en,
    max_len_vi=max_len_vi
)
```

#### Evaluation

- Visualize learning curves (loss vs. epochs) to monitor training progress of experiment model.
  <img width="770" height="329" alt="Screenshot 2025-10-08 at 13 00 12" src="https://github.com/user-attachments/assets/34388f2c-4fff-484a-9958-85c3d7d0705e" />


## References

1. [The Transformer Model](https://www.tensorflow.org/text/tutorials/transformer)

