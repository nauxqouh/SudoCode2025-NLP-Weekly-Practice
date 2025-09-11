# #04 VNTC - Vietnamese News Text Classification using Simple Neural Network

This folder contains a Text Classification project in Python for the [VNTC/27Topics](https://github.com/duyvuleo/VNTC). The project applies **simple Neural Network** both TensorFlow version and Pytorch version.

## Folder Structure

```
#04 DL for NLP/
├── notebooks/                     # Jupyter notebooks
│   ├── 250101-DLforNLP.ipynb
├── README.md
└── requirements.txt
```

(Comming soon for updating full pipeline)

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 2 - ML, DL for NLP/#04 DL for NLP"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation

## Key stages

#### Data Preparation
- Extract and preprocess raw data using a simple cleaning pipeline (without tokenization or stopword removal, since PhoBERT will handle subword segmentation).
- Split into **train**, **validation**, and **test** sets.

#### Text Embedding (Feature Extraction)
- Use **PhoBERT** (`transformers` + `torch`) to encode text into embeddings to **768-dimensional embeddings**.
  ```
  tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
  phobert = AutoModel.from_pretrained("vinai/phobert-base")
  ```
- Save pre-computed embeddings: X_train_embeddings.npy, X_val_embeddings.npy, X_test_embeddings.npy.
- Encode topic labels into integer and save as: y_train_encoded.npy, y_val_encoded.npy, y_test_encoded.npy.

_(Available at my Kaggle Dataset: [text-classification-prepared-dataset](https://www.kaggle.com/datasets/nauxqouh/text-classification-prepared-dataset))_

#### Model Training
- Train an **MLP classifier** end-to-end using embeddings as input, implemented in both **TensorFlow** and **PyTorch** versions.
  - Tensor Flow, Keras
    ```
    model_tf = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_size,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    ```
  - Pytorch
    ```
    class MLPClassifier(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.6),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            
        def forward(self, x):
            return self.net(x)
    ```
- Optimize with **Adam**, **CrossEntropyLoss**, and a **learning rate scheduler**.
- Apply **early stopping** and save the **best-performing checkpoint**.

#### Evaluation & Testing

- Evaluate on validation with accuracy, loss and test sets with accuracy, precision, recall, F1-score per topic label.
- Visualize learning curves (loss/accuracy vs. epochs) to monitor training progress.
  <img width="990" height="451" alt="image" src="https://github.com/user-attachments/assets/610e3614-270d-450e-a173-0fd1a222734a" />
  
## References

1. [Deep Learning Book](https://www.deeplearningbook.org/)
2. [TensorFlow Tutorial](https://www.tensorflow.org/tutorials)
3. [Pytorch Tutorial](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
