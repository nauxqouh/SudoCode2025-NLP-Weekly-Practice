# #06 Text Summarization Seq2Seq with Attention in PyTorch

This folder contains a Text Summarization project in Python using the [VNDS - A Vietnamese Dataset for Summarization](https://huggingface.co/datasets/nam194/vietnews) dataset. The project applies **Encoder-Decoder GRU-based with Attention Model**.

## Folder Structure

```
Week 4 - Attention/
├── src/                          
│   ├── models/
│   │   ├── 
│   │   └── 
├── notebooks/                         # Jupyter notebooks
│   └── 250101-AttentionModel.ipynb    # Experiments notebook
├── README.md
└── requirements.txt
```

_(Comming soon for updating full pipeline)_

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/nauxqouh/SudoCode2025-NLP-Weekly-Practice.git
cd "SudoCode2025-NLP-Weekly-Practice/Week 4 - Attention"
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the notebooks in `notebooks/` for experimentation

## Model Usage

1. Load checkpoints:
```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("../src/models/checkpoint.pt", map_location=device)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
encoder.to(device)
decoder.to(device)
```

2. Example usage:
```
summaries = generate_batch_summary(
    encoder, decoder, sample_batch,
    tokenizer,
    start_token_id=tokenizer.cls_token_id,
    end_token_id=tokenizer.sep_token_id,
    max_len=50,
    device=device
)
```

## Model Training
- Encoder
  ```
  class EncoderRNN(nn.Module):
      def __init__(self, vocab_size, embed_size, hidden_size, pad_token_id, dropout_p=0.1):
          super(EncoderRNN, self).__init__()
          self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
          self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
          self.dropout = nn.Dropout(dropout_p)
  
      def forward(self, input_ids, attention_mask):
          embedded = self.dropout(self.embedding(input_ids))
          input_length = attention_mask.sum(dim=1).cpu()
          packed = nn.utils.rnn.pack_padded_sequence(embedded, input_length, enforce_sorted=False, batch_first=True)
          outputs, hidden = self.gru(packed)
          outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
          return outputs, hidden
  ```
  
- Attention:
  ```
  class LuongAttention(nn.Module):
      def __init__(self, hidden_size):
          super(LuongAttention, self).__init__()
          self.attn = nn.Linear(hidden_size, hidden_size)
          self.softmax = nn.Softmax(dim=-1)
  
      def forward(self, hidden, encoder_outputs, mask=None):
          hidden = hidden[-1].unsqueeze(2)
          scores = torch.bmm(encoder_outputs, hidden).squeeze(2)
          if mask is not None:
              scores = scores.masked_fill(mask == 0, -1e9)
          attn_weights = self.softmax(scores)
          context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
          context = context.squeeze(1) 
          return context, attn_weights
  ```
  
- Decoder with Attention:
  ```
  class AttnDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, pad_token_id, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_token_id)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.attention = LuongAttention(hidden_size)
        self.out = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, input_step, last_hidden, encoder_outputs, mask=None):
        embedded = self.dropout(self.embedding(input_step))
        embedded = embedded.unsqueeze(1)
        context, attn_weights = self.attention(last_hidden, encoder_outputs, mask)
        context = context.unsqueeze(1)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(1)
        context = context.squeeze(1)
        output = torch.cat((output, context), dim=1)
        logits = self.out(output)
        return logits, hidden, attn_weights
  ```

#### Evaluation

- Visualize learning curves (loss vs. epochs) to monitor training progress of experiment model.
  <img width="606" height="435" alt="Screenshot 2025-10-02 at 13 37 56" src="https://github.com/user-attachments/assets/2e79c4bc-5168-4839-9c4e-54cae2de1bb8" />

## References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
