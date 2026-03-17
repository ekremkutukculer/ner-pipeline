import torch
import torch.nn as nn
from typing import List, Optional
from src.models.char_cnn import CharCNN
from src.models.crf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size: int, num_chars: int, num_tags: int,
                 word_emb_dim: int = 300, char_emb_dim: int = 50,
                 char_filters: int = 50, char_kernel: int = 3,
                 hidden_size: int = 256, num_layers: int = 1,
                 dropout: float = 0.5, pretrained_word_emb: Optional[torch.Tensor] = None):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        if pretrained_word_emb is not None:
            self.word_embedding.weight.data.copy_(pretrained_word_emb)
        self.char_cnn = CharCNN(num_chars, char_emb_dim, char_filters, char_kernel)
        input_dim = word_emb_dim + char_filters
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                           num_layers=num_layers, batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)
        self.crf = CRF(num_tags)

    def _get_emissions(self, words, chars, lengths):
        word_embs = self.word_embedding(words)
        char_embs = self.char_cnn(chars)
        combined = torch.cat([word_embs, char_embs], dim=2)
        combined = self.dropout(combined)
        packed = nn.utils.rnn.pack_padded_sequence(combined, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        return self.hidden2tag(lstm_out)

    def loss(self, words, chars, tags, lengths):
        emissions = self._get_emissions(words, chars, lengths)
        return self.crf(emissions, tags, lengths)

    def predict(self, words, chars, lengths):
        emissions = self._get_emissions(words, chars, lengths)
        return self.crf.decode(emissions, lengths)
