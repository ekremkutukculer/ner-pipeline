import torch
import torch.nn as nn


class CharCNN(nn.Module):
    def __init__(
        self,
        num_chars: int,
        char_emb_dim: int = 50,
        num_filters: int = 50,
        kernel_size: int = 3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(num_chars, char_emb_dim, padding_idx=padding_idx)
        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word_len = char_ids.shape
        char_ids = char_ids.view(batch_size * seq_len, max_word_len)
        char_embs = self.char_embedding(char_ids)        # (B*S, W, E)
        char_embs = char_embs.transpose(1, 2)            # (B*S, E, W)
        conv_out = self.conv(char_embs)                  # (B*S, F, W)
        char_features = conv_out.max(dim=2)[0]           # (B*S, F)
        return char_features.view(batch_size, seq_len, -1)  # (B, S, F)
