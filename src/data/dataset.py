import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary


class NERDataset(Dataset):
    """PyTorch Dataset for NER. Each item is one sentence."""

    def __init__(
        self,
        sentences: List[List[str]],
        labels: List[List[str]],
        word_vocab: Vocabulary,
        label_vocab: LabelVocabulary,
        char_vocab: CharVocabulary | None = None,
    ):
        self.sentences = sentences
        self.labels = labels
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab

        if char_vocab is None:
            all_tokens = [t for s in sentences for t in s]
            self.char_vocab = CharVocabulary.build(all_tokens)
        else:
            self.char_vocab = char_vocab

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = self.sentences[idx]
        tags = self.labels[idx]

        word_ids = torch.tensor(
            [self.word_vocab.word_to_idx(t) for t in tokens], dtype=torch.long
        )
        label_ids = torch.tensor(
            [self.label_vocab.label_to_idx(t) for t in tags], dtype=torch.long
        )

        char_id_lists = []
        max_word_len = max(len(t) for t in tokens)
        for token in tokens:
            char_ids = [self.char_vocab.char_to_idx(c) for c in token]
            char_ids += [0] * (max_word_len - len(char_ids))
            char_id_lists.append(char_ids)
        char_ids_tensor = torch.tensor(char_id_lists, dtype=torch.long)

        return word_ids, char_ids_tensor, label_ids


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a batch of variable-length sentences."""
    word_seqs, char_seqs, label_seqs = zip(*batch)

    lengths = torch.tensor([len(s) for s in word_seqs], dtype=torch.long)
    max_seq_len = lengths.max().item()
    max_word_len = max(c.shape[1] for c in char_seqs)

    words_padded = pad_sequence(word_seqs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(label_seqs, batch_first=True, padding_value=0)

    batch_size = len(batch)
    chars_padded = torch.zeros(batch_size, max_seq_len, max_word_len, dtype=torch.long)
    for i, c in enumerate(char_seqs):
        seq_len, wl = c.shape
        chars_padded[i, :seq_len, :wl] = c

    return words_padded, chars_padded, labels_padded, lengths
