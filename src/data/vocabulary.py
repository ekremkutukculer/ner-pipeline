from collections import Counter
from typing import List, Optional


class Vocabulary:
    """Maps tokens to integer indices and back.

    Special tokens:
        <PAD> (index 0): padding for batch alignment
        <UNK> (index 1): unknown/out-of-vocabulary words
    """

    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self._word2idx: dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self._idx2word: dict[int, str] = {0: self.PAD, 1: self.UNK}

    def add(self, word: str) -> int:
        if word not in self._word2idx:
            idx = len(self._word2idx)
            self._word2idx[word] = idx
            self._idx2word[idx] = word
        return self._word2idx[word]

    def word_to_idx(self, word: str) -> int:
        return self._word2idx.get(word, self._word2idx[self.UNK])

    def idx_to_word(self, idx: int) -> str:
        return self._idx2word.get(idx, self.UNK)

    def __len__(self) -> int:
        return len(self._word2idx)

    def __contains__(self, word: str) -> bool:
        return word in self._word2idx

    @classmethod
    def build(cls, tokens: List[str], max_size: Optional[int] = None) -> "Vocabulary":
        vocab = cls()
        counts = Counter(tokens)
        most_common = counts.most_common(max_size - 2 if max_size else None)
        for word, _ in most_common:
            vocab.add(word)
        return vocab


class LabelVocabulary:
    """Maps BIO labels to indices. No PAD/UNK — every label must be known."""

    def __init__(self):
        self._label2idx: dict[str, int] = {}
        self._idx2label: dict[int, str] = {}

    def add(self, label: str) -> int:
        if label not in self._label2idx:
            idx = len(self._label2idx)
            self._label2idx[label] = idx
            self._idx2label[idx] = label
        return self._label2idx[label]

    def label_to_idx(self, label: str) -> int:
        return self._label2idx[label]

    def idx_to_label(self, idx: int) -> str:
        return self._idx2label[idx]

    def __len__(self) -> int:
        return len(self._label2idx)

    @classmethod
    def build(cls, labels: List[str]) -> "LabelVocabulary":
        vocab = cls()
        for label in sorted(set(labels)):
            vocab.add(label)
        return vocab


class CharVocabulary:
    """Maps individual characters to indices."""

    PAD = "<PAD>"

    def __init__(self):
        self._char2idx: dict[str, int] = {self.PAD: 0}

    def add(self, char: str) -> int:
        if char not in self._char2idx:
            self._char2idx[char] = len(self._char2idx)
        return self._char2idx[char]

    def char_to_idx(self, char: str) -> int:
        return self._char2idx.get(char, 0)

    def __len__(self) -> int:
        return len(self._char2idx)

    @classmethod
    def build(cls, tokens: List[str]) -> "CharVocabulary":
        vocab = cls()
        for token in tokens:
            for char in token:
                vocab.add(char)
        return vocab
