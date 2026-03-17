# NER Pipeline Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a BiLSTM-CRF Named Entity Recognition pipeline with training, evaluation, REST API, and web demo.

**Architecture:** PyTorch BiLSTM-CRF model trained on CoNLL-2003. Data loaded via HuggingFace Datasets, preprocessed into vocabulary-indexed tensors. FastAPI serves predictions, Gradio provides an interactive demo.

**Tech Stack:** Python 3.10+, PyTorch, GloVe 300d, seqeval, FastAPI, Gradio, SpaCy, TensorBoard, pytest

---

## File Structure

```
ner-pipeline/
├── configs/
│   └── conll2003.yaml              # Training config (hyperparameters, paths)
├── data/
│   └── glove/                      # Downloaded GloVe embeddings
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── vocabulary.py           # Word + char + label vocabularies
│   │   ├── dataset.py              # PyTorch Dataset for CoNLL format
│   │   └── preprocessing.py        # Load HuggingFace data, build vocabs
│   ├── models/
│   │   ├── __init__.py
│   │   ├── char_cnn.py             # Character-level CNN embedder
│   │   ├── crf.py                  # CRF layer (forward, decode, loss)
│   │   └── bilstm_crf.py          # Full BiLSTM-CRF model
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py              # Training loop, early stopping, checkpoints
│   │   └── metrics.py              # seqeval wrapper for NER metrics
│   └── inference/
│       ├── __init__.py
│       └── predictor.py            # Load model, tokenize, predict
├── api/
│   ├── __init__.py
│   └── main.py                     # FastAPI app with /predict endpoint
├── demo/
│   └── app.py                      # Gradio demo
├── scripts/
│   ├── download_glove.py           # Download GloVe embeddings
│   └── train.py                    # Training entry point
├── tests/
│   ├── __init__.py
│   ├── test_vocabulary.py
│   ├── test_dataset.py
│   ├── test_char_cnn.py
│   ├── test_crf.py
│   ├── test_bilstm_crf.py
│   ├── test_metrics.py
│   ├── test_trainer.py
│   ├── test_predictor.py
│   └── test_api.py
├── notebooks/                          # Exploration and analysis
├── requirements.txt
└── README.md
```

---

## Chunk 1: Project Setup & Data Layer

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `configs/conll2003.yaml`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/models/__init__.py`, `src/training/__init__.py`, `src/inference/__init__.py`
- Create: `tests/__init__.py`, `api/__init__.py`
- Create: `notebooks/.gitkeep`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0.0
datasets>=2.14.0
seqeval>=1.2.2
fastapi>=0.104.0
uvicorn>=0.24.0
gradio>=4.0.0
pyyaml>=6.0
pytest>=7.4.0
spacy>=3.7.0
tensorboard>=2.15.0
httpx>=0.25.0
```

- [ ] **Step 2: Create training config**

```yaml
# configs/conll2003.yaml
data:
  dataset_name: conll2003
  glove_path: data/glove/glove.6B.300d.txt
  glove_dim: 300
  max_vocab_size: 50000

model:
  char_embedding_dim: 50
  char_conv_filters: 50
  char_conv_kernel: 3
  hidden_size: 256
  num_lstm_layers: 1
  dropout: 0.5

training:
  optimizer: adam
  learning_rate: 0.001
  gradient_clip: 5.0
  batch_size: 64
  max_epochs: 50
  early_stopping_patience: 5
  checkpoint_dir: checkpoints/conll2003

logging:
  tensorboard_dir: runs/conll2003
```

- [ ] **Step 3: Create all `__init__.py` files**

All empty files. Creates the Python package structure.

- [ ] **Step 4: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Run: `python -c "import torch; print(torch.__version__)"`
Expected: PyTorch version prints without error.

- [ ] **Step 5: Download SpaCy model**

Run: `python -m spacy download en_core_web_sm`

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "feat: project scaffolding with dependencies and config"
```

---

### Task 2: Vocabulary classes

**Files:**
- Create: `src/data/vocabulary.py`
- Create: `tests/test_vocabulary.py`

**Kavram — Vocabulary nedir?**
Model sayılarla çalışır, kelimelerle değil. Vocabulary, her kelimeye bir numara atar: {"the": 0, "cat": 1, "sat": 2, ...}. Ayrıca iki özel token vardır:
- `<PAD>`: Batch'lerde kısa cümleleri uzunlarla eşitlemek için kullanılır
- `<UNK>`: Eğitimde hiç görülmemiş kelimeler için kullanılır

- [ ] **Step 1: Write failing tests for Vocabulary**

```python
# tests/test_vocabulary.py
import pytest
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary


class TestVocabulary:
    def test_add_and_lookup(self):
        vocab = Vocabulary()
        idx = vocab.add("hello")
        assert vocab.word_to_idx("hello") == idx
        assert vocab.idx_to_word(idx) == "hello"

    def test_pad_and_unk_exist(self):
        vocab = Vocabulary()
        assert vocab.word_to_idx("<PAD>") == 0
        assert vocab.word_to_idx("<UNK>") == 1

    def test_unknown_word_returns_unk(self):
        vocab = Vocabulary()
        assert vocab.word_to_idx("nonexistent") == vocab.word_to_idx("<UNK>")

    def test_len(self):
        vocab = Vocabulary()
        vocab.add("a")
        vocab.add("b")
        assert len(vocab) == 4  # PAD + UNK + a + b

    def test_build_from_tokens(self):
        tokens = ["the", "cat", "sat", "the", "the", "dog"]
        vocab = Vocabulary.build(tokens, max_size=4)
        # PAD + UNK + "the"(3) + "cat"(1) or "sat"(1) or "dog"(1)
        assert len(vocab) <= 4
        assert vocab.word_to_idx("the") != vocab.word_to_idx("<UNK>")

    def test_no_duplicates(self):
        vocab = Vocabulary()
        idx1 = vocab.add("hello")
        idx2 = vocab.add("hello")
        assert idx1 == idx2
        assert len(vocab) == 3  # PAD + UNK + hello


class TestLabelVocabulary:
    def test_build_and_lookup(self):
        labels = ["O", "B-PER", "I-PER", "B-LOC", "O"]
        vocab = LabelVocabulary.build(labels)
        assert len(vocab) == 4  # O, B-PER, I-PER, B-LOC (sorted unique)
        assert vocab.idx_to_label(vocab.label_to_idx("B-PER")) == "B-PER"

    def test_unknown_label_raises(self):
        vocab = LabelVocabulary.build(["O", "B-PER"])
        with pytest.raises(KeyError):
            vocab.label_to_idx("B-LOC")

    def test_sorted_order(self):
        labels = ["O", "B-PER", "B-LOC"]
        vocab = LabelVocabulary.build(labels)
        # Sorted: B-LOC=0, B-PER=1, O=2
        assert vocab.label_to_idx("B-LOC") == 0


class TestCharVocabulary:
    def test_build_from_tokens(self):
        vocab = CharVocabulary.build(["hello", "world"])
        assert len(vocab) > 1  # PAD + unique chars
        assert vocab.char_to_idx("h") != 0  # not PAD

    def test_unknown_char_returns_zero(self):
        vocab = CharVocabulary.build(["abc"])
        assert vocab.char_to_idx("z") == 0  # unknown → PAD idx
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_vocabulary.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.vocabulary'`

- [ ] **Step 3: Implement Vocabulary**

```python
# src/data/vocabulary.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_vocabulary.py -v`
Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/vocabulary.py tests/test_vocabulary.py
git commit -m "feat: vocabulary classes for word, char, and label indexing"
```

---

### Task 3: Dataset class

**Files:**
- Create: `src/data/dataset.py`
- Create: `tests/test_dataset.py`

**Kavram — PyTorch Dataset nedir?**
PyTorch modele veri beslemek için `Dataset` ve `DataLoader` kullanır. `Dataset` tek bir örneği döner (bir cümle + etiketler), `DataLoader` bunları batch'ler halinde gruplar ve padding ekler.

- [ ] **Step 1: Write failing tests for NERDataset**

```python
# tests/test_dataset.py
import pytest
import torch
from src.data.vocabulary import Vocabulary, LabelVocabulary
from src.data.dataset import NERDataset, collate_fn


@pytest.fixture
def sample_data():
    sentences = [
        ["John", "lives", "in", "London"],
        ["Apple", "is", "big"],
    ]
    labels = [
        ["B-PER", "O", "O", "B-LOC"],
        ["B-ORG", "O", "O"],
    ]
    return sentences, labels


@pytest.fixture
def vocabs(sample_data):
    sentences, labels = sample_data
    all_tokens = [t for s in sentences for t in s]
    all_labels = [l for ls in labels for l in ls]
    word_vocab = Vocabulary.build(all_tokens)
    label_vocab = LabelVocabulary.build(all_labels)
    return word_vocab, label_vocab


class TestNERDataset:
    def test_length(self, sample_data, vocabs):
        sentences, labels = sample_data
        word_vocab, label_vocab = vocabs
        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        assert len(dataset) == 2

    def test_getitem_returns_tensors(self, sample_data, vocabs):
        sentences, labels = sample_data
        word_vocab, label_vocab = vocabs
        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        word_ids, char_ids, label_ids = dataset[0]
        assert isinstance(word_ids, torch.Tensor)
        assert isinstance(char_ids, torch.Tensor)
        assert isinstance(label_ids, torch.Tensor)
        assert len(word_ids) == 4  # "John lives in London"

    def test_char_ids_shape(self, sample_data, vocabs):
        sentences, labels = sample_data
        word_vocab, label_vocab = vocabs
        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        word_ids, char_ids, label_ids = dataset[0]
        # char_ids shape: (num_words, max_word_len)
        assert char_ids.dim() == 2
        assert char_ids.shape[0] == 4

    def test_collate_fn_pads(self, sample_data, vocabs):
        sentences, labels = sample_data
        word_vocab, label_vocab = vocabs
        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        batch = [dataset[0], dataset[1]]
        words, chars, labs, lengths = collate_fn(batch)
        assert words.shape[0] == 2  # batch size
        assert words.shape[1] == 4  # padded to longest sentence
        assert lengths.tolist() == [4, 3]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement NERDataset**

```python
# src/data/dataset.py
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

        # Character IDs: (num_words, max_word_len_in_sentence)
        char_id_lists = []
        max_word_len = max(len(t) for t in tokens)
        for token in tokens:
            char_ids = [self.char_vocab.char_to_idx(c) for c in token]
            char_ids += [0] * (max_word_len - len(char_ids))  # pad chars
            char_id_lists.append(char_ids)
        char_ids_tensor = torch.tensor(char_id_lists, dtype=torch.long)

        return word_ids, char_ids_tensor, label_ids


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pads a batch of variable-length sentences.

    Returns:
        words: (batch_size, max_seq_len)
        chars: (batch_size, max_seq_len, max_word_len)
        labels: (batch_size, max_seq_len)
        lengths: (batch_size,)
    """
    word_seqs, char_seqs, label_seqs = zip(*batch)

    lengths = torch.tensor([len(s) for s in word_seqs], dtype=torch.long)
    max_seq_len = lengths.max().item()

    # Find global max word length across all sentences in the batch
    max_word_len = max(c.shape[1] for c in char_seqs)

    # Pad words and labels
    words_padded = pad_sequence(word_seqs, batch_first=True, padding_value=0)
    labels_padded = pad_sequence(label_seqs, batch_first=True, padding_value=0)

    # Pad chars: need (batch, max_seq_len, max_word_len)
    batch_size = len(batch)
    chars_padded = torch.zeros(batch_size, max_seq_len, max_word_len, dtype=torch.long)
    for i, c in enumerate(char_seqs):
        seq_len, wl = c.shape
        chars_padded[i, :seq_len, :wl] = c

    return words_padded, chars_padded, labels_padded, lengths
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_dataset.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: NER dataset with char-level encoding and batch collation"
```

---

### Task 4: Data preprocessing (HuggingFace → our format)

**Files:**
- Create: `src/data/preprocessing.py`
- Create: `scripts/download_glove.py`

**Kavram — Preprocessing nedir?**
HuggingFace'den gelen veriyi bizim Vocabulary + Dataset sınıflarımıza uygun hale getirme. Ayrıca GloVe embedding'lerini yükleme.

- [ ] **Step 1: Implement preprocessing**

```python
# src/data/preprocessing.py
import numpy as np
import torch
from datasets import load_dataset
from typing import Tuple
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary
from src.data.dataset import NERDataset


# CoNLL-2003 label mapping from HuggingFace integer IDs
CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_conll2003() -> dict:
    """Load CoNLL-2003 from HuggingFace and convert to token/label lists."""
    dataset = load_dataset("conll2003", trust_remote_code=True)
    result = {}
    for split in ["train", "validation", "test"]:
        sentences = []
        labels = []
        for example in dataset[split]:
            tokens = example["tokens"]
            ner_tags = [CONLL_LABELS[t] for t in example["ner_tags"]]
            sentences.append(tokens)
            labels.append(ner_tags)
        result[split] = (sentences, labels)
    return result


def build_vocabs(
    train_sentences: list, train_labels: list, max_vocab_size: int = 50000
) -> Tuple[Vocabulary, LabelVocabulary, CharVocabulary]:
    """Build word, label, and char vocabularies from training data."""
    all_tokens = [t for s in train_sentences for t in s]
    all_labels = [l for ls in train_labels for l in ls]

    word_vocab = Vocabulary.build(all_tokens, max_size=max_vocab_size)
    label_vocab = LabelVocabulary.build(all_labels)
    char_vocab = CharVocabulary.build(all_tokens)

    return word_vocab, label_vocab, char_vocab


def load_glove_embeddings(
    glove_path: str, word_vocab: Vocabulary, dim: int = 300
) -> torch.Tensor:
    """Load GloVe vectors for words in our vocabulary.

    Returns an embedding matrix of shape (vocab_size, dim).
    Words not in GloVe get random initialization.
    """
    embeddings = np.random.uniform(-0.25, 0.25, (len(word_vocab), dim))
    embeddings[0] = np.zeros(dim)  # PAD gets zero vector

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word_vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word_vocab.word_to_idx(word)] = vector
                found += 1

    print(f"GloVe: found {found}/{len(word_vocab)} words ({100*found/len(word_vocab):.1f}%)")
    return torch.tensor(embeddings, dtype=torch.float32)


def create_datasets(
    data: dict,
    word_vocab: Vocabulary,
    label_vocab: LabelVocabulary,
    char_vocab: CharVocabulary,
) -> dict:
    """Create NERDataset objects for each split."""
    datasets = {}
    for split, (sentences, labels) in data.items():
        datasets[split] = NERDataset(sentences, labels, word_vocab, label_vocab, char_vocab)
    return datasets
```

- [ ] **Step 2: Create GloVe download script**

```python
# scripts/download_glove.py
"""Download GloVe embeddings (300d, 6B tokens)."""
import os
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"
GLOVE_DIR = os.path.join("data", "glove")
GLOVE_FILE = os.path.join(GLOVE_DIR, "glove.6B.300d.txt")


def main():
    if os.path.exists(GLOVE_FILE):
        print(f"GloVe already exists at {GLOVE_FILE}")
        return

    os.makedirs(GLOVE_DIR, exist_ok=True)
    zip_path = os.path.join(GLOVE_DIR, "glove.6B.zip")

    print(f"Downloading GloVe from {GLOVE_URL}...")
    print("This is ~862MB and may take a few minutes.")
    urllib.request.urlretrieve(GLOVE_URL, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract("glove.6B.300d.txt", GLOVE_DIR)

    os.remove(zip_path)
    print(f"Done. GloVe saved to {GLOVE_FILE}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Verify preprocessing loads data**

Run: `python -c "from src.data.preprocessing import load_conll2003; d = load_conll2003(); print(f'Train: {len(d[\"train\"][0])} sentences')"`
Expected: `Train: 14041 sentences`

- [ ] **Step 4: Download GloVe**

Run: `python scripts/download_glove.py`
Expected: GloVe file appears at `data/glove/glove.6B.300d.txt`

- [ ] **Step 5: Add data/ to .gitignore**

```
# .gitignore
data/glove/
checkpoints/
runs/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 6: Commit**

```bash
git add src/data/preprocessing.py scripts/download_glove.py .gitignore
git commit -m "feat: data preprocessing, GloVe loading, and HuggingFace integration"
```

---

## Chunk 2: Model Architecture

### Task 5: Character CNN

**Files:**
- Create: `src/models/char_cnn.py`
- Create: `tests/test_char_cnn.py`

**Kavram — Character CNN nedir?**
Her kelimenin harflerine bakarak bir vektör üretir. Örneğin "PyTorch" kelimesini hiç görmemiş olsa bile, "P-y-T-o-r-c-h" harflerinden bu kelimenin büyük harfle başladığını, teknik bir terim olabileceğini anlar.

Nasıl çalışır:
1. Her harf bir embedding vektörüne dönüşür (50 boyut)
2. 1D convolution harflerin ardışık örüntülerini yakalar ("ing", "tion" gibi)
3. Max-pooling en önemli örüntüyü seçer → sabit boyutlu bir kelime vektörü çıkar

- [ ] **Step 1: Write failing tests**

```python
# tests/test_char_cnn.py
import pytest
import torch
from src.models.char_cnn import CharCNN


class TestCharCNN:
    def test_output_shape(self):
        # batch=2, seq_len=3, max_word_len=5
        char_ids = torch.randint(0, 26, (2, 3, 5))
        model = CharCNN(num_chars=30, char_emb_dim=50, num_filters=50, kernel_size=3)
        output = model(char_ids)
        assert output.shape == (2, 3, 50)  # (batch, seq_len, num_filters)

    def test_different_word_lengths(self):
        # Words of different lengths (padded to max)
        char_ids = torch.randint(0, 26, (1, 4, 10))
        model = CharCNN(num_chars=30, char_emb_dim=50, num_filters=50, kernel_size=3)
        output = model(char_ids)
        assert output.shape == (1, 4, 50)

    def test_padding_invariance(self):
        """Padding (zeros) should not significantly change output for real chars."""
        model = CharCNN(num_chars=30, char_emb_dim=50, num_filters=50, kernel_size=3)
        model.eval()
        # Same word, different padding
        chars1 = torch.tensor([[[1, 2, 3, 0, 0]]])  # "abc" + padding
        chars2 = torch.tensor([[[1, 2, 3, 0, 0]]])
        out1 = model(chars1)
        out2 = model(chars2)
        assert torch.equal(out1, out2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_char_cnn.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CharCNN**

```python
# src/models/char_cnn.py
import torch
import torch.nn as nn


class CharCNN(nn.Module):
    """Character-level CNN that produces a fixed-size vector per word.

    Input: char IDs of shape (batch, seq_len, max_word_len)
    Output: char features of shape (batch, seq_len, num_filters)

    How it works:
    1. Embed each character into a dense vector
    2. Apply 1D convolution over the character sequence
    3. Max-pool over the sequence to get one vector per word
    """

    def __init__(
        self,
        num_chars: int,
        char_emb_dim: int = 50,
        num_filters: int = 50,
        kernel_size: int = 3,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(
            num_chars, char_emb_dim, padding_idx=padding_idx
        )
        self.conv = nn.Conv1d(
            in_channels=char_emb_dim,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,  # same padding
        )

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_word_len = char_ids.shape

        # Reshape to (batch * seq_len, max_word_len) for embedding
        char_ids = char_ids.view(batch_size * seq_len, max_word_len)

        # Embed: (batch*seq_len, max_word_len, char_emb_dim)
        char_embs = self.char_embedding(char_ids)

        # Conv1d expects (batch, channels, length)
        char_embs = char_embs.transpose(1, 2)

        # Convolve: (batch*seq_len, num_filters, max_word_len)
        conv_out = self.conv(char_embs)

        # Max-pool over word length: (batch*seq_len, num_filters)
        char_features = conv_out.max(dim=2)[0]

        # Reshape back: (batch, seq_len, num_filters)
        return char_features.view(batch_size, seq_len, -1)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_char_cnn.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/char_cnn.py tests/test_char_cnn.py
git commit -m "feat: character CNN for sub-word feature extraction"
```

---

### Task 6: CRF Layer

**Files:**
- Create: `src/models/crf.py`
- Create: `tests/test_crf.py`

**Kavram — CRF (Conditional Random Field) nedir?**
CRF, tag dizisinin "mantıklı" olmasını sağlar. BiLSTM her token için bağımsız bir tahmin yapar, ama CRF tüm diziye birlikte bakar.

Örnek: BiLSTM "B-PER I-ORG" diyebilir (saçma — bir kişi ismi organizasyon olarak devam edemez). CRF bir geçiş matrisi öğrenir:
- B-PER → I-PER: yüksek skor ✓
- B-PER → I-ORG: düşük skor ✗

İki ana işlem:
- **Forward (eğitim):** Negatif log-likelihood loss hesaplar
- **Decode (inference):** Viterbi algoritması ile en iyi tag dizisini bulur

- [ ] **Step 1: Write failing tests**

```python
# tests/test_crf.py
import pytest
import torch
from src.models.crf import CRF


class TestCRF:
    def test_decode_returns_valid_shape(self):
        num_tags = 5
        crf = CRF(num_tags)
        # emissions: (batch=2, seq_len=3, num_tags)
        emissions = torch.randn(2, 3, num_tags)
        lengths = torch.tensor([3, 2])
        best_tags = crf.decode(emissions, lengths)
        assert len(best_tags) == 2
        assert len(best_tags[0]) == 3
        assert len(best_tags[1]) == 2

    def test_tags_are_valid_indices(self):
        num_tags = 5
        crf = CRF(num_tags)
        emissions = torch.randn(1, 4, num_tags)
        lengths = torch.tensor([4])
        best_tags = crf.decode(emissions, lengths)
        for tag in best_tags[0]:
            assert 0 <= tag < num_tags

    def test_loss_is_scalar(self):
        num_tags = 5
        crf = CRF(num_tags)
        emissions = torch.randn(2, 3, num_tags)
        tags = torch.randint(0, num_tags, (2, 3))
        lengths = torch.tensor([3, 2])
        loss = crf.forward(emissions, tags, lengths)
        assert loss.dim() == 0  # scalar

    def test_loss_decreases_with_training(self):
        """CRF should learn to assign higher score to correct tags."""
        num_tags = 3
        crf = CRF(num_tags)
        optimizer = torch.optim.Adam(crf.parameters(), lr=0.1)

        emissions = torch.randn(1, 5, num_tags)
        tags = torch.tensor([[0, 1, 2, 0, 1]])
        lengths = torch.tensor([5])

        loss_before = crf.forward(emissions, tags, lengths).item()
        for _ in range(50):
            loss = crf.forward(emissions, tags, lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_after = crf.forward(emissions, tags, lengths).item()
        assert loss_after < loss_before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crf.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CRF**

```python
# src/models/crf.py
import torch
import torch.nn as nn
from typing import List


class CRF(nn.Module):
    """Linear-chain Conditional Random Field.

    Learns a transition matrix between tags so that the model produces
    valid tag sequences (e.g., B-PER followed by I-PER, not I-ORG).

    Key concepts:
    - transitions[i][j] = score of transitioning from tag i to tag j
    - start_transitions[i] = score of starting with tag i
    - end_transitions[i] = score of ending with tag i
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(
        self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative log-likelihood loss.

        Args:
            emissions: (batch, seq_len, num_tags) — scores from BiLSTM
            tags: (batch, seq_len) — gold labels
            lengths: (batch,) — actual sentence lengths
        Returns:
            Scalar loss (mean over batch).
        """
        log_numerator = self._compute_score(emissions, tags, lengths)
        log_denominator = self._compute_log_partition(emissions, lengths)
        return (log_denominator - log_numerator).mean()

    def _compute_score(
        self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Score of the gold tag sequence."""
        batch_size, seq_len, _ = emissions.shape

        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)

        for t in range(1, seq_len):
            mask = (t < lengths).float()
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit_score + trans_score) * mask

        # End transition
        last_idx = (lengths - 1).long()
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]

        return score

    def _compute_log_partition(
        self, emissions: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Log-sum-exp over all possible tag sequences (forward algorithm)."""
        batch_size, seq_len, num_tags = emissions.shape

        # alpha[i] = log-sum-exp of all paths ending at tag i
        alpha = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            mask = (t < lengths).float().unsqueeze(1)
            emit = emissions[:, t].unsqueeze(1)  # (batch, 1, num_tags)
            trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            scores = alpha.unsqueeze(2) + emit + trans  # (batch, num_tags, num_tags)
            new_alpha = torch.logsumexp(scores, dim=1)  # (batch, num_tags)
            alpha = new_alpha * mask + alpha * (1 - mask)

        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(
        self, emissions: torch.Tensor, lengths: torch.Tensor
    ) -> List[List[int]]:
        """Find best tag sequence using Viterbi algorithm.

        Viterbi is like the forward algorithm but uses max instead of logsumexp.
        It finds the single highest-scoring tag sequence.
        """
        batch_size, seq_len, num_tags = emissions.shape

        # viterbi[i] = best score of paths ending at tag i
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []

        for t in range(1, seq_len):
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_scores, best_prev = scores.max(dim=1)
            viterbi = max_scores + emissions[:, t]
            backpointers.append(best_prev)

        viterbi += self.end_transitions

        # Backtrace
        best_paths = []
        for b in range(batch_size):
            length = lengths[b].item()
            best_last = viterbi[b].argmax().item()
            path = [best_last]
            for t in range(length - 2, -1, -1):
                best_last = backpointers[t][b][best_last].item()
                path.append(best_last)
            path.reverse()
            best_paths.append(path[:length])

        return best_paths
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_crf.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/crf.py tests/test_crf.py
git commit -m "feat: CRF layer with forward algorithm and Viterbi decoding"
```

---

### Task 7: BiLSTM-CRF Model

**Files:**
- Create: `src/models/bilstm_crf.py`
- Create: `tests/test_bilstm_crf.py`

**Kavram — Tüm parçalar nasıl birleşir?**
```
Kelime → GloVe embedding (300d) ─┐
                                  ├─ Concat (350d) → BiLSTM (512d) → Linear (num_tags) → CRF → Tags
Harfler → CharCNN (50d) ─────────┘
```

BiLSTM çıktısı her token için bir skor vektörü verir (her tag için bir skor). CRF bu skorları alıp en mantıklı tag dizisini seçer.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bilstm_crf.py
import pytest
import torch
from src.models.bilstm_crf import BiLSTMCRF


@pytest.fixture
def model():
    return BiLSTMCRF(
        vocab_size=100,
        num_chars=50,
        num_tags=9,
        word_emb_dim=300,
        char_emb_dim=50,
        char_filters=50,
        hidden_size=256,
        dropout=0.5,
    )


class TestBiLSTMCRF:
    def test_loss_output(self, model):
        words = torch.randint(0, 100, (2, 5))
        chars = torch.randint(0, 50, (2, 5, 8))
        tags = torch.randint(0, 9, (2, 5))
        lengths = torch.tensor([5, 3])
        loss = model.loss(words, chars, tags, lengths)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_predict_output(self, model):
        model.eval()
        words = torch.randint(0, 100, (2, 5))
        chars = torch.randint(0, 50, (2, 5, 8))
        lengths = torch.tensor([5, 3])
        paths = model.predict(words, chars, lengths)
        assert len(paths) == 2
        assert len(paths[0]) == 5
        assert len(paths[1]) == 3

    def test_pretrained_embeddings(self):
        pretrained = torch.randn(100, 300)
        model = BiLSTMCRF(
            vocab_size=100,
            num_chars=50,
            num_tags=9,
            pretrained_word_emb=pretrained,
        )
        assert torch.equal(model.word_embedding.weight.data, pretrained)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bilstm_crf.py -v`
Expected: FAIL

- [ ] **Step 3: Implement BiLSTMCRF**

```python
# src/models/bilstm_crf.py
import torch
import torch.nn as nn
from typing import List, Optional
from src.models.char_cnn import CharCNN
from src.models.crf import CRF


class BiLSTMCRF(nn.Module):
    """Bidirectional LSTM with CRF for sequence labeling.

    Architecture:
        Word Embedding (GloVe) + Char CNN → BiLSTM → Linear → CRF

    The BiLSTM reads the sentence left-to-right AND right-to-left,
    giving each token a context-aware representation. The linear layer
    projects this to tag-space scores, and the CRF picks the best
    globally consistent tag sequence.
    """

    def __init__(
        self,
        vocab_size: int,
        num_chars: int,
        num_tags: int,
        word_emb_dim: int = 300,
        char_emb_dim: int = 50,
        char_filters: int = 50,
        char_kernel: int = 3,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.5,
        pretrained_word_emb: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        # Word embedding
        self.word_embedding = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        if pretrained_word_emb is not None:
            self.word_embedding.weight.data.copy_(pretrained_word_emb)

        # Char CNN
        self.char_cnn = CharCNN(num_chars, char_emb_dim, char_filters, char_kernel)

        # BiLSTM
        input_dim = word_emb_dim + char_filters
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        # Project BiLSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_size * 2, num_tags)

        # CRF
        self.crf = CRF(num_tags)

    def _get_emissions(
        self, words: torch.Tensor, chars: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Run input through embeddings + BiLSTM + linear to get emission scores."""
        word_embs = self.word_embedding(words)
        char_embs = self.char_cnn(chars)

        # Concatenate word + char embeddings
        combined = torch.cat([word_embs, char_embs], dim=2)
        combined = self.dropout(combined)

        # Pack for efficient LSTM processing (ignores padding)
        packed = nn.utils.rnn.pack_padded_sequence(
            combined, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def loss(
        self,
        words: torch.Tensor,
        chars: torch.Tensor,
        tags: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CRF negative log-likelihood loss."""
        emissions = self._get_emissions(words, chars, lengths)
        return self.crf(emissions, tags, lengths)

    def predict(
        self,
        words: torch.Tensor,
        chars: torch.Tensor,
        lengths: torch.Tensor,
    ) -> List[List[int]]:
        """Predict best tag sequence using Viterbi decoding."""
        emissions = self._get_emissions(words, chars, lengths)
        return self.crf.decode(emissions, lengths)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_bilstm_crf.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/models/bilstm_crf.py tests/test_bilstm_crf.py
git commit -m "feat: BiLSTM-CRF model combining word embeddings, char CNN, and CRF"
```

---

## Chunk 3: Training Pipeline

### Task 8: Metrics

**Files:**
- Create: `src/training/metrics.py`
- Create: `tests/test_metrics.py` (in test_trainer.py)

**Kavram — NER Metrikleri**
NER'de accuracy yerine F1 score kullanılır çünkü çoğu token "O" (entity değil) olur. %95 accuracy elde edip hiçbir entity'yi doğru bulamayabilirsin. F1, precision ve recall'ın harmonik ortalamasıdır:
- **Precision:** Bulduğun entity'lerin kaçı doğru?
- **Recall:** Gerçek entity'lerin kaçını buldun?
- **F1:** İkisinin dengesi

- [ ] **Step 1: Implement metrics wrapper**

```python
# src/training/metrics.py
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from typing import List
from src.data.vocabulary import LabelVocabulary


def compute_metrics(
    predictions: List[List[int]],
    targets: List[List[int]],
    label_vocab: LabelVocabulary,
    lengths: List[int],
) -> dict:
    """Compute entity-level NER metrics using seqeval.

    Args:
        predictions: List of predicted tag index sequences
        targets: List of gold tag index sequences
        label_vocab: Maps indices back to BIO labels
        lengths: Actual sentence lengths (to ignore padding)
    Returns:
        Dict with precision, recall, f1, and detailed report string.
    """
    pred_labels = []
    true_labels = []

    for pred_seq, true_seq, length in zip(predictions, targets, lengths):
        pred_labels.append(
            [label_vocab.idx_to_label(p) for p in pred_seq[:length]]
        )
        true_labels.append(
            [label_vocab.idx_to_label(t) for t in true_seq[:length]]
        )

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "report": classification_report(true_labels, pred_labels),
    }
```

- [ ] **Step 2: Write tests for metrics**

```python
# tests/test_metrics.py
import pytest
from src.data.vocabulary import LabelVocabulary
from src.training.metrics import compute_metrics


class TestComputeMetrics:
    @pytest.fixture
    def label_vocab(self):
        return LabelVocabulary.build(
            ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC"]
        )

    def test_perfect_predictions(self, label_vocab):
        preds = [[label_vocab.label_to_idx(l) for l in ["B-PER", "I-PER", "O"]]]
        targets = [[label_vocab.label_to_idx(l) for l in ["B-PER", "I-PER", "O"]]]
        metrics = compute_metrics(preds, targets, label_vocab, [3])
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self, label_vocab):
        preds = [[label_vocab.label_to_idx(l) for l in ["O", "O", "O"]]]
        targets = [[label_vocab.label_to_idx(l) for l in ["B-PER", "I-PER", "O"]]]
        metrics = compute_metrics(preds, targets, label_vocab, [3])
        assert metrics["recall"] == 0.0

    def test_respects_lengths(self, label_vocab):
        # Padding after length should be ignored
        preds = [[label_vocab.label_to_idx(l) for l in ["B-PER", "O", "B-ORG"]]]
        targets = [[label_vocab.label_to_idx(l) for l in ["B-PER", "O", "O"]]]
        metrics = compute_metrics(preds, targets, label_vocab, [2])  # ignore last token
        assert metrics["f1"] == 1.0
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_metrics.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add src/training/metrics.py tests/test_metrics.py
git commit -m "feat: NER metrics wrapper using seqeval"
```

---

### Task 9: Trainer

**Files:**
- Create: `src/training/trainer.py`
- Create: `tests/test_trainer.py`
- Create: `scripts/train.py`

**Kavram — Training Loop nedir?**
Modeli eğitme döngüsü:
1. Batch al → Loss hesapla → Gradyanları hesapla → Ağırlıkları güncelle
2. Her epoch sonunda validation setinde değerlendir
3. En iyi modeli kaydet (checkpoint)
4. F1 artmazsa erken dur (early stopping)

- [ ] **Step 1: Implement Trainer**

```python
# src/training/trainer.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Optional
from src.models.bilstm_crf import BiLSTMCRF
from src.data.dataset import collate_fn
from src.data.vocabulary import LabelVocabulary
from src.training.metrics import compute_metrics


class Trainer:
    """Handles the full training loop with early stopping and checkpointing.

    What happens each epoch:
    1. Train: iterate over batches, compute loss, backprop, update weights
    2. Evaluate: run model on validation set, compute F1
    3. Checkpoint: save model if validation F1 improved
    4. Early stop: if F1 hasn't improved for `patience` epochs, stop
    """

    def __init__(
        self,
        model: BiLSTMCRF,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_vocab: LabelVocabulary,
        learning_rate: float = 1e-3,
        gradient_clip: float = 5.0,
        max_epochs: int = 50,
        patience: int = 5,
        checkpoint_dir: str = "checkpoints",
        tensorboard_dir: str = "runs",
        device: Optional[str] = None,
        vocabs: dict | None = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_vocab = label_vocab
        self.vocabs = vocabs
        self.gradient_clip = gradient_clip
        self.max_epochs = max_epochs
        self.patience = patience
        self.checkpoint_dir = checkpoint_dir

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.writer = SummaryWriter(tensorboard_dir)

        self.best_f1 = 0.0
        self.epochs_without_improvement = 0

    def train(self) -> dict:
        """Run the full training loop. Returns best metrics."""
        best_metrics = {}

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._evaluate()

            # Log to TensorBoard
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            self.writer.add_scalar("Precision/val", val_metrics["precision"], epoch)
            self.writer.add_scalar("Recall/val", val_metrics["recall"], epoch)

            print(
                f"Epoch {epoch}/{self.max_epochs} — "
                f"Loss: {train_loss:.4f} — "
                f"Val F1: {val_metrics['f1']:.4f}"
            )

            # Checkpoint best model
            if val_metrics["f1"] > self.best_f1:
                self.best_f1 = val_metrics["f1"]
                self.epochs_without_improvement = 0
                best_metrics = val_metrics
                self._save_checkpoint(epoch)
                print(f"  ↑ New best F1: {self.best_f1:.4f}")
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

        self.writer.close()
        return best_metrics

    def _train_epoch(self) -> float:
        """One training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for words, chars, labels, lengths in self.train_loader:
            words = words.to(self.device)
            chars = chars.to(self.device)
            labels = labels.to(self.device)
            lengths = lengths.to(self.device)

            self.optimizer.zero_grad()
            loss = self.model.loss(words, chars, labels, lengths)
            loss.backward()

            # Clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def _evaluate(self) -> dict:
        """Evaluate on validation set. Returns metrics dict."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_lengths = []

        with torch.no_grad():
            for words, chars, labels, lengths in self.val_loader:
                words = words.to(self.device)
                chars = chars.to(self.device)
                lengths = lengths.to(self.device)

                predictions = self.model.predict(words, chars, lengths)
                all_predictions.extend(predictions)
                all_targets.extend(labels.tolist())
                all_lengths.extend(lengths.tolist())

        return compute_metrics(
            all_predictions, all_targets, self.label_vocab, all_lengths
        )

    def _save_checkpoint(self, epoch: int):
        """Save model weights and vocabularies."""
        import pickle
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_f1": self.best_f1,
            },
            path,
        )
        if self.vocabs:
            vocab_path = os.path.join(self.checkpoint_dir, "vocabs.pkl")
            with open(vocab_path, "wb") as f:
                pickle.dump(self.vocabs, f)
```

- [ ] **Step 2: Create training script**

```python
# scripts/train.py
"""Training entry point. Usage: python scripts/train.py --config configs/conll2003.yaml"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.preprocessing import load_conll2003, build_vocabs, load_glove_embeddings, create_datasets
from src.data.dataset import collate_fn
from src.models.bilstm_crf import BiLSTMCRF
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/conll2003.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Load data
    print("Loading CoNLL-2003...")
    data = load_conll2003()
    train_sents, train_labels = data["train"]

    # Build vocabularies
    print("Building vocabularies...")
    word_vocab, label_vocab, char_vocab = build_vocabs(
        train_sents, train_labels, max_vocab_size=config["data"]["max_vocab_size"]
    )
    print(f"Word vocab: {len(word_vocab)}, Char vocab: {len(char_vocab)}, Labels: {len(label_vocab)}")

    # Load GloVe
    print("Loading GloVe embeddings...")
    glove = load_glove_embeddings(
        config["data"]["glove_path"], word_vocab, config["data"]["glove_dim"]
    )

    # Create datasets
    datasets = create_datasets(data, word_vocab, label_vocab, char_vocab)
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        datasets["validation"],
        batch_size=config["training"]["batch_size"],
        collate_fn=collate_fn,
    )

    # Build model
    model = BiLSTMCRF(
        vocab_size=len(word_vocab),
        num_chars=len(char_vocab),
        num_tags=len(label_vocab),
        word_emb_dim=config["data"]["glove_dim"],
        char_emb_dim=config["model"]["char_embedding_dim"],
        char_filters=config["model"]["char_conv_filters"],
        char_kernel=config["model"]["char_conv_kernel"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_lstm_layers"],
        dropout=config["model"]["dropout"],
        pretrained_word_emb=glove,
    )

    # Train
    vocabs = {
        "word_vocab": word_vocab,
        "label_vocab": label_vocab,
        "char_vocab": char_vocab,
    }
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_vocab=label_vocab,
        learning_rate=config["training"]["learning_rate"],
        gradient_clip=config["training"]["gradient_clip"],
        max_epochs=config["training"]["max_epochs"],
        patience=config["training"]["early_stopping_patience"],
        checkpoint_dir=config["training"]["checkpoint_dir"],
        tensorboard_dir=config["logging"]["tensorboard_dir"],
        vocabs=vocabs,
    )

    print("Starting training...")
    best_metrics = trainer.train()
    print(f"\nBest validation F1: {best_metrics['f1']:.4f}")
    print(best_metrics["report"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write integration test**

```python
# tests/test_trainer.py
import pytest
import torch
from torch.utils.data import DataLoader
from src.data.vocabulary import Vocabulary, LabelVocabulary
from src.data.dataset import NERDataset, collate_fn
from src.models.bilstm_crf import BiLSTMCRF
from src.training.trainer import Trainer


class TestTrainer:
    def test_training_loop_on_tiny_data(self, tmp_path):
        """Training loop runs without errors on a 5-sentence dataset."""
        sentences = [
            ["John", "lives", "in", "London", "."],
            ["Apple", "bought", "a", "startup", "."],
            ["Mary", "works", "at", "Google", "."],
            ["Berlin", "is", "in", "Germany", "."],
            ["The", "Olympics", "started", "today", "."],
        ]
        labels = [
            ["B-PER", "O", "O", "B-LOC", "O"],
            ["B-ORG", "O", "O", "O", "O"],
            ["B-PER", "O", "O", "B-ORG", "O"],
            ["B-LOC", "O", "O", "B-LOC", "O"],
            ["O", "B-MISC", "O", "O", "O"],
        ]

        all_tokens = [t for s in sentences for t in s]
        all_labels = [l for ls in labels for l in ls]
        word_vocab = Vocabulary.build(all_tokens)
        label_vocab = LabelVocabulary.build(all_labels)

        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)

        model = BiLSTMCRF(
            vocab_size=len(word_vocab),
            num_chars=len(dataset.char_vocab),
            num_tags=len(label_vocab),
            word_emb_dim=50,
            hidden_size=32,
            dropout=0.0,
        )

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,  # same data for test simplicity
            label_vocab=label_vocab,
            max_epochs=3,
            patience=10,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            tensorboard_dir=str(tmp_path / "runs"),
        )

        metrics = trainer.train()
        assert "f1" in metrics
        assert metrics["f1"] >= 0.0
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_trainer.py -v`
Expected: PASS (training loop completes in seconds on tiny data).

- [ ] **Step 5: Commit**

```bash
git add src/training/metrics.py src/training/trainer.py scripts/train.py tests/test_trainer.py
git commit -m "feat: training pipeline with early stopping, checkpointing, and TensorBoard"
```

---

## Chunk 4: Inference, API & Demo

### Task 10: Predictor

**Files:**
- Create: `src/inference/predictor.py`
- Create: `tests/test_predictor.py`

**Kavram — Inference Pipeline**
Eğitilmiş modeli yükleyip yeni metinler üzerinde tahmin yapma. Sıra:
1. Raw text → SpaCy ile tokenize et
2. Token'ları vocabulary ile indekslere çevir
3. Modelden geçir → tag dizisi al
4. Tag'leri entity'lere grupla (B-PER I-PER → tek bir entity)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predictor.py
import pytest
import torch
import os
from src.inference.predictor import NERPredictor, group_entities
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary
from src.models.bilstm_crf import BiLSTMCRF


class TestGroupEntities:
    def test_basic_grouping(self):
        tokens = ["John", "Smith", "works", "at", "Google"]
        tags = ["B-PER", "I-PER", "O", "O", "B-ORG"]
        entities = group_entities(tokens, tags)
        assert len(entities) == 2
        assert entities[0] == {"text": "John Smith", "label": "PER", "start": 0, "end": 2}
        assert entities[1] == {"text": "Google", "label": "ORG", "start": 4, "end": 5}

    def test_no_entities(self):
        tokens = ["the", "cat", "sat"]
        tags = ["O", "O", "O"]
        entities = group_entities(tokens, tags)
        assert entities == []

    def test_consecutive_different_entities(self):
        tokens = ["John", "Google"]
        tags = ["B-PER", "B-ORG"]
        entities = group_entities(tokens, tags)
        assert len(entities) == 2


class TestNERPredictor:
    @pytest.fixture
    def predictor(self, tmp_path):
        word_vocab = Vocabulary()
        for w in ["john", "works", "at", "google"]:
            word_vocab.add(w)
        label_vocab = LabelVocabulary.build(
            ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        )
        char_vocab = CharVocabulary.build(["john", "works", "at", "google"])

        model = BiLSTMCRF(
            vocab_size=len(word_vocab),
            num_chars=len(char_vocab),
            num_tags=len(label_vocab),
            word_emb_dim=50,
            hidden_size=32,
        )

        # Save checkpoint
        checkpoint_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        return NERPredictor(model, word_vocab, label_vocab, char_vocab)

    def test_predict_returns_entities(self, predictor):
        result = predictor.predict("John works at Google")
        assert isinstance(result, list)
        for entity in result:
            assert "text" in entity
            assert "label" in entity
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Predictor**

```python
# src/inference/predictor.py
import torch
import spacy
from typing import List
from src.models.bilstm_crf import BiLSTMCRF
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary


def group_entities(tokens: List[str], tags: List[str]) -> List[dict]:
    """Group BIO tags into entity spans.

    Example:
        tokens: ["John", "Smith", "works", "at", "Google"]
        tags:   ["B-PER", "I-PER", "O",    "O",  "B-ORG"]
        result: [
            {"text": "John Smith", "label": "PER", "start": 0, "end": 2},
            {"text": "Google", "label": "ORG", "start": 4, "end": 5},
        ]
    """
    entities = []
    current_entity = None

    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            label = tag[2:]
            current_entity = {"text": token, "label": label, "start": i, "end": i + 1}
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["label"]:
            current_entity["text"] += " " + token
            current_entity["end"] = i + 1
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None

    if current_entity:
        entities.append(current_entity)

    return entities


class NERPredictor:
    """End-to-end NER prediction: raw text → entities."""

    def __init__(
        self,
        model: BiLSTMCRF,
        word_vocab: Vocabulary,
        label_vocab: LabelVocabulary,
        char_vocab: CharVocabulary,
        device: str | None = None,
    ):
        self.model = model
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])

    def predict(self, text: str) -> List[dict]:
        """Predict entities in raw text."""
        doc = self.nlp(text)
        tokens = [token.text for token in doc]

        if not tokens:
            return []

        # Convert to indices
        word_ids = torch.tensor(
            [[self.word_vocab.word_to_idx(t) for t in tokens]], dtype=torch.long
        )

        max_word_len = max(len(t) for t in tokens)
        char_id_lists = []
        for token in tokens:
            char_ids = [self.char_vocab.char_to_idx(c) for c in token]
            char_ids += [0] * (max_word_len - len(char_ids))
            char_id_lists.append(char_ids)
        char_ids = torch.tensor([char_id_lists], dtype=torch.long)

        lengths = torch.tensor([len(tokens)], dtype=torch.long)

        with torch.no_grad():
            word_ids = word_ids.to(self.device)
            char_ids = char_ids.to(self.device)
            lengths = lengths.to(self.device)
            tag_indices = self.model.predict(word_ids, char_ids, lengths)[0]

        tags = [self.label_vocab.idx_to_label(idx) for idx in tag_indices]
        return group_entities(tokens, tags)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_predictor.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/inference/predictor.py tests/test_predictor.py
git commit -m "feat: NER predictor with SpaCy tokenization and entity grouping"
```

---

### Task 11: FastAPI Service

**Files:**
- Create: `api/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient


class TestAPI:
    def test_health_endpoint(self):
        from api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_predict_empty_text(self):
        from api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_too_long(self):
        from api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"text": "a" * 10001})
        assert response.status_code == 422
```

- [ ] **Step 2: Implement FastAPI app**

```python
# api/main.py
import os
import torch
import yaml
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="NER Pipeline API", version="1.0.0")

# Global predictor — loaded once at startup
predictor = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class PredictResponse(BaseModel):
    entities: list[Entity]
    tokens: list[str]


@app.on_event("startup")
def load_model():
    """Load model and vocabularies at startup."""
    global predictor

    config_path = os.environ.get("NER_CONFIG", "configs/conll2003.yaml")
    checkpoint_path = os.environ.get("NER_CHECKPOINT", "checkpoints/conll2003/best_model.pt")

    if not os.path.exists(checkpoint_path):
        print(f"WARNING: No checkpoint found at {checkpoint_path}. API will return empty results.")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load vocabs and model (saved during training)
    import pickle
    vocab_path = os.path.join(os.path.dirname(checkpoint_path), "vocabs.pkl")
    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)

    word_vocab = vocabs["word_vocab"]
    label_vocab = vocabs["label_vocab"]
    char_vocab = vocabs["char_vocab"]

    from src.models.bilstm_crf import BiLSTMCRF
    model = BiLSTMCRF(
        vocab_size=len(word_vocab),
        num_chars=len(char_vocab),
        num_tags=len(label_vocab),
        word_emb_dim=config["data"]["glove_dim"],
        char_emb_dim=config["model"]["char_embedding_dim"],
        char_filters=config["model"]["char_conv_filters"],
        char_kernel=config["model"]["char_conv_kernel"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_lstm_layers"],
        dropout=0.0,  # no dropout at inference
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    from src.inference.predictor import NERPredictor
    predictor = NERPredictor(model, word_vocab, label_vocab, char_vocab)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        return PredictResponse(entities=[], tokens=[])

    entities = predictor.predict(request.text)
    doc = predictor.nlp(request.text)
    tokens = [t.text for t in doc]

    return PredictResponse(
        entities=[Entity(**e) for e in entities],
        tokens=tokens,
    )
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_api.py -v`
Expected: Health test PASS. Validation tests PASS.

- [ ] **Step 4: Commit**

```bash
git add api/main.py tests/test_api.py
git commit -m "feat: FastAPI service with /predict and /health endpoints"
```

---

### Task 12: Gradio Demo

**Files:**
- Create: `demo/app.py`

- [ ] **Step 1: Implement Gradio demo**

```python
# demo/app.py
"""Interactive NER demo with Gradio. Run: python demo/app.py"""
import os
import sys
import torch
import yaml
import pickle
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bilstm_crf import BiLSTMCRF
from src.inference.predictor import NERPredictor

# Color map for entity types
ENTITY_COLORS = {
    "PER": "#FF6B6B",
    "ORG": "#4ECDC4",
    "LOC": "#45B7D1",
    "MISC": "#96CEB4",
    "TECH": "#FFEAA7",
    "MONEY": "#DDA0DD",
    "ROLE": "#98D8C8",
    "DATE": "#F7DC6F",
}


def load_predictor():
    config_path = os.environ.get("NER_CONFIG", "configs/conll2003.yaml")
    checkpoint_path = os.environ.get("NER_CHECKPOINT", "checkpoints/conll2003/best_model.pt")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    vocab_path = os.path.join(os.path.dirname(checkpoint_path), "vocabs.pkl")
    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)

    model = BiLSTMCRF(
        vocab_size=len(vocabs["word_vocab"]),
        num_chars=len(vocabs["char_vocab"]),
        num_tags=len(vocabs["label_vocab"]),
        word_emb_dim=config["data"]["glove_dim"],
        char_emb_dim=config["model"]["char_embedding_dim"],
        char_filters=config["model"]["char_conv_filters"],
        char_kernel=config["model"]["char_conv_kernel"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_lstm_layers"],
        dropout=0.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    return NERPredictor(model, vocabs["word_vocab"], vocabs["label_vocab"], vocabs["char_vocab"])


def predict_and_highlight(text: str):
    """Predict entities and format for Gradio HighlightedText."""
    entities = predictor.predict(text)
    return [(e["text"], e["label"]) for e in entities]


# Load model
predictor = load_predictor()

# Build Gradio interface
examples = [
    "Elon Musk announced that Tesla will open a new factory in Berlin next year.",
    "The United Nations held a meeting in New York about climate change.",
    "Google and Microsoft are competing in the artificial intelligence market.",
    "President Biden visited London to meet with Prime Minister Starmer.",
]

demo = gr.Interface(
    fn=predict_and_highlight,
    inputs=gr.Textbox(label="Enter text", lines=3, placeholder="Type or paste text here..."),
    outputs=gr.HighlightedText(label="Named Entities", color_map=ENTITY_COLORS),
    title="NER Pipeline — Named Entity Recognition",
    description="BiLSTM-CRF model trained on CoNLL-2003. Recognizes Person, Organization, Location, and Miscellaneous entities.",
    examples=examples,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
```

- [ ] **Step 2: Test manually after training**

Run: `python demo/app.py`
Expected: Browser opens with Gradio interface.

- [ ] **Step 3: Commit**

```bash
git add demo/app.py
git commit -m "feat: Gradio web demo with color-coded entity highlighting"
```

---

## Chunk 5: End-to-End Test

### Task 13: End-to-end training run

This is the moment of truth. Train on real data and see results.

- [ ] **Step 1: Verify data and GloVe are ready**

Run: `python -c "from src.data.preprocessing import load_conll2003; d = load_conll2003(); print(len(d['train'][0]))"`
Run: `ls -la data/glove/glove.6B.300d.txt`
Expected: 14041 sentences, GloVe file exists.

- [ ] **Step 2: Train the model**

Run: `python scripts/train.py --config configs/conll2003.yaml`
Expected: Training starts, loss decreases over epochs, F1 increases. Target: F1 ≥ 85% on validation.

- [ ] **Step 3: Run the full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Test the API**

Run: `uvicorn api.main:app --port 8000 &`
Run: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"text": "Elon Musk works at Tesla in California"}'`
Expected: JSON response with PER, ORG, LOC entities.

- [ ] **Step 5: Test the demo**

Run: `python demo/app.py`
Expected: Gradio interface opens, entities highlighted correctly.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "feat: complete NER pipeline — model trained and serving"
```

---

## Summary

| Chunk | Tasks | What It Builds |
|-------|-------|----------------|
| 1 | 1-4 | Project setup, Vocabulary, Dataset, Preprocessing |
| 2 | 5-7 | CharCNN, CRF, BiLSTM-CRF model |
| 3 | 8-9 | Metrics, Trainer, Training script (with vocab saving) |
| 4 | 10-12 | Predictor, FastAPI, Gradio demo |
| 5 | 13 | End-to-end training and verification |
