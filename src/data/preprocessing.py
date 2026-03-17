import numpy as np
import torch
from datasets import load_dataset
from typing import Tuple
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary
from src.data.dataset import NERDataset


CONLL_LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]


def load_conll2003() -> dict:
    """Load CoNLL-2003 from HuggingFace and convert to token/label lists."""
    dataset = load_dataset("lhoestq/conll2003")
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
    """Load GloVe vectors for words in our vocabulary."""
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
