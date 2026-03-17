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
        assert len(word_ids) == 4

    def test_char_ids_shape(self, sample_data, vocabs):
        sentences, labels = sample_data
        word_vocab, label_vocab = vocabs
        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        word_ids, char_ids, label_ids = dataset[0]
        assert char_ids.dim() == 2
        assert char_ids.shape[0] == 4

    def test_collate_fn_pads(self, sample_data, vocabs):
        sentences, labels = sample_data
        word_vocab, label_vocab = vocabs
        dataset = NERDataset(sentences, labels, word_vocab, label_vocab)
        batch = [dataset[0], dataset[1]]
        words, chars, labs, lengths = collate_fn(batch)
        assert words.shape[0] == 2
        assert words.shape[1] == 4
        assert lengths.tolist() == [4, 3]
