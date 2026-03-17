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
        assert len(vocab) == 4
        assert vocab.idx_to_label(vocab.label_to_idx("B-PER")) == "B-PER"

    def test_unknown_label_raises(self):
        vocab = LabelVocabulary.build(["O", "B-PER"])
        with pytest.raises(KeyError):
            vocab.label_to_idx("B-LOC")

    def test_sorted_order(self):
        labels = ["O", "B-PER", "B-LOC"]
        vocab = LabelVocabulary.build(labels)
        assert vocab.label_to_idx("B-LOC") == 0


class TestCharVocabulary:
    def test_build_from_tokens(self):
        vocab = CharVocabulary.build(["hello", "world"])
        assert len(vocab) > 1
        assert vocab.char_to_idx("h") != 0

    def test_unknown_char_returns_zero(self):
        vocab = CharVocabulary.build(["abc"])
        assert vocab.char_to_idx("z") == 0
