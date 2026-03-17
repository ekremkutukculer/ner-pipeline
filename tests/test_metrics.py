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
        preds = [[label_vocab.label_to_idx(l) for l in ["B-PER", "O", "B-ORG"]]]
        targets = [[label_vocab.label_to_idx(l) for l in ["B-PER", "O", "O"]]]
        metrics = compute_metrics(preds, targets, label_vocab, [2])
        assert metrics["f1"] == 1.0
