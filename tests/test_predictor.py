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
            vocab_size=len(word_vocab), num_chars=len(char_vocab),
            num_tags=len(label_vocab), word_emb_dim=50, hidden_size=32,
        )
        checkpoint_path = str(tmp_path / "model.pt")
        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        return NERPredictor(model, word_vocab, label_vocab, char_vocab)

    def test_predict_returns_entities(self, predictor):
        result = predictor.predict("John works at Google")
        assert isinstance(result, list)
        for entity in result:
            assert "text" in entity
            assert "label" in entity
