import pytest
import torch
from src.models.bilstm_crf import BiLSTMCRF

@pytest.fixture
def model():
    return BiLSTMCRF(
        vocab_size=100, num_chars=50, num_tags=9,
        word_emb_dim=300, char_emb_dim=50, char_filters=50,
        hidden_size=256, dropout=0.5,
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
        model = BiLSTMCRF(vocab_size=100, num_chars=50, num_tags=9, pretrained_word_emb=pretrained)
        assert torch.equal(model.word_embedding.weight.data, pretrained)
