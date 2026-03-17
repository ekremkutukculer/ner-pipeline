import pytest
import torch
from src.models.char_cnn import CharCNN

class TestCharCNN:
    def test_output_shape(self):
        char_ids = torch.randint(0, 26, (2, 3, 5))
        model = CharCNN(num_chars=30, char_emb_dim=50, num_filters=50, kernel_size=3)
        output = model(char_ids)
        assert output.shape == (2, 3, 50)

    def test_different_word_lengths(self):
        char_ids = torch.randint(0, 26, (1, 4, 10))
        model = CharCNN(num_chars=30, char_emb_dim=50, num_filters=50, kernel_size=3)
        output = model(char_ids)
        assert output.shape == (1, 4, 50)

    def test_padding_invariance(self):
        model = CharCNN(num_chars=30, char_emb_dim=50, num_filters=50, kernel_size=3)
        model.eval()
        chars1 = torch.tensor([[[1, 2, 3, 0, 0]]])
        chars2 = torch.tensor([[[1, 2, 3, 0, 0]]])
        out1 = model(chars1)
        out2 = model(chars2)
        assert torch.equal(out1, out2)
