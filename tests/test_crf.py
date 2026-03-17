import pytest
import torch
from src.models.crf import CRF

class TestCRF:
    def test_decode_returns_valid_shape(self):
        num_tags = 5
        crf = CRF(num_tags)
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
        assert loss.dim() == 0

    def test_loss_decreases_with_training(self):
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
