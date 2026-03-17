import pytest
import torch
from torch.utils.data import DataLoader
from src.data.vocabulary import Vocabulary, LabelVocabulary
from src.data.dataset import NERDataset, collate_fn
from src.models.bilstm_crf import BiLSTMCRF
from src.training.trainer import Trainer

class TestTrainer:
    def test_training_loop_on_tiny_data(self, tmp_path):
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
            vocab_size=len(word_vocab), num_chars=len(dataset.char_vocab),
            num_tags=len(label_vocab), word_emb_dim=50, hidden_size=32, dropout=0.0,
        )
        trainer = Trainer(
            model=model, train_loader=loader, val_loader=loader, label_vocab=label_vocab,
            max_epochs=3, patience=10, checkpoint_dir=str(tmp_path / "checkpoints"),
            tensorboard_dir=str(tmp_path / "runs"),
        )
        metrics = trainer.train()
        assert "f1" in metrics
        assert metrics["f1"] >= 0.0
