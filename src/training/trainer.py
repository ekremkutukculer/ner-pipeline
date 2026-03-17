import os
import pickle
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
    def __init__(self, model: BiLSTMCRF, train_loader: DataLoader, val_loader: DataLoader,
                 label_vocab: LabelVocabulary, learning_rate: float = 1e-3,
                 gradient_clip: float = 5.0, max_epochs: int = 50, patience: int = 5,
                 checkpoint_dir: str = "checkpoints", tensorboard_dir: str = "runs",
                 device: Optional[str] = None, vocabs: dict | None = None):
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
        best_metrics = {}
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch()
            val_metrics = self._evaluate()
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            self.writer.add_scalar("Precision/val", val_metrics["precision"], epoch)
            self.writer.add_scalar("Recall/val", val_metrics["recall"], epoch)
            print(f"Epoch {epoch}/{self.max_epochs} — Loss: {train_loss:.4f} — Val F1: {val_metrics['f1']:.4f}")
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
            nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches

    def _evaluate(self) -> dict:
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
        return compute_metrics(all_predictions, all_targets, self.label_vocab, all_lengths)

    def _save_checkpoint(self, epoch: int):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_f1": self.best_f1,
        }, path)
        if self.vocabs:
            vocab_path = os.path.join(self.checkpoint_dir, "vocabs.pkl")
            with open(vocab_path, "wb") as f:
                pickle.dump(self.vocabs, f)
