"""Training entry point. Usage: python scripts/train.py --config configs/conll2003.yaml"""
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from src.data.preprocessing import load_conll2003, build_vocabs, load_glove_embeddings, create_datasets
from src.data.dataset import collate_fn
from src.models.bilstm_crf import BiLSTMCRF
from src.training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/conll2003.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    print("Loading CoNLL-2003...")
    data = load_conll2003()
    train_sents, train_labels = data["train"]
    print("Building vocabularies...")
    word_vocab, label_vocab, char_vocab = build_vocabs(train_sents, train_labels, max_vocab_size=config["data"]["max_vocab_size"])
    print(f"Word vocab: {len(word_vocab)}, Char vocab: {len(char_vocab)}, Labels: {len(label_vocab)}")
    print("Loading GloVe embeddings...")
    glove = load_glove_embeddings(config["data"]["glove_path"], word_vocab, config["data"]["glove_dim"])
    datasets = create_datasets(data, word_vocab, label_vocab, char_vocab)
    train_loader = DataLoader(datasets["train"], batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(datasets["validation"], batch_size=config["training"]["batch_size"], collate_fn=collate_fn)
    model = BiLSTMCRF(
        vocab_size=len(word_vocab), num_chars=len(char_vocab), num_tags=len(label_vocab),
        word_emb_dim=config["data"]["glove_dim"], char_emb_dim=config["model"]["char_embedding_dim"],
        char_filters=config["model"]["char_conv_filters"], char_kernel=config["model"]["char_conv_kernel"],
        hidden_size=config["model"]["hidden_size"], num_layers=config["model"]["num_lstm_layers"],
        dropout=config["model"]["dropout"], pretrained_word_emb=glove,
    )
    vocabs = {"word_vocab": word_vocab, "label_vocab": label_vocab, "char_vocab": char_vocab}
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader, label_vocab=label_vocab,
        learning_rate=config["training"]["learning_rate"], gradient_clip=config["training"]["gradient_clip"],
        max_epochs=config["training"]["max_epochs"], patience=config["training"]["early_stopping_patience"],
        checkpoint_dir=config["training"]["checkpoint_dir"], tensorboard_dir=config["logging"]["tensorboard_dir"],
        vocabs=vocabs,
    )
    print("Starting training...")
    best_metrics = trainer.train()
    print(f"\nBest validation F1: {best_metrics['f1']:.4f}")
    print(best_metrics["report"])

if __name__ == "__main__":
    main()
