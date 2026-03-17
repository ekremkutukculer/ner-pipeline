"""Interactive NER demo with Gradio. Run: python demo/app.py"""
import os
import sys
import torch
import yaml
import pickle
import gradio as gr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.bilstm_crf import BiLSTMCRF
from src.inference.predictor import NERPredictor

ENTITY_COLORS = {
    "PER": "#FF6B6B",
    "ORG": "#4ECDC4",
    "LOC": "#45B7D1",
    "MISC": "#96CEB4",
    "TECH": "#FFEAA7",
    "MONEY": "#DDA0DD",
    "ROLE": "#98D8C8",
    "DATE": "#F7DC6F",
}


def load_predictor():
    config_path = os.environ.get("NER_CONFIG", "configs/conll2003.yaml")
    checkpoint_path = os.environ.get("NER_CHECKPOINT", "checkpoints/conll2003/best_model.pt")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    vocab_path = os.path.join(os.path.dirname(checkpoint_path), "vocabs.pkl")
    with open(vocab_path, "rb") as f:
        vocabs = pickle.load(f)
    model = BiLSTMCRF(
        vocab_size=len(vocabs["word_vocab"]), num_chars=len(vocabs["char_vocab"]),
        num_tags=len(vocabs["label_vocab"]), word_emb_dim=config["data"]["glove_dim"],
        char_emb_dim=config["model"]["char_embedding_dim"],
        char_filters=config["model"]["char_conv_filters"],
        char_kernel=config["model"]["char_conv_kernel"],
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_lstm_layers"], dropout=0.0,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return NERPredictor(model, vocabs["word_vocab"], vocabs["label_vocab"], vocabs["char_vocab"])


def predict_and_highlight(text: str):
    entities = predictor.predict(text)
    return [(e["text"], e["label"]) for e in entities]


predictor = load_predictor()

examples = [
    "Elon Musk announced that Tesla will open a new factory in Berlin next year.",
    "The United Nations held a meeting in New York about climate change.",
    "Google and Microsoft are competing in the artificial intelligence market.",
    "President Biden visited London to meet with Prime Minister Starmer.",
]

demo = gr.Interface(
    fn=predict_and_highlight,
    inputs=gr.Textbox(label="Enter text", lines=3, placeholder="Type or paste text here..."),
    outputs=gr.HighlightedText(label="Named Entities", color_map=ENTITY_COLORS),
    title="NER Pipeline — Named Entity Recognition",
    description="BiLSTM-CRF model trained on CoNLL-2003. Recognizes Person, Organization, Location, and Miscellaneous entities.",
    examples=examples,
)

if __name__ == "__main__":
    demo.launch()
