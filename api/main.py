import os
import torch
import yaml
import pickle
from fastapi import FastAPI
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    config_path = os.environ.get("NER_CONFIG", "configs/conll2003.yaml")
    checkpoint_path = os.environ.get("NER_CHECKPOINT", "checkpoints/conll2003/best_model.pt")
    if os.path.exists(checkpoint_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)
        vocab_path = os.path.join(os.path.dirname(checkpoint_path), "vocabs.pkl")
        with open(vocab_path, "rb") as f:
            vocabs = pickle.load(f)
        from src.models.bilstm_crf import BiLSTMCRF
        model = BiLSTMCRF(
            vocab_size=len(vocabs["word_vocab"]),
            num_chars=len(vocabs["char_vocab"]),
            num_tags=len(vocabs["label_vocab"]),
            word_emb_dim=config["data"]["glove_dim"],
            char_emb_dim=config["model"]["char_embedding_dim"],
            char_filters=config["model"]["char_conv_filters"],
            char_kernel=config["model"]["char_conv_kernel"],
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_lstm_layers"],
            dropout=0.0,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        from src.inference.predictor import NERPredictor
        predictor = NERPredictor(
            model,
            vocabs["word_vocab"],
            vocabs["label_vocab"],
            vocabs["char_vocab"],
        )
    else:
        print(f"WARNING: No checkpoint at {checkpoint_path}. API returns empty results.")
    yield


app = FastAPI(title="NER Pipeline API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class PredictResponse(BaseModel):
    entities: list[Entity]
    tokens: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if predictor is None:
        return PredictResponse(entities=[], tokens=[])
    entities = predictor.predict(request.text)
    doc = predictor.nlp(request.text)
    tokens = [t.text for t in doc]
    return PredictResponse(
        entities=[Entity(**e) for e in entities],
        tokens=tokens,
    )
