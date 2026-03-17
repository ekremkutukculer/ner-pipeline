# NER Pipeline — Named Entity Recognition from Scratch

A production-ready Named Entity Recognition system built from scratch with PyTorch. Implements a BiLSTM-CRF architecture trained on CoNLL-2003, achieving **87.4% F1 score**.

No pretrained transformers — every layer (Char CNN, BiLSTM, CRF) is implemented and trained from the ground up.

## Architecture

```
Raw Text → SpaCy Tokenizer → Word Embeddings (GloVe 300d) + Char CNN → BiLSTM → CRF → Entities
```

| Layer | What it does |
|-------|-------------|
| **Char CNN** | Extracts sub-word features from character sequences (handles unseen words) |
| **GloVe** | Pretrained word embeddings (300d, 6B tokens) |
| **BiLSTM** | Reads context in both directions to understand each token |
| **CRF** | Enforces valid tag transitions (e.g., B-PER can't be followed by I-ORG) |

## Results

Trained on CoNLL-2003 (14,041 sentences), evaluated on the validation set:

| Entity | Precision | Recall | F1 |
|--------|-----------|--------|----|
| **PER** (Person) | 0.85 | 0.93 | 0.89 |
| **ORG** (Organization) | 0.89 | 0.80 | 0.84 |
| **LOC** (Location) | 0.95 | 0.88 | 0.92 |
| **MISC** (Miscellaneous) | 0.81 | 0.80 | 0.81 |
| **Overall** | **0.88** | **0.87** | **0.87** |

## Quick Start

### Installation

```bash
git clone https://github.com/ekremkutukculer/ner-pipeline.git
cd ner-pipeline
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Download GloVe & Train

```bash
python scripts/download_glove.py      # Downloads GloVe embeddings (~862MB)
python scripts/train.py               # Trains the model (~12 epochs, early stopping)
```

### Run the API

```bash
PYTHONPATH=. uvicorn api.main:app --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Elon Musk works at Tesla in California"}'
```

```json
{
  "entities": [
    {"text": "Elon Musk", "label": "PER", "start": 0, "end": 2},
    {"text": "Tesla", "label": "ORG", "start": 4, "end": 5},
    {"text": "California", "label": "LOC", "start": 6, "end": 7}
  ]
}
```

### Run the Web Demo

```bash
PYTHONPATH=. python demo/app.py
```

Opens an interactive Gradio interface where you can type text and see entities highlighted with colors.

## Project Structure

```
ner-pipeline/
├── src/
│   ├── data/           # Vocabulary, Dataset, Preprocessing (HuggingFace + GloVe)
│   ├── models/         # CharCNN, CRF (Viterbi + Forward), BiLSTM-CRF
│   ├── training/       # Trainer (early stopping, checkpointing, TensorBoard)
│   └── inference/      # Predictor (SpaCy tokenization, entity grouping)
├── api/                # FastAPI REST service (/predict, /health)
├── demo/               # Gradio interactive web demo
├── scripts/            # Training and data download scripts
├── tests/              # 36 unit + integration tests (pytest)
└── configs/            # YAML training configs
```

## Key Design Decisions

- **BiLSTM-CRF over Transformer**: Demonstrates deep understanding of sequence labeling fundamentals. Easy to extend to new tasks.
- **Char CNN**: Handles out-of-vocabulary words by learning character-level patterns.
- **Config-driven training**: Change hyperparameters without touching code.
- **Modular architecture**: Adding a new sequence labeling task requires only a new config and dataset.

## Tech Stack

Python 3.10+ · PyTorch · GloVe · seqeval · FastAPI · Gradio · SpaCy · TensorBoard · pytest

## Testing

```bash
pytest tests/ -v    # 36 tests covering vocabulary, dataset, models, metrics, trainer, API
```

## License

MIT
