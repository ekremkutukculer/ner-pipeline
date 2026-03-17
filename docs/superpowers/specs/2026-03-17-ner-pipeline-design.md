# NER Pipeline — Design Specification

## Purpose

Build a Named Entity Recognition pipeline from scratch using PyTorch's BiLSTM-CRF architecture. The project serves two goals: deep learning of NLP fundamentals, and a portfolio piece for Upwork AI/NLP freelance work.

## Target Job Profile

- PyTorch + Python deep learning model development
- Text-only data, 5-10 data scenarios
- End-to-end: data processing, training, testing, deployment
- Expert level, long-term engagement

## Approach

**Phase 1:** Train a BiLSTM-CRF model on CoNLL-2003 (PER, ORG, LOC, MISC).
**Phase 2:** Extend with a custom dataset (job postings) adding TECH, MONEY, ROLE, DATE entities.

Start narrow, prove the architecture works, then expand — demonstrating both depth and extensibility.

### Data Acquisition

**Phase 1 — CoNLL-2003:** The original dataset requires a Reuters RCV1 license. We will use the widely available preprocessed version from HuggingFace Datasets (`conll2003`), which provides train/validation/test splits ready to use.

**Phase 2 — Custom Job Postings:** Scrape 500-1000 job postings from public sources. Annotate manually using Label Studio (open-source annotation tool). Target: 500 annotated sentences minimum. Double-annotate 50 sentences to measure consistency. Train/val/test split: 70/15/15. This is a separate effort after Phase 1 is complete.

### Phase 2 Architecture

Phase 2 trains a separate model on the custom dataset with its own entity types (TECH, MONEY, ROLE, DATE). It does not merge with Phase 1's CoNLL model. The same BiLSTM-CRF architecture and training pipeline are reused — only the config and data change.

## Architecture

```
Raw Text → Preprocessing → BiLSTM-CRF Model → Entities (JSON)
                                ↑
                          Word Embeddings
                          (GloVe + Char CNN)
```

### Components

**1. Data Layer**
- Read CoNLL-format data (token-per-line, blank-line sentence boundaries)
- Build vocabulary from training data; map tokens and labels to integer indices
- Character-level vocabulary for the Char CNN
- PyTorch DataLoader with padding and batching

**2. Embedding Layer**
- Pretrained GloVe vectors (300d default; 100d as lightweight option) for word-level representation
- Character CNN: embed each character, run 1D convolution, max-pool to a fixed-size vector per word
- Concatenate word + char embeddings before feeding into BiLSTM
- Char CNN lets the model handle unseen words by recognizing sub-word patterns

**3. BiLSTM Layer**
- Bidirectional LSTM processes each token with left and right context
- Output: hidden state per token capturing contextual meaning
- Example: "Apple" after "bought" → likely ORG; after "ate" → not an entity

**4. CRF Layer**
- Conditional Random Field on top of BiLSTM emissions
- Learns valid tag transitions (B-PER → I-PER: yes; B-PER → I-ORG: no)
- Viterbi decoding finds the globally optimal tag sequence
- Without CRF, the model picks each tag independently — CRF enforces consistency

**5. Training Pipeline**
- YAML config files for hyperparameters
- Default hyperparameters: optimizer=Adam, lr=1e-3, hidden_size=256, char_emb=50, dropout=0.5, batch_size=64, max_epochs=50, gradient_clip=5.0
- Early stopping on validation F1 (patience=5)
- Model checkpointing (save best model)
- Metrics: precision, recall, F1 per entity type + overall (seqeval format)
- TensorBoard logging for loss curves and F1 tracking

**6. Inference API (FastAPI)**
- `POST /predict` — accepts raw text, returns entities with labels and positions
- Tokenization at inference: SpaCy tokenizer (handles punctuation, contractions correctly)
- Model loaded once at startup
- Input validation: max 10,000 characters, non-empty text, UTF-8
- Error handling with proper HTTP status codes

**7. Web Demo (Gradio)**
- Text input box, submit button
- Entities displayed with color-coded highlights
- Example texts pre-loaded for quick demonstration

## Directory Structure

```
ner-pipeline/
├── configs/              # Training and model configs (YAML)
├── data/
│   ├── raw/              # Original dataset files
│   └── processed/        # Preprocessed, split data
├── src/
│   ├── data/             # Dataset class, vocabulary, preprocessing
│   ├── models/           # BiLSTM-CRF model definition
│   ├── training/         # Trainer, metrics, early stopping
│   └── inference/        # Prediction logic
├── api/                  # FastAPI service
├── demo/                 # Gradio web demo
├── tests/                # Unit and integration tests
├── notebooks/            # Exploration and analysis notebooks
├── scripts/              # Train, evaluate, download scripts
├── requirements.txt
└── README.md
```

## Entity Types

### Phase 1 (CoNLL-2003)
| Tag  | Meaning        | Example          |
|------|----------------|------------------|
| PER  | Person         | Elon Musk        |
| ORG  | Organization   | Google, NATO     |
| LOC  | Location       | Istanbul, Germany|
| MISC | Miscellaneous  | Olympics, English |

### Phase 2 (Custom — Job Postings)
| Tag   | Meaning    | Example              |
|-------|------------|----------------------|
| TECH  | Technology | Python, PyTorch      |
| MONEY | Salary     | $50/hr, $120K        |
| ROLE  | Position   | Data Scientist       |
| DATE  | Date       | March 2026           |

## Tagging Scheme

BIO format:
- **B-XXX** — Beginning of entity XXX
- **I-XXX** — Inside (continuation) of entity XXX
- **O** — Outside any entity

Example: "Elon Musk works at Tesla" → `B-PER I-PER O O B-ORG`

## Extensibility

Adding a new **sequence labeling** task (POS tagging, chunking, custom NER domain) requires:
1. New config in `configs/` with entity types and hyperparameters
2. New dataset in CoNLL format under `data/`
3. New API endpoint in `api/`

The model architecture, training pipeline, and serving infrastructure remain unchanged. Tasks with different structures (classification, generation) would require new model classes and data layers.

## Tech Stack

- **Python 3.10+**
- **PyTorch** — model definition and training
- **GloVe** — pretrained word embeddings
- **seqeval** — NER metrics (entity-level F1)
- **FastAPI** — REST API
- **Gradio** — interactive web demo
- **PyYAML** — config management
- **pytest** — testing
- **SpaCy** — tokenization at inference time
- **TensorBoard** — training metrics visualization
- **HuggingFace Datasets** — CoNLL-2003 data loading

## Tokenization Strategy

- **Training (CoNLL-2003):** Data comes pre-tokenized, one token per line. No additional tokenization needed.
- **Inference (raw text):** SpaCy English tokenizer (`en_core_web_sm`) splits raw text into tokens.
- **Known limitation:** CoNLL-2003 and SpaCy tokenize differently in some edge cases (hyphenated words, abbreviations). Tokens that SpaCy splits differently may map to `<UNK>`. The Char CNN mitigates this by extracting character-level features from unseen words.
- **Mitigation plan:** After training, measure F1 on the test set using both gold tokenization and SpaCy re-tokenization. If the gap exceeds 2 F1 points, add a normalization step to align SpaCy output with training vocabulary.

## Testing Strategy

- **Unit tests:** Vocabulary building, dataset loading, BIO tag validation, CRF forward/decode
- **Integration tests:** Full training loop on a tiny dataset (5 sentences), API endpoint request/response
- **Smoke tests:** Model loads, predicts on sample text, Gradio demo renders

## Hardware Requirements

BiLSTM-CRF on CoNLL-2003 trains on CPU in ~30-60 minutes per epoch. GPU (any CUDA-capable) reduces this to ~2-5 minutes per epoch. The project detects and uses GPU when available, falls back to CPU.

## Success Criteria

1. BiLSTM-CRF model trains and converges on CoNLL-2003
2. Entity-level F1 score ≥ 85% on CoNLL-2003 test set (SOTA baseline for BiLSTM-CRF: ~91%)
3. FastAPI endpoint returns correct predictions for arbitrary text
4. Gradio demo runs and displays entities visually
5. Code is clean, documented, and structured for extension
