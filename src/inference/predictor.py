import torch
import spacy
from typing import List
from src.models.bilstm_crf import BiLSTMCRF
from src.data.vocabulary import Vocabulary, LabelVocabulary, CharVocabulary


def group_entities(tokens: List[str], tags: List[str]) -> List[dict]:
    entities = []
    current_entity = None
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            label = tag[2:]
            current_entity = {"text": token, "label": label, "start": i, "end": i + 1}
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["label"]:
            current_entity["text"] += " " + token
            current_entity["end"] = i + 1
        else:
            if current_entity:
                entities.append(current_entity)
                current_entity = None
    if current_entity:
        entities.append(current_entity)
    return entities


class NERPredictor:
    def __init__(self, model: BiLSTMCRF, word_vocab: Vocabulary,
                 label_vocab: LabelVocabulary, char_vocab: CharVocabulary,
                 device: str | None = None):
        self.model = model
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])

    def predict(self, text: str) -> List[dict]:
        doc = self.nlp(text)
        tokens = [token.text for token in doc]
        if not tokens:
            return []
        word_ids = torch.tensor(
            [[self.word_vocab.word_to_idx(t) for t in tokens]], dtype=torch.long
        )
        max_word_len = max(len(t) for t in tokens)
        char_id_lists = []
        for token in tokens:
            char_ids = [self.char_vocab.char_to_idx(c) for c in token]
            char_ids += [0] * (max_word_len - len(char_ids))
            char_id_lists.append(char_ids)
        char_ids = torch.tensor([char_id_lists], dtype=torch.long)
        lengths = torch.tensor([len(tokens)], dtype=torch.long)
        with torch.no_grad():
            word_ids = word_ids.to(self.device)
            char_ids = char_ids.to(self.device)
            lengths = lengths.to(self.device)
            tag_indices = self.model.predict(word_ids, char_ids, lengths)[0]
        tags = [self.label_vocab.idx_to_label(idx) for idx in tag_indices]
        return group_entities(tokens, tags)
