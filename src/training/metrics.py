from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from typing import List
from src.data.vocabulary import LabelVocabulary


def compute_metrics(
    predictions: List[List[int]],
    targets: List[List[int]],
    label_vocab: LabelVocabulary,
    lengths: List[int],
) -> dict:
    pred_labels = []
    true_labels = []
    for pred_seq, true_seq, length in zip(predictions, targets, lengths):
        pred_labels.append([label_vocab.idx_to_label(p) for p in pred_seq[:length]])
        true_labels.append([label_vocab.idx_to_label(t) for t in true_seq[:length]])
    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
        "report": classification_report(true_labels, pred_labels),
    }
