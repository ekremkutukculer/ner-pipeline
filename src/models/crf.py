import torch
import torch.nn as nn
from typing import List


class CRF(nn.Module):
    """Linear-chain Conditional Random Field."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        log_numerator = self._compute_score(emissions, tags, lengths)
        log_denominator = self._compute_log_partition(emissions, lengths)
        return (log_denominator - log_numerator).mean()

    def _compute_score(self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, seq_len):
            mask = (t < lengths).float()
            emit_score = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            trans_score = self.transitions[tags[:, t - 1], tags[:, t]]
            score += (emit_score + trans_score) * mask
        last_idx = (lengths - 1).long()
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)
        score += self.end_transitions[last_tags]
        return score

    def _compute_log_partition(self, emissions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_tags = emissions.shape
        alpha = self.start_transitions + emissions[:, 0]
        for t in range(1, seq_len):
            mask = (t < lengths).float().unsqueeze(1)
            emit = emissions[:, t].unsqueeze(1)
            trans = self.transitions.unsqueeze(0)
            scores = alpha.unsqueeze(2) + emit + trans
            new_alpha = torch.logsumexp(scores, dim=1)
            alpha = new_alpha * mask + alpha * (1 - mask)
        alpha += self.end_transitions
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions: torch.Tensor, lengths: torch.Tensor) -> List[List[int]]:
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = self.start_transitions + emissions[:, 0]
        backpointers = []
        for t in range(1, seq_len):
            scores = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_scores, best_prev = scores.max(dim=1)
            viterbi = max_scores + emissions[:, t]
            backpointers.append(best_prev)
        viterbi += self.end_transitions
        best_paths = []
        for b in range(batch_size):
            length = lengths[b].item()
            best_last = viterbi[b].argmax().item()
            path = [best_last]
            for t in range(length - 2, -1, -1):
                best_last = backpointers[t][b][best_last].item()
                path.append(best_last)
            path.reverse()
            best_paths.append(path[:length])
        return best_paths
