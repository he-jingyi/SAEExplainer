"""Batch collation for vec-only DPO training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.Tensor:
    max_length = max(len(seq) for seq in sequences)
    padded = [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]
    return torch.tensor(padded, dtype=torch.long)


class VecOnlyDPOCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id for DPO collation.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        chosen_input_ids = _pad_sequences([f["chosen_input_ids"] for f in features], self.tokenizer.pad_token_id)
        chosen_labels = _pad_sequences([f["chosen_labels"] for f in features], -100)
        chosen_attention_mask = (chosen_input_ids != self.tokenizer.pad_token_id).long()

        rejected_input_ids = _pad_sequences([f["rejected_input_ids"] for f in features], self.tokenizer.pad_token_id)
        rejected_labels = _pad_sequences([f["rejected_labels"] for f in features], -100)
        rejected_attention_mask = (rejected_input_ids != self.tokenizer.pad_token_id).long()

        return {
            "record_indices": [f["record_index"] for f in features],
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "rejected_labels": rejected_labels,
            "feature_vectors": torch.from_numpy(np.stack([f["feature_vector"] for f in features])).to(torch.float32),
            "act_positions": torch.tensor([f["act_position"] for f in features], dtype=torch.long),
            "inject_mask": torch.tensor([bool(f["inject_mask"]) for f in features], dtype=torch.bool),
            "feature_ids": [f["feature_id"] for f in features],
            "chosen_texts": [f["chosen_text"] for f in features],
            "rejected_texts": [f["rejected_text"] for f in features],
            "chosen_scores": torch.tensor([f["chosen_score"] for f in features], dtype=torch.float32),
            "rejected_scores": torch.tensor([f["rejected_score"] for f in features], dtype=torch.float32),
            "score_gaps": torch.tensor([f["score_gap"] for f in features], dtype=torch.float32),
            "has_cached_reference_logps": torch.tensor(
                [bool(f["has_cached_reference_logps"]) for f in features], dtype=torch.bool
            ),
            "cached_ref_chosen_logps": torch.tensor(
                [float(f["cached_ref_chosen_logps"]) for f in features], dtype=torch.float32
            ),
            "cached_ref_rejected_logps": torch.tensor(
                [float(f["cached_ref_rejected_logps"]) for f in features], dtype=torch.float32
            ),
        }
