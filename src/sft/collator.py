"""Batch collation for hybrid explainer SFT."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


class HybridSFTCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must have a pad_token_id before creating the collator.")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        batch_feature_vectors = []
        batch_act_positions = []
        batch_inject_mask = []

        for feature in features:
            pad_len = max_length - len(feature["input_ids"])
            batch_input_ids.append(
                feature["input_ids"] + [self.tokenizer.pad_token_id] * pad_len
            )
            batch_attention_mask.append([1] * len(feature["input_ids"]) + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
            batch_feature_vectors.append(feature["feature_vector"])
            batch_act_positions.append(feature["act_position"])
            batch_inject_mask.append(bool(feature["inject_mask"]))

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "feature_vectors": torch.from_numpy(np.stack(batch_feature_vectors)).to(dtype=torch.float32),
            "act_positions": torch.tensor(batch_act_positions, dtype=torch.long),
            "inject_mask": torch.tensor(batch_inject_mask, dtype=torch.bool),
        }
