"""Custom Trainer for vec-only DPO with injected feature vectors."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from transformers import TrainerCallback
from transformers.trainer_callback import ExportableState
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from src.dpo.loss import dpo_loss, sequence_log_probs_from_logits
from src.sft.trainer import HybridSFTTrainer


def _normalize_eval_metric_name(metric_name: str) -> str:
    return metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"


def _metric_is_better(
    value: float,
    best_value: float,
    *,
    greater_is_better: bool,
    min_delta: float = 0.0,
) -> bool:
    if greater_is_better:
        return value > best_value + min_delta
    return value < best_value - min_delta


def _metric_is_tied(value: float, best_value: float, *, tie_epsilon: float) -> bool:
    return abs(value - best_value) <= tie_epsilon


def _get_metric_value(metrics: dict[str, float], metric_name: str) -> float:
    metric_to_check = _normalize_eval_metric_name(metric_name)
    try:
        return float(metrics[metric_to_check])
    except KeyError as exc:
        raise KeyError(
            f"The checkpoint-selection metric '{metric_to_check}' was not found in the evaluation metrics. "
            f"The available evaluation metrics are: {list(metrics.keys())}."
        ) from exc


class TieBreakEarlyStoppingCallback(TrainerCallback, ExportableState):
    """Early stopping that matches DPO lexicographic best-checkpoint selection."""

    def __init__(
        self,
        *,
        early_stopping_patience: int,
        metric_for_best_model: str,
        greater_is_better: bool,
        secondary_metric_for_best_model: str | None = None,
        secondary_greater_is_better: bool | None = None,
        tie_epsilon: float = 0.0,
        early_stopping_threshold: float = 0.0,
    ) -> None:
        self.early_stopping_patience = early_stopping_patience
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self.secondary_metric_for_best_model = secondary_metric_for_best_model
        self.secondary_greater_is_better = (
            secondary_greater_is_better
            if secondary_greater_is_better is not None
            else greater_is_better
        )
        self.tie_epsilon = tie_epsilon
        self.early_stopping_threshold = early_stopping_threshold
        self.early_stopping_patience_counter = 0
        self._best_primary_metric: float | None = None
        self._best_secondary_metric: float | None = None

    def _is_new_best(self, metrics: dict[str, float]) -> bool:
        primary_value = _get_metric_value(metrics, self.metric_for_best_model)
        secondary_value = (
            _get_metric_value(metrics, self.secondary_metric_for_best_model)
            if self.secondary_metric_for_best_model is not None
            else None
        )

        if self._best_primary_metric is None:
            self._best_primary_metric = primary_value
            self._best_secondary_metric = secondary_value
            return True

        primary_is_better = _metric_is_better(
            primary_value,
            self._best_primary_metric,
            greater_is_better=self.greater_is_better,
            min_delta=self.early_stopping_threshold,
        )
        primary_is_tied = _metric_is_tied(
            primary_value,
            self._best_primary_metric,
            tie_epsilon=self.tie_epsilon,
        )
        secondary_is_better = (
            primary_is_tied
            and secondary_value is not None
            and (
                self._best_secondary_metric is None
                or _metric_is_better(
                    secondary_value,
                    self._best_secondary_metric,
                    greater_is_better=bool(self.secondary_greater_is_better),
                    min_delta=self.early_stopping_threshold,
                )
            )
        )

        if primary_is_better or secondary_is_better:
            self._best_primary_metric = primary_value
            self._best_secondary_metric = secondary_value
            return True
        return False

    def on_train_begin(self, args, state, control, **kwargs):
        if not args.load_best_model_at_end:
            raise ValueError("TieBreakEarlyStoppingCallback requires load_best_model_at_end=True.")
        if getattr(args, "eval_strategy", None) in (IntervalStrategy.NO, "no"):
            raise ValueError("TieBreakEarlyStoppingCallback requires evaluation during training.")

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if self._is_new_best(metrics):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1

        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True
        return control

    def state(self) -> dict[str, Any]:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "metric_for_best_model": self.metric_for_best_model,
                "greater_is_better": self.greater_is_better,
                "secondary_metric_for_best_model": self.secondary_metric_for_best_model,
                "secondary_greater_is_better": self.secondary_greater_is_better,
                "tie_epsilon": self.tie_epsilon,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
                "_best_primary_metric": self._best_primary_metric,
                "_best_secondary_metric": self._best_secondary_metric,
            },
        }


class HybridDPOTrainer(HybridSFTTrainer):
    def __init__(
        self,
        *args,
        reference_model: torch.nn.Module | None,
        reference_device: str | None = None,
        beta: float,
        average_log_prob: bool = False,
        secondary_metric_for_best_model: str | None = None,
        secondary_greater_is_better: bool | None = None,
        best_model_tie_epsilon: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        policy_device = next(self.model.parameters()).device
        self.reference_device = torch.device(reference_device) if reference_device else policy_device
        self.reference_model = reference_model.to(self.reference_device) if reference_model is not None else None
        if self.reference_device != policy_device and getattr(self.args, "_n_gpu", 1) > 1:
            # We are manually splitting policy/reference across two visible GPUs in a
            # single process. Prevent Trainer from wrapping the policy model in
            # nn.DataParallel, which would replicate it onto the reference GPU and
            # defeat the purpose of the split.
            self.args._n_gpu = 1
            self._train_batch_size = self.args.train_batch_size
        if self.reference_model is not None:
            self.reference_model.eval()
            for param in self.reference_model.parameters():
                param.requires_grad = False
        self.beta = beta
        self.average_log_prob = average_log_prob
        self.secondary_metric_for_best_model = secondary_metric_for_best_model
        self.secondary_greater_is_better = secondary_greater_is_better
        self.best_model_tie_epsilon = best_model_tie_epsilon
        self._best_secondary_metric: float | None = None
        self._metric_accumulators: dict[str, dict[str, list[float]]] = {
            "train": defaultdict(list),
            "eval": defaultdict(list),
        }

    @property
    def best_secondary_metric(self) -> float | None:
        return self._best_secondary_metric

    def _restore_best_secondary_metric_from_history(self, secondary_metric_name: str) -> None:
        if self._best_secondary_metric is not None:
            return

        best_step = getattr(self.state, "best_global_step", None)
        if not best_step:
            return

        for log_entry in reversed(getattr(self.state, "log_history", [])):
            if int(log_entry.get("step", -1)) != int(best_step):
                continue
            if secondary_metric_name not in log_entry:
                continue
            self._best_secondary_metric = float(log_entry[secondary_metric_name])
            return

    def _determine_best_metric(self, metrics, trial):
        if self.args.metric_for_best_model is None or self.secondary_metric_for_best_model is None:
            return super()._determine_best_metric(metrics, trial)

        primary_metric_name = _normalize_eval_metric_name(self.args.metric_for_best_model)
        secondary_metric_name = _normalize_eval_metric_name(self.secondary_metric_for_best_model)
        primary_value = _get_metric_value(metrics, primary_metric_name)
        secondary_value = _get_metric_value(metrics, secondary_metric_name)
        primary_greater_is_better = bool(self.args.greater_is_better)
        secondary_greater_is_better = (
            self.secondary_greater_is_better
            if self.secondary_greater_is_better is not None
            else primary_greater_is_better
        )

        if self.state.best_metric is None:
            self.state.best_metric = float("-inf") if primary_greater_is_better else float("inf")

        self._restore_best_secondary_metric_from_history(secondary_metric_name)
        best_primary_value = float(self.state.best_metric)
        primary_is_better = _metric_is_better(
            primary_value,
            best_primary_value,
            greater_is_better=primary_greater_is_better,
        )
        primary_is_tied = _metric_is_tied(
            primary_value,
            best_primary_value,
            tie_epsilon=self.best_model_tie_epsilon,
        )
        secondary_is_better = (
            primary_is_tied
            and (
                self._best_secondary_metric is None
                or _metric_is_better(
                    secondary_value,
                    self._best_secondary_metric,
                    greater_is_better=bool(secondary_greater_is_better),
                )
            )
        )

        is_new_best_metric = primary_is_better or secondary_is_better
        if is_new_best_metric:
            self.state.best_metric = primary_value
            self._best_secondary_metric = secondary_value
            self.state.best_secondary_metric = secondary_value

            if self.args.save_strategy in [SaveStrategy.STEPS, SaveStrategy.EPOCH]:
                self.state.best_global_step = self.state.global_step

        return is_new_best_metric

    def _move_batch_to_model_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        return self._move_batch_to_device(batch, device)

    def _move_batch_to_reference_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self._move_batch_to_device(batch, self.reference_device)

    def _move_batch_to_device(self, batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
        moved = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(device)
            else:
                moved[key] = value
        return moved

    def prediction_step(
        self,
        model,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        """Run DPO evaluation through compute_loss instead of model(**inputs).

        The collator produces chosen/rejected fields, which do not match
        HybridExplainerModel.forward(input_ids=..., ...). Training already goes
        through compute_loss(); evaluation must do the same.
        """
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.compute_loss_context_manager():
                num_items_in_batch = self._get_num_items_in_batch([inputs], self.args.device)
                loss, outputs = self.compute_loss(
                    model,
                    inputs,
                    return_outputs=True,
                    num_items_in_batch=num_items_in_batch,
                )
            loss = loss.detach().mean()

        if prediction_loss_only:
            return (loss, None, None)

        logits = None
        if isinstance(outputs, dict):
            logits = outputs.get("preference_logits")
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            logits = outputs[0]

        if isinstance(logits, torch.Tensor):
            logits = logits.detach()

        return (loss, logits, None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        policy_batch = self._move_batch_to_model_device(inputs)

        shared_kwargs = {
            "feature_vectors": policy_batch["feature_vectors"],
            "act_positions": policy_batch["act_positions"],
            "inject_mask": policy_batch["inject_mask"],
        }

        policy_chosen = model(
            input_ids=policy_batch["chosen_input_ids"],
            attention_mask=policy_batch["chosen_attention_mask"],
            **shared_kwargs,
        )
        policy_rejected = model(
            input_ids=policy_batch["rejected_input_ids"],
            attention_mask=policy_batch["rejected_attention_mask"],
            **shared_kwargs,
        )

        policy_chosen_logps = sequence_log_probs_from_logits(
            policy_chosen.logits,
            policy_batch["chosen_labels"],
            average_log_prob=self.average_log_prob,
        )
        policy_rejected_logps = sequence_log_probs_from_logits(
            policy_rejected.logits,
            policy_batch["rejected_labels"],
            average_log_prob=self.average_log_prob,
        )
        cached_ref_mask = policy_batch.get("has_cached_reference_logps")
        use_cached_reference_logps = cached_ref_mask is not None and bool(cached_ref_mask.all().item())

        if use_cached_reference_logps:
            ref_chosen_logps = policy_batch["cached_ref_chosen_logps"]
            ref_rejected_logps = policy_batch["cached_ref_rejected_logps"]
        else:
            if self.reference_model is None:
                raise ValueError(
                    "Reference log-probs were not fully cached for this batch, but no reference_model was provided."
                )

            ref_batch = self._move_batch_to_reference_device(inputs)
            ref_shared_kwargs = {
                "feature_vectors": ref_batch["feature_vectors"],
                "act_positions": ref_batch["act_positions"],
                "inject_mask": ref_batch["inject_mask"],
            }

            with torch.no_grad():
                ref_chosen = self.reference_model(
                    input_ids=ref_batch["chosen_input_ids"],
                    attention_mask=ref_batch["chosen_attention_mask"],
                    **ref_shared_kwargs,
                )
                ref_rejected = self.reference_model(
                    input_ids=ref_batch["rejected_input_ids"],
                    attention_mask=ref_batch["rejected_attention_mask"],
                    **ref_shared_kwargs,
                )

            ref_chosen_logps = sequence_log_probs_from_logits(
                ref_chosen.logits,
                ref_batch["chosen_labels"],
                average_log_prob=self.average_log_prob,
            ).to(policy_chosen_logps.device)
            ref_rejected_logps = sequence_log_probs_from_logits(
                ref_rejected.logits,
                ref_batch["rejected_labels"],
                average_log_prob=self.average_log_prob,
            ).to(policy_chosen_logps.device)

        loss, preference_logits = dpo_loss(
            policy_chosen_logps=policy_chosen_logps,
            policy_rejected_logps=policy_rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
            beta=self.beta,
        )

        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        reward_margin = chosen_rewards - rejected_rewards
        preference_accuracy = (reward_margin > 0).float().mean()

        phase = "train" if model.training else "eval"
        phase_metrics = self._metric_accumulators[phase]
        metrics = {
            "chosen_reward": chosen_rewards.mean().detach(),
            "rejected_reward": rejected_rewards.mean().detach(),
            "reward_margin": reward_margin.mean().detach(),
            "preference_accuracy": preference_accuracy.detach(),
            "policy_chosen_logps": policy_chosen_logps.mean().detach(),
            "policy_rejected_logps": policy_rejected_logps.mean().detach(),
            "ref_chosen_logps": ref_chosen_logps.mean().detach(),
            "ref_rejected_logps": ref_rejected_logps.mean().detach(),
        }
        for key, value in metrics.items():
            phase_metrics[key].append(float(value.item()))

        outputs = {
            "preference_logits": preference_logits,
            "policy_chosen_logps": policy_chosen_logps.detach(),
            "policy_rejected_logps": policy_rejected_logps.detach(),
        }
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        phase = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        accum = self._metric_accumulators[phase]
        if accum:
            for key, values in accum.items():
                if not values:
                    continue
                metric_name = f"eval_{key}" if phase == "eval" else key
                logs.setdefault(metric_name, sum(values) / len(values))
            self._metric_accumulators[phase] = defaultdict(list)
        super().log(logs, start_time=start_time)
