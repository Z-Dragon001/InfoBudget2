"""Fine-tuned BERT plus checkpoint MLP and TextTiling topic segmentation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from infobudget.config import SegmentationConfig
from infobudget.segmentation.text_tiling import PairScorer, TextTilingSegmenter


class BertMLPTextTilingSegmenter(TextTilingSegmenter):
    """Restore fine-tuned BERT and coherence_decoder weights from a checkpoint."""

    method_name = "bert_mlp_text_tiling"
    boundary_reason = "bert_mlp_texttiling_depth"

    def __init__(
        self,
        cfg: SegmentationConfig,
        model_dir: str | Path,
        checkpoint_path: str | Path,
        pair_scorer: PairScorer | None = None,
    ):
        super().__init__(cfg, pair_scorer)
        self.model_dir = Path(model_dir).resolve()
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self._tokenizer = None
        self._bert = None
        self._decoder = None
        self._device = None

    def _score_adjacent_pairs(self, utterances: list[str]) -> list[float]:
        self._load_model()
        import torch

        scores: list[float] = []
        for start in range(0, len(utterances) - 1, self.cfg.bert_batch_size):
            end = min(start + self.cfg.bert_batch_size, len(utterances) - 1)
            inputs = self._tokenizer(
                utterances[start:end],
                utterances[start + 1 : end + 1],
                padding=True,
                truncation=True,
                max_length=self.cfg.bert_max_length,
                return_tensors="pt",
            )
            inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}
            with torch.inference_mode():
                outputs = self._bert(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                logits = self._decoder(cls_embeddings)
                probabilities = torch.softmax(logits, dim=-1)[:, 0]
            scores.extend(float(value) for value in probabilities.cpu().tolist())
        return scores

    def _load_model(self) -> None:
        if self._bert is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Local BERT model not found: {self.model_dir}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"BERT+MLP checkpoint not found: {self.checkpoint_path}. "
                "Place the trained checkpoint at segmentation.bert_mlp_checkpoint."
            )

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self._bert = AutoModel.from_pretrained(self.model_dir, local_files_only=True)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(state_dict, dict):
            raise TypeError("BERT+MLP checkpoint must contain a state_dict mapping")
        state_dict = {_normalize_key(str(key)): value for key, value in state_dict.items()}

        bert_state = {
            key.removeprefix("bert."): value
            for key, value in state_dict.items()
            if key.startswith("bert.")
        }
        if not bert_state:
            raise ValueError("Checkpoint does not contain bert.* parameters")
        self._bert.load_state_dict(bert_state, strict=True)
        self._decoder = _build_decoder(state_dict, self.cfg.bert_mlp_activation)
        self._bert.to(self._device).eval()
        self._decoder.to(self._device).eval()


def _normalize_key(key: str) -> str:
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            key = key[len(prefix) :]
    return key


def _natural_key(value: str) -> list[Any]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def _build_decoder(state_dict: dict[str, Any], activation_name: str):
    """Infer decoder linear dimensions from coherence_decoder.* checkpoint tensors."""
    import torch
    from torch import nn

    weight_items = sorted(
        (
            (key, value)
            for key, value in state_dict.items()
            if key.startswith("coherence_decoder.")
            and key.endswith(".weight")
            and getattr(value, "ndim", 0) == 2
        ),
        key=lambda item: _natural_key(item[0]),
    )
    if not weight_items:
        raise ValueError("Checkpoint does not contain coherence_decoder linear weights")

    supported_keys = {
        key
        for weight_key, _weight in weight_items
        for key in (weight_key, f"{weight_key[: -len('.weight')]}.bias")
    }
    unsupported_keys = sorted(
        key
        for key in state_dict
        if key.startswith("coherence_decoder.") and key not in supported_keys
    )
    if unsupported_keys:
        raise ValueError(
            "Checkpoint coherence_decoder contains unsupported learned layers: "
            + ", ".join(unsupported_keys)
        )

    layers = nn.ModuleList()
    with torch.no_grad():
        for weight_key, weight in weight_items:
            prefix = weight_key[: -len(".weight")]
            bias = state_dict.get(f"{prefix}.bias")
            layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]), bias=bias is not None)
            layer.weight.copy_(weight)
            if bias is not None:
                layer.bias.copy_(bias)
            layers.append(layer)

    activations = {
        "relu": torch.relu,
        "gelu": torch.nn.functional.gelu,
        "tanh": torch.tanh,
    }
    try:
        activation = activations[activation_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported bert_mlp_activation: {activation_name}") from exc

    class CheckpointMLP(nn.Module):
        def __init__(self, linear_layers, hidden_activation):
            super().__init__()
            self.layers = linear_layers
            self.hidden_activation = hidden_activation

        def forward(self, inputs):
            outputs = inputs
            for index, layer in enumerate(self.layers):
                outputs = layer(outputs)
                if index < len(self.layers) - 1:
                    outputs = self.hidden_activation(outputs)
            return outputs

    return CheckpointMLP(layers, activation)
