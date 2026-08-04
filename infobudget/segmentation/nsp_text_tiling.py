"""BERT Next Sentence Prediction plus TextTiling topic segmentation."""

from __future__ import annotations

from pathlib import Path

from infobudget.config import SegmentationConfig
from infobudget.segmentation.text_tiling import PairScorer, TextTilingSegmenter


class NSPTextTilingSegmenter(TextTilingSegmenter):
    """Use the local pretrained BERT NSP head as the coherence scorer."""

    method_name = "nsp_text_tiling"
    boundary_reason = "nsp_texttiling_depth"

    def __init__(
        self,
        cfg: SegmentationConfig,
        model_dir: str | Path,
        pair_scorer: PairScorer | None = None,
    ):
        super().__init__(cfg, pair_scorer)
        self.model_dir = Path(model_dir).resolve()
        self._tokenizer = None
        self._model = None
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
                logits = self._model(**inputs).logits
                probabilities = torch.softmax(logits, dim=-1)[:, 0]
            scores.extend(float(value) for value in probabilities.cpu().tolist())
        return scores

    def _load_model(self) -> None:
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(
                f"Local BERT NSP model not found: {self.model_dir}. "
                "Download google-bert/bert-base-uncased into segmentation.bert_model_dir first."
            )
        import torch
        from transformers import AutoModelForNextSentencePrediction, AutoTokenizer

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir, local_files_only=True)
        self._model = AutoModelForNextSentencePrediction.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self._model.to(self._device)
        self._model.eval()
