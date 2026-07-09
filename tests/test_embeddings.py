"""Tests for configurable text encoders."""

from __future__ import annotations

from infobudget.utils.embeddings import HashingTextEncoder, SentenceTransformerTextEncoder, build_text_encoder


def test_build_text_encoder_uses_sentence_transformer_for_minilm() -> None:
    encoder = build_text_encoder("sentence-transformers/all-MiniLM-L6-v2")

    assert isinstance(encoder, SentenceTransformerTextEncoder)
    assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert encoder.dim == 384


def test_sentence_transformer_encoder_falls_back_to_hashing_when_disabled() -> None:
    encoder = SentenceTransformerTextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        fallback_encoder=HashingTextEncoder(dim=16),
    )
    encoder._disabled_reason = "offline test"

    embeddings = encoder.encode_batch(["InfoBudget routes memory extraction."])

    assert embeddings.shape == (1, 16)


def test_sentence_transformer_default_fallback_keeps_configured_dimension() -> None:
    encoder = SentenceTransformerTextEncoder(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dim=384,
    )
    encoder._disabled_reason = "offline test"

    embeddings = encoder.encode_batch(["InfoBudget routes memory extraction."])

    assert embeddings.shape == (1, 384)
