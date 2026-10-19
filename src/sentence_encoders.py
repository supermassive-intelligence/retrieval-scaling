"""Encoders loaded through sentence-transformers: which names, at what length, with what query prompt.

embed.py and search.py each picked the loader by substring, in four places that had to agree;
BAAI/bge-* matched none of them and was refused. The two choices below change the
embeddings without any error, so they are made here, once, and logged.
"""
import logging

from sentence_transformers import SentenceTransformer

# Substrings of model names loaded with SentenceTransformer. "bge" covers BAAI/bge-small-en-v1.5;
# MiniLM is reached as sentence-transformers/all-MiniLM-L6-v2.
SENTENCE_TRANSFORMER_KEYS = ("sentence-transformers", "e5", "Qwen3", "bge")


def uses_sentence_transformers(model_name_or_path):
    return any(k in model_name_or_path for k in SENTENCE_TRANSFORMER_KEYS)


def load_sentence_transformer(model_name_or_path, max_length):
    """Cap the model at max_length tokens, as the contriever path does with its tokenizer.

    Without the cap each model used its own: 256 for all-MiniLM-L6-v2 and 512 for
    bge-small-en-v1.5, against the 256 (passage_maxlength = chunk_size) contriever was given,
    so a comparison of encoders would also have been a comparison of how much of each
    passage they read. A model is never run past its own limit.
    """
    model = SentenceTransformer(model_name_or_path)
    own = model.max_seq_length
    model.max_seq_length = min(own, max_length) if own else max_length
    logging.info(f"{model_name_or_path}: max_seq_length {model.max_seq_length} (model's own {own}, asked {max_length})")
    return model


def query_prompt(args, model_name_or_path):
    """The text put before every question, or None for none.

    BGE's model card asks for "Represent this sentence for searching relevant passages: "
    before short retrieval queries and the passages take none, so leaving it out is a choice
    that changes the score; for BGE it must be set (to "" for none) rather than defaulted.
    """
    prompt = args.get("query_prompt", None)
    if prompt is None and "bge" in model_name_or_path:
        raise ValueError(
            f"{model_name_or_path} needs evaluation.search.query_prompt set: the model card's "
            f"'Represent this sentence for searching relevant passages: ', or '' for none")
    logging.info(f"Query prompt for {model_name_or_path}: {prompt!r}")
    return prompt or None
