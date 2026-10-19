"""BGE and MiniLM load through sentence-transformers, capped at the length contriever is given,
and BGE refuses to run without its query prompt being chosen."""
from omegaconf import OmegaConf

import src.sentence_encoders as se


def test_names():
    for name in ("BAAI/bge-small-en-v1.5", "sentence-transformers/all-MiniLM-L6-v2",
                 "intfloat/e5-base-v2", "Qwen/Qwen3-Embedding-0.6B"):
        assert se.uses_sentence_transformers(name), name
    for name in ("facebook/contriever-msmarco", "facebook/dragon-plus-query-encoder"):
        assert not se.uses_sentence_transformers(name), name


class FakeModel:
    def __init__(self, own):
        self.max_seq_length = own


def test_length_is_capped_never_raised():
    real = se.SentenceTransformer
    try:
        for own, asked, used in ((512, 256, 256), (256, 256, 256), (256, 512, 256)):
            se.SentenceTransformer = lambda name: FakeModel(own)
            assert se.load_sentence_transformer("BAAI/bge-small-en-v1.5", asked).max_seq_length == used
    finally:
        se.SentenceTransformer = real


def test_bge_needs_a_prompt_chosen():
    try:
        se.query_prompt(OmegaConf.create({}), "BAAI/bge-small-en-v1.5")
        raise AssertionError("BGE ran without query_prompt set")
    except ValueError as err:
        assert "query_prompt" in str(err)
    assert se.query_prompt(OmegaConf.create({"query_prompt": ""}), "BAAI/bge-small-en-v1.5") is None
    p = "Represent this sentence for searching relevant passages: "
    assert se.query_prompt(OmegaConf.create({"query_prompt": p}), "BAAI/bge-small-en-v1.5") == p
    assert se.query_prompt(OmegaConf.create({}), "sentence-transformers/all-MiniLM-L6-v2") is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
