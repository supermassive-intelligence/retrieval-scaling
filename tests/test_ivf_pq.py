"""IVFPQ inherits IVF-Flat's build fixes, is named by its code size, and its exact re-scoring
returns the exact top k once every vector is a candidate."""
import glob
import os
import pickle
import tempfile
from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from src.indicies.base import index_file_name
from src.indicies.ivf_pq import IVFPQIndexer


def _shards(d, sizes, seed=0):
    rng = np.random.default_rng(seed)
    shards, paths = [rng.standard_normal((n, 8)).astype(np.float16) for n in sizes], []
    for k, e in enumerate(shards):
        paths.append(os.path.join(d, f"passages_{k:02d}.pkl"))
        with open(paths[-1], "wb") as f:
            pickle.dump((list(range(len(e))), e), f)
    return shards, paths


def _build(d, paths, **kw):
    kw = dict(sample_train_size=80, dimension=8, ncentroids=4, probe=4,
              n_subquantizers=2, code_size=4) | kw
    return IVFPQIndexer(embed_paths=paths, index_path=os.path.join(d, "i.faiss"),
                        meta_file=os.path.join(d, "i.meta"), trained_index_path=os.path.join(d, "i.faiss.trained"),
                        **kw)


def test_chunked_add_matches_whole_shard_add():
    with tempfile.TemporaryDirectory() as d:
        shards, paths = _shards(d, (50, 30))
        ix = _build(d, paths, num_keys_to_add_at_a_time=7)
        assert isinstance(ix.index, faiss.IndexIVFPQ)
        ref = faiss.read_index(os.path.join(d, "i.faiss.trained"))
        ref.add(np.concatenate(shards).astype(np.float32))
        with open(os.path.join(d, "i.faiss"), "rb") as f:
            assert f.read() == faiss.serialize_index(ref).tobytes()
        assert ix.index_id_to_db_id == [[0, i] for i in range(50)] + [[1, i] for i in range(30)]


def test_rebuild_is_identical():
    built = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            _, paths = _shards(d, (200,), seed=1)
            _build(d, paths, sample_train_size=60)
            with open(os.path.join(d, "i.faiss"), "rb") as f:
                built.append(f.read())
    assert built[0] == built[1]


def test_rescoring_every_vector_gives_the_exact_top_k():
    with tempfile.TemporaryDirectory() as d:
        shards, paths = _shards(d, (50, 30))
        vec = os.path.join(d, "v.npy")
        ix = _build(d, paths, rerank_k=80, rescore_vectors_path=vec)
        all_embs = np.concatenate(shards)
        assert np.array_equal(np.load(vec), all_embs)
        ix.get_retrieved_passages = lambda idx: (idx.tolist(), idx.tolist())
        q = np.random.default_rng(2).standard_normal((5, 8)).astype(np.float32)
        scores, ids, _ = ix.search(q, 6)
        exact = q @ all_embs.astype(np.float32).T
        top = np.argsort(-exact, axis=1)[:, :6]
        assert np.array_equal(np.array(ids), top)
        assert np.allclose(scores, np.take_along_axis(exact, top, axis=1), atol=1e-5)
        # PQ scores alone are approximate: without re-scoring they differ from the exact ones.
        pq_scores, _ = ix.index.search(q, 6)
        assert not np.allclose(pq_scores, scores, atol=1e-3)


def test_rescore_vectors_are_built_once():
    with tempfile.TemporaryDirectory() as d:
        _, paths = _shards(d, (50, 30))
        vec = os.path.join(d, "v.npy")
        _build(d, paths, rerank_k=10, rescore_vectors_path=vec)
        first = os.stat(vec).st_mtime_ns
        _build(d, paths, rerank_k=10, rescore_vectors_path=vec)  # loads the saved index
        assert os.stat(vec).st_mtime_ns == first
        assert not glob.glob(vec + ".*.tmp")


def test_too_few_candidates_is_refused():
    with tempfile.TemporaryDirectory() as d:
        _, paths = _shards(d, (50, 30))
        ix = _build(d, paths, probe=1, rerank_k=80, rescore_vectors_path=os.path.join(d, "v.npy"))
        q = np.random.default_rng(3).standard_normal((3, 8)).astype(np.float32)
        with pytest.raises(ValueError, match="3 of 3 queries got fewer than 80 candidates at nprobe=1"):
            ix.search(q, 5)


def test_pq_index_named_by_code_size():
    a = SimpleNamespace(sample_train_size=1000000, projection_size=768, ncentroids=8192, n_subquantizers=16, n_bits=8)
    b = SimpleNamespace(**{**vars(a), "n_subquantizers": 64})
    assert index_file_name("IVFPQ", a) == "index_IVFPQ.1000000.768.8192.pq16x8.faiss"
    assert index_file_name("IVFPQ", a) != index_file_name("IVFPQ", b)
    # The IVF-Flat indexes already built keep their names.
    assert index_file_name("IVFFlat", a) == "index_IVFFlat.1000000.768.8192.faiss"
    assert index_file_name("Flat", a) == "index_Flat.faiss"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("ok")
