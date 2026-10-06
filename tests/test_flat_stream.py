"""The streamed flat index file is byte-identical to faiss.write_index of the same rows."""
import os
import pickle
import tempfile

import faiss
import numpy as np

from src.indicies.flat import FlatIndexer


def test_streamed_index_matches_faiss():
    rng = np.random.default_rng(0)
    shards = [rng.standard_normal((n, 8)).astype(np.float16) for n in (5, 3)]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, e in enumerate(shards):
            paths.append(os.path.join(d, f"passages_{k:02d}.pkl"))
            with open(paths[-1], "wb") as f:
                pickle.dump((list(range(len(e))), e), f)
        ix = FlatIndexer(embed_paths=paths, index_path=os.path.join(d, "i.faiss"),
                         meta_file=os.path.join(d, "i.meta"), dimension=8)
        ref = faiss.IndexFlatIP(8)
        ref.add(np.concatenate(shards).astype(np.float32))
        with open(os.path.join(d, "i.faiss"), "rb") as f:
            assert f.read() == faiss.serialize_index(ref).tobytes()
        assert ix.index_id_to_db_id == [[0, i] for i in range(5)] + [[1, i] for i in range(3)]
        q = rng.standard_normal((2, 8)).astype(np.float32)
        assert np.array_equal(ix.index.search(q, 4)[1], ref.search(q, 4)[1])


if __name__ == "__main__":
    test_streamed_index_matches_faiss()
    print("ok")
