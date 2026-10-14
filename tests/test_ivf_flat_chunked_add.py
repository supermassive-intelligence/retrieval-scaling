"""The chunked IVF-Flat build is byte-identical to adding the whole stacked shard at once,
and a rebuild trains on the same sample, so it gives the same index."""
import os
import pickle
import tempfile

import faiss
import numpy as np

from src.indicies.ivf_flat import IVFFlatIndexer


def test_chunked_add_matches_whole_shard_add():
    rng = np.random.default_rng(0)
    shards = [rng.standard_normal((n, 8)).astype(np.float16) for n in (50, 30)]
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for k, e in enumerate(shards):
            paths.append(os.path.join(d, f"passages_{k:02d}.pkl"))
            with open(paths[-1], "wb") as f:
                pickle.dump((list(range(len(e))), e), f)
        trained = os.path.join(d, "i.faiss.trained")
        # Chunks of 7 rows, so every shard is split and the last chunk is short.
        ix = IVFFlatIndexer(embed_paths=paths, index_path=os.path.join(d, "i.faiss"),
                            meta_file=os.path.join(d, "i.meta"), trained_index_path=trained,
                            sample_train_size=80, dimension=8, ncentroids=4, probe=2,
                            num_keys_to_add_at_a_time=7)
        ref = faiss.read_index(trained)
        ref.add(np.concatenate(shards).astype(np.float32))
        with open(os.path.join(d, "i.faiss"), "rb") as f:
            assert f.read() == faiss.serialize_index(ref).tobytes()
        assert ix.index_id_to_db_id == [[0, i] for i in range(50)] + [[1, i] for i in range(30)]
        ref.nprobe = 2
        q = rng.standard_normal((3, 8)).astype(np.float32)
        assert np.array_equal(ix.index.search(q, 5)[1], ref.search(q, 5)[1])


def test_rebuild_is_identical():
    rng = np.random.default_rng(1)
    e = rng.standard_normal((200, 8)).astype(np.float16)
    built = []
    for _ in range(2):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "passages_00.pkl")
            with open(path, "wb") as f:
                pickle.dump((list(range(len(e))), e), f)
            # A sample smaller than the shard, so which rows are drawn matters.
            IVFFlatIndexer(embed_paths=[path], index_path=os.path.join(d, "i.faiss"),
                           meta_file=os.path.join(d, "i.meta"),
                           trained_index_path=os.path.join(d, "i.faiss.trained"),
                           sample_train_size=60, dimension=8, ncentroids=4, probe=2)
            with open(os.path.join(d, "i.faiss"), "rb") as f:
                built.append(f.read())
    assert built[0] == built[1]


if __name__ == "__main__":
    test_chunked_add_matches_whole_shard_add()
    test_rebuild_is_identical()
    print("ok")
