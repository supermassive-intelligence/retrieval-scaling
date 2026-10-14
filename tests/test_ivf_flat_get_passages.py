"""The IVF-Flat batched passage read returns exactly what the per-hit read returns."""
import json
import os
import tempfile

import pytest

from src.indicies.ivf_flat import IVFFlatIndexer


def _indexer(d):
    pos_map, db_ids = {}, []
    for shard in range(2):
        path = os.path.join(d, f"raw_passages-{shard}.jsonl")
        pos_map[shard] = []
        with open(path, "w") as f:
            for chunk in range(50):
                pos_map[shard].append((path, f.tell()))
                f.write(json.dumps({"shard_id": shard, "id": chunk, "text": f"p{shard}-{chunk} é"}) + "\n")
                db_ids.append([shard, chunk])
    ix = IVFFlatIndexer.__new__(IVFFlatIndexer)
    ix.psg_pos_id_map, ix.index_id_to_db_id, ix.probe = pos_map, db_ids, 4
    return ix


def test_batched_read_matches_per_hit_read():
    ix = _indexer(tempfile.mkdtemp())
    hits = [[99, 3, 3, 51], [0, 77, 12, 99]]
    passages, ids = ix.get_retrieved_passages(hits)
    assert passages == [[ix._get_passage(i)["text"] for i in q] for q in hits]
    assert ids == [[ix.index_id_to_db_id[i] for i in q] for q in hits]


def test_padded_row_is_refused():
    ix = _indexer(tempfile.mkdtemp())
    with pytest.raises(ValueError, match="1 of 2 queries got fewer than 4 hits at nprobe=4"):
        ix.get_retrieved_passages([[0, 1, 2, 3], [5, 6, -1, -1]])


if __name__ == "__main__":
    test_batched_read_matches_per_hit_read()
    test_padded_row_is_refused()
    print("ok")
