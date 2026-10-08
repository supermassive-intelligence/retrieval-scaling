"""The batched passage read returns exactly what the per-hit read returns."""
import json
import os
import tempfile

from src.indicies.flat import FlatIndexer


def test_batched_read_matches_per_hit_read():
    d = tempfile.mkdtemp()
    pos_map, db_ids = {}, []
    for shard in range(2):
        path = os.path.join(d, f"raw_passages-{shard}.jsonl")
        pos_map[shard] = []
        with open(path, "w") as f:
            for chunk in range(50):
                pos_map[shard].append((path, f.tell()))
                f.write(json.dumps({"shard_id": shard, "id": chunk, "text": f"p{shard}-{chunk} é"}) + "\n")
                db_ids.append((shard, chunk))
    ix = FlatIndexer.__new__(FlatIndexer)
    ix.psg_pos_id_map, ix.index_id_to_db_id, ix.default_shard_id = pos_map, db_ids, 0
    hits = [[99, 3, 3, 51], [0, 77, 12, 99]]
    passages, ids, _ = ix.get_retrieved_passages(hits)
    assert passages == [[ix._get_passage(i)["text"] for i in q] for q in hits]
    assert ids == [[db_ids[i] for i in q] for q in hits]


if __name__ == "__main__":
    test_batched_read_matches_per_hit_read()
    print("ok")
