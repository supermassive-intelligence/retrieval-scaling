"""GPU search (FlatIndexer._gpu_search) against faiss's CPU search on a real 1.4T shard index.

    python tests/check_gpu_search.py <index_Flat.faiss> [n_queries]

Queries are stored passage vectors plus noise, so each has a realistic neighbourhood. Reports the
share of identical top-100 lists, the mean overlap, and the largest score difference.
"""
import sys, time
import faiss, numpy as np, torch
from src.indicies.flat import FlatIndexer

path, nq = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2000
t = time.time(); index = faiss.read_index(path); print(f"loaded {index.ntotal:,} x {index.d} in {time.time()-t:.0f}s", flush=True)
xb = faiss.rev_swig_ptr(index.get_xb(), index.ntotal * index.d).reshape(index.ntotal, index.d)
rng = np.random.default_rng(0)
q = xb[rng.choice(index.ntotal, nq, replace=False)] + rng.normal(0, 0.05, (nq, index.d)).astype(np.float32)
fi = FlatIndexer.__new__(FlatIndexer); fi.index = index
torch.backends.cuda.matmul.allow_tf32 = False
t = time.time(); gs, gi = fi._gpu_search(q, 100); tg = time.time() - t
t = time.time(); cs, ci = index.search(q, 100); tc = time.time() - t
same = np.mean([np.array_equal(a, b) for a, b in zip(gi, ci)])
overlap = np.mean([len(set(a) & set(b)) / 100 for a, b in zip(gi, ci)])
print(f"gpu {tg:.1f}s  cpu {tc:.1f}s  identical lists {same:.4f}  mean overlap {overlap:.5f}  max |score diff| {np.abs(gs-cs).max():.2e}")
assert overlap > 0.999 and np.abs(gs - cs).max() < 1e-4, "GPU search disagrees with faiss"
print("ok")
