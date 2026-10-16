import os
import json
import random
import logging
import pickle
import time
import glob
from tqdm import tqdm
import pdb
from typing import List, Tuple, Any
from abc import ABC, abstractmethod
from omegaconf import ListConfig
import subprocess
import re
import uuid

import faiss
import numpy as np
import torch
from transformers import GPTNeoXTokenizerFast

import contriever.src.contriever
import contriever.src.utils
import contriever.src.slurm
from contriever.src.evaluation import calculate_matches
import contriever.src.normalize_text

from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids
from src.indicies.ivf_flat import IVFFlatIndexer


os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


# Local patch (muisti): IVFPQIndexer was a copy of IVFFlatIndexer that differed only in the
# faiss index it trains, so it lacked the fixes made to IVF-Flat since (seeded training
# sample, chunked add, batched passage read, refusing rows faiss padded with -1). It now
# inherits them and overrides only the training step and the search.
class IVFPQIndexer(IVFFlatIndexer):

    def __init__(self, *args, n_subquantizers=16, code_size=8, rerank_k=0, rescore_vectors_path=None, **kwargs):
        # Set before the base __init__, which trains and builds the index.
        self.n_subquantizers = n_subquantizers
        self.code_size = code_size
        self.rerank_k = rerank_k
        super().__init__(*args, **kwargs)
        # Local patch (muisti): optional exact re-scoring. PQ scores are approximate, so with
        # rerank_k > 0 the index returns its top rerank_k by PQ score, those are re-scored
        # with the stored fp16 vectors (the values the Flat index holds, cast to fp32), and
        # the top k by exact score are kept. The vectors are memory-mapped and only the
        # candidate rows are read, so the in-memory index stays the compressed one.
        self.vectors = None
        if self.rerank_k:
            assert rescore_vectors_path, "rerank_k needs rescore_vectors_path"
            self.vectors = self._load_or_build_vectors(rescore_vectors_path)

    def _train_index(self, sampled_embs, trained_index_path):
        quantizer = faiss.IndexFlatIP(self.dimension)
        start_index = faiss.IndexIVFPQ(quantizer,
                                       self.dimension,
                                       self.ncentroids,
                                       self.n_subquantizers,
                                       self.code_size,
                                       faiss.METRIC_INNER_PRODUCT
                                       )
        start_index.nprobe = self.probe
        np.random.seed(1)
        # Faiss does not handle adding keys in fp16 as of writing this.
        start_index.train(sampled_embs)
        faiss.write_index(start_index, trained_index_path)

    def _load_or_build_vectors(self, path):
        # One fp16 row per index id, in the order _add_keys added them (self.embed_paths
        # order), so row i is index id i. Written to a temporary name and renamed, so a
        # killed build never leaves a short file that a later run would trust. The name is
        # unique per writer: runs at other PQ settings share this file and may build it at
        # the same time; their files are identical and the last rename wins.
        n = len(self.index_id_to_db_id)
        if not os.path.exists(path):
            logging.info(f"Writing re-scoring vectors to {path}...")
            tmp = f"{path}.{uuid.uuid4().hex}.tmp"
            out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16, shape=(n, self.dimension))
            offset = 0
            for embed_path in self.embed_paths:
                with open(embed_path, "rb") as fin:
                    _, embs = pickle.load(fin)
                embs = np.asarray(embs)
                assert embs.ndim == 2 and embs.shape[1] == self.dimension, embs.shape
                out[offset:offset + len(embs)] = embs
                offset += len(embs)
                del embs
            assert offset == n, (offset, n)
            out.flush()
            del out
            os.replace(tmp, path)
        vectors = np.load(path, mmap_mode="r")
        assert vectors.shape == (n, self.dimension), (vectors.shape, n)
        return vectors

    def _rescore(self, query_embs, candidates, k, batch_size=64):
        scores = np.empty((len(candidates), k), dtype=np.float32)
        indices = np.empty((len(candidates), k), dtype=np.int64)
        for b in range(0, len(candidates), batch_size):
            cand = candidates[b:b + batch_size]
            # Read each candidate row once, in file order.
            rows, inverse = np.unique(cand, return_inverse=True)
            embs = np.asarray(self.vectors[rows], dtype=np.float32)[inverse.reshape(cand.shape)]
            exact = np.einsum("qd,qkd->qk", query_embs[b:b + batch_size], embs)
            order = np.argsort(-exact, axis=1, kind="stable")[:, :k]
            scores[b:b + batch_size] = np.take_along_axis(exact, order, axis=1)
            indices[b:b + batch_size] = np.take_along_axis(cand, order, axis=1)
        return scores, indices

    def search(self, query_embs, k=4096):
        if not self.rerank_k:
            return super().search(query_embs, k)
        assert self.rerank_k >= k, f"rerank_k={self.rerank_k} must be at least k={k}"
        query_embs = query_embs.astype(np.float32)
        _, candidates = self.index.search(query_embs, self.rerank_k)
        short = int((candidates.min(axis=1) < 0).sum())
        if short:
            raise ValueError(f"{short} of {len(candidates)} queries got fewer than "
                             f"{self.rerank_k} candidates at nprobe={self.probe}; raise the probe")
        all_scores, all_indices = self._rescore(query_embs, candidates, k)
        all_passages, db_ids = self.get_retrieved_passages(all_indices)
        return all_scores.tolist(), all_passages, db_ids


def test_build_multi_shard_dpr_wiki():
    embed_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/embeddings/facebook/contriever-msmarco/dpr_wiki/1-shards'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    n_subquantizers = 16
    code_size = 8
    formatted_index_name = f"index_ivf_pq_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/embeddings/facebook/contriever-msmarco/dpr_wiki/1-shards/index_ivf_pq/'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
    pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
    passage_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/passages/dpr_wiki/1-shards'
    index = IVFPQIndexer(
        embed_paths,
        index_path,
        meta_file,
        trained_index_path,
        passage_dir=passage_dir,
        pos_map_save_path=pos_map_save_path,
        sample_train_size=sample_train_size,
        dimension=projection_size,
        ncentroids=ncentroids,
        n_subquantizers=n_subquantizers,
        code_size=code_size,
        )


def test_build_single_shard_dpr_wiki():
    embed_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards'
    embed_paths = [os.path.join(embed_dir, filename) for filename in os.listdir(embed_dir) if filename.endswith('.pkl')]
    sample_train_size = 6000000
    projection_size = 768
    ncentroids = 4096
    n_subquantizers = 16
    code_size = 8
    formatted_index_name = f"index_ivf_pq_ip.{sample_train_size}.{projection_size}.{ncentroids}.faiss"
    index_dir = '/checkpoint/amaia/explore/comem/data/scaling_out/embeddings/facebook/dragon-plus-context-encoder/dpr_wiki/8-shards/index_ivf_pq/'
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, formatted_index_name)
    meta_file = os.path.join(index_dir, formatted_index_name+'.meta')
    trained_index_path = os.path.join(index_dir, formatted_index_name+'.trained')
    pos_map_save_path = os.path.join(index_dir, 'passage_pos_id_map.pkl')
    passage_dir = '/fsx-comem/rulin/data/truth_teller/scaling_out/passages/dpr_wiki/8-shards'
    index = IVFPQIndexer(
        embed_paths,
        index_path,
        meta_file,
        trained_index_path,
        passage_dir=passage_dir,
        pos_map_save_path=pos_map_save_path,
        sample_train_size=sample_train_size,
        dimension=projection_size,
        ncentroids=ncentroids,
        n_subquantizers=n_subquantizers,
        code_size=code_size,
        )

if __name__ == '__main__':
    test_build_multi_shard_dpr_wiki()