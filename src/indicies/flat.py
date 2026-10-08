import os
import re
import json
import time
import pickle
import faiss
import struct
import numpy as np
import torch

from src.indicies.index_utils import convert_pkl_to_jsonl, get_passage_pos_ids


os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _flat_ip_header(d, ntotal):
    """faiss's on-disk header for an IndexFlatIP of ntotal rows, up to the float payload:
    fourcc, d, ntotal, two reserved 1<<20 fields, is_trained, metric (0 = inner product),
    then the code vector's length in floats."""
    return b"IxFI" + struct.pack("<iqqq?iQ", d, ntotal, 1 << 20, 1 << 20, True, 0, ntotal * d)

device = 'cuda' if torch.cuda.is_available()  else 'cpu'


class FlatIndexer(object):

    def __init__(self, 
                embed_paths=None,
                index_path=None,
                meta_file=None,
                passage_dir=None,
                pos_map_save_path=None,
                dimension=768,
                ):
    
        self.embed_paths = embed_paths
        self.index_path = index_path  # path to store the final index
        self.meta_file = meta_file  # path to save the index id to db id map
        self.passage_dir = passage_dir
        self.pos_map_save_path = pos_map_save_path
        self.dimension=dimension
        self.cuda = False
        # Local patch (muisti, MAS-396): the released per-shard metas are bare chunk ids,
        # and _get_passage used to assume shard 0 for them, reading shard k's hits out of
        # shard 0's file. A single-shard index lives in index_<type>/<k>/; take k from there.
        _leaf = os.path.basename(os.path.dirname(index_path))
        self.default_shard_id = int(_leaf) if _leaf.isdigit() else 0

        if os.path.exists(index_path) and os.path.exists(self.meta_file):
            print("Loading index...")
            self.index = faiss.read_index(index_path)
            self.index_id_to_db_id = self.load_index_id_to_db_id()
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.index_id_to_db_id = []
            print ("Building index...")
            self._build_index()
        
        if self.pos_map_save_path is not None:
            self.psg_pos_id_map = self.load_psg_pos_id_map()

    def _build_index(self,):
        # Local patch (muisti, MAS-396): stream the fp32 rows of every shard straight into
        # an IndexFlatIP file, then read it back. Holding the fp16 shard and a full fp32
        # index at once (~3x the shard) OOM-killed 26-33 GB rpj_c4/commoncrawl shards at
        # 110 GiB; this peaks at the fp32 index alone. The file is byte-identical to
        # faiss.write_index of the same rows (checked by tests/test_flat_stream.py).
        start_time = time.time()
        ntotal = 0
        with open(self.index_path + '.partial', 'wb') as out:
            out.write(_flat_ip_header(self.dimension, 0))  # rewritten once ntotal is known
            for embed_path in self.embed_paths:
                filename = os.path.basename(embed_path)
                match = re.search(r"passages_(\d+)\.pkl", filename)
                shard_id = int(match.group(1))
                with open(embed_path, "rb") as fin:
                    _, embs = pickle.load(fin)
                del _
                embs = np.asarray(embs)
                assert embs.ndim == 2 and embs.shape[1] == self.dimension, embs.shape
                for i in range(0, len(embs), 1_000_000):
                    out.write(np.ascontiguousarray(embs[i:i + 1_000_000], dtype=np.float32).tobytes())
                self.index_id_to_db_id.extend([shard_id, chunk_id] for chunk_id in range(len(embs)))
                ntotal += len(embs)
                del embs
                print ('Added %d / %d shards, (%d min)' % (shard_id+1, len(self.embed_paths), (time.time()-start_time)/60))
            out.seek(0)
            out.write(_flat_ip_header(self.dimension, ntotal))
        os.replace(self.index_path + '.partial', self.index_path)
        with open(self.meta_file, 'wb') as fout:
            pickle.dump(self.index_id_to_db_id, fout)
        print ('Adding took {} s'.format(time.time() - start_time))
        self.index = faiss.read_index(self.index_path)
        assert self.index.ntotal == ntotal == len(self.index_id_to_db_id)
        print(f'Total data indexed {len(self.index_id_to_db_id)}')

    def load_embeds(self, shard_id=None):
        all_ids, all_embeds = [], []
        offset = 0
        for embed_path in self.embed_paths:
            loaded_shard_id = int(re.search(r'_(\d+).pkl$', embed_path).group(1))
            if shard_id is not None and loaded_shard_id != shard_id:
                continue
            print(f"Loading pickle embedding from {embed_path}...")
            with open(embed_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)
            all_ids.extend([i + offset for i in ids])
            all_embeds.extend(embeddings)
            offset += len(ids)
        all_embeds = np.stack(all_embeds).astype(np.float32)
        datastore_size = len(all_ids)
        return all_embeds

    def get_embs(self, indices=None, shard_id=None):
        if indices is not None:
            embs = self.embs[indices]
        elif shard_id is not None:
            embs = self.load_embeds(shard_id)
        return embs
    
    def load_index_id_to_db_id(self,):
        with open(self.meta_file, "rb") as reader:
            index_id_to_db_id = pickle.load(reader)
        return index_id_to_db_id
    
    def build_passage_pos_id_map(self, ):
        convert_pkl_to_jsonl(self.passage_dir)
        passage_pos_ids = get_passage_pos_ids(self.passage_dir, self.pos_map_save_path)
        return passage_pos_ids

    def load_psg_pos_id_map(self,):
        if os.path.exists(self.pos_map_save_path):
            with open(self.pos_map_save_path, 'rb') as f:
                psg_pos_id_map = pickle.load(f)
        else:
            psg_pos_id_map = self.build_passage_pos_id_map()
        return psg_pos_id_map
    
    def _id2psg(self, shard_id, chunk_id, file=None):
        filename, position = self.psg_pos_id_map[shard_id][chunk_id]
        if file is None:
            with open(filename, 'r') as file:
                file.seek(position)
                line = file.readline()
        else:
            file.seek(position)
            line = file.readline()
        item = json.loads(line)
        # Every released passage records its own position; a mismatch means the index
        # and the passage file disagree, and the text would be some other passage.
        if (item.get("shard_id", shard_id), item.get("id", chunk_id)) != (shard_id, chunk_id):
            raise ValueError(f"passage lookup ({shard_id}, {chunk_id}) returned "
                             f"({item.get('shard_id')}, {item.get('id')}) from {filename}")
        return item
    
    def _db_id(self, index_id):
        try:
            shard_id, chunk_id = self.index_id_to_db_id[index_id]
        except:
            shard_id, chunk_id = self.default_shard_id, self.index_id_to_db_id[index_id]
        return shard_id, chunk_id

    def _get_passage(self, index_id):
        return self._id2psg(*self._db_id(index_id))

    def _get_passages(self, index_ids):
        # Local patch (muisti, MAS-396): read each distinct passage once, in file and offset
        # order, with one open per passage file. Opening the file for every hit (1.8M opens
        # on the NAS for TriviaQA's 17,944 x top 100) took ~1 h 50 min per 1.4T shard.
        db_ids = {i: self._db_id(i) for i in set(index_ids)}
        order = sorted(db_ids, key=lambda i: self.psg_pos_id_map[db_ids[i][0]][db_ids[i][1]])
        items, file, current = {}, None, None
        try:
            for i in order:
                filename = self.psg_pos_id_map[db_ids[i][0]][db_ids[i][1]][0]
                if filename != current:
                    if file is not None:
                        file.close()
                    file, current = open(filename, 'r'), filename
                items[i] = self._id2psg(*db_ids[i], file=file)
        finally:
            if file is not None:
                file.close()
        return items

    def get_retrieved_passages(self, all_indices, additional_metadata=[]):
        passages, db_ids, metadata = [], [], []
        items = self._get_passages([int(i) for query_indices in all_indices for i in query_indices])
        for query_indices in all_indices:
            retrieved_data_per_query = [items[int(index_id)] for index_id in query_indices]
            passages_per_query = [item["text"] for item in retrieved_data_per_query]
            additional_metadata_per_query = {metadata_key: [item.get(metadata_key, None) for item in retrieved_data_per_query] for metadata_key in additional_metadata}
            db_ids_per_query = [self.index_id_to_db_id[int(index_id)] for index_id in query_indices]
            passages.append(passages_per_query)
            db_ids.append(db_ids_per_query)
            metadata.append(additional_metadata_per_query)
        return passages, db_ids, metadata
    
    def _gpu_search(self, query_embs, k, q_block=1024, x_block=2_000_000):
        # Local patch (muisti, MAS-396): exact inner-product top-k on the GPU, in fp32 (no
        # TF32), reading the vectors in place from the faiss index. The CPU flat search took
        # ~2.5 h per 1.4T shard for TriviaQA; this is the same arithmetic, so scores agree to
        # fp32 rounding and the top-k differs only where two scores tie to that precision.
        xb = faiss.rev_swig_ptr(self.index.get_xb(), self.index.ntotal * self.index.d)
        xb = xb.reshape(self.index.ntotal, self.index.d)
        q = torch.from_numpy(np.ascontiguousarray(query_embs, dtype=np.float32)).cuda()
        best_s = torch.full((len(q), k), -float("inf"), device="cuda")
        best_i = torch.full((len(q), k), -1, dtype=torch.int64, device="cuda")
        with torch.no_grad():
            for j in range(0, len(xb), x_block):
                x = torch.from_numpy(xb[j:j + x_block]).cuda()
                for i in range(0, len(q), q_block):
                    s, idx = torch.topk(q[i:i + q_block] @ x.T, min(k, len(x)), dim=1)
                    cs = torch.cat([best_s[i:i + q_block], s], dim=1)
                    ci = torch.cat([best_i[i:i + q_block], idx + j], dim=1)
                    top = torch.topk(cs, k, dim=1)
                    best_s[i:i + q_block] = top.values
                    best_i[i:i + q_block] = torch.gather(ci, 1, top.indices)
                del x
        return best_s.cpu().numpy(), best_i.cpu().numpy()

    def search(self, query_embs, k=4096, additional_metadata=[]):
        if torch.cuda.is_available() and os.environ.get("MUISTI_SEARCH_CPU") != "1":
            torch.backends.cuda.matmul.allow_tf32 = False
            all_scores, all_indices = self._gpu_search(query_embs, k)
        else:
            all_scores, all_indices = self.index.search(query_embs.astype(np.float32), k)
        if len(additional_metadata) > 0:
            all_passages, db_ids, metadata = self.get_retrieved_passages(all_indices, additional_metadata)
            return all_scores.tolist(), all_passages, db_ids, metadata
        else:
            all_passages, db_ids, metadata = self.get_retrieved_passages(all_indices)
            return all_scores.tolist(), all_passages, db_ids, metadata
