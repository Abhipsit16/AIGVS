import faulthandler
faulthandler.enable()
import os
import sys
import subprocess

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    sys.exit(subprocess.call([sys.executable, "-X", "utf8"] + sys.argv))

# CRITICAL: Import pyarrow before absolutely anything else (especially tensorflow/pytorch/faiss)
import pyarrow
import pyarrow.dataset

import json
import time
import numpy as np
import psutil
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
import gc

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from aigvs.pipeline import AIGVSPipeline
from aigvs.storage import InMemoryStorage

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "hotpot_bench_data")
QUERIES_FILE = os.path.join(os.path.dirname(__file__), "benchmark_queries_export.json")

EMBEDDING_DIM = 384
TOP_K_DEEP = 100
TOP_K_SHALLOW = 10

def calc_ndcg(results, golden_targets, k=10):
    dcg = 0.0
    for i, res in enumerate(results[:k]):
        if res in golden_targets:
            dcg += 1.0 / np.log2(i + 2)
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(golden_targets), k))])
    return dcg / idcg if idcg > 0 else 0.0

def yield_all_vectors_and_ids(total_chunks, load_text=False):
    for chunk_idx in range(total_chunks):
        npy_path = os.path.join(DATA_DIR, f"embeddings_chunk_{chunk_idx}.npy")
        txt_path = os.path.join(DATA_DIR, f"corpus_chunk_{chunk_idx}.jsonl")
        
        if not os.path.exists(npy_path):
            continue
            
        vectors = np.load(npy_path, mmap_mode='r')
        ids, texts = [], []
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                ids.append(data["id"])
                if load_text:
                    texts.append(data["text"])
                    
        yield vectors, ids, texts

def ensure_corpus_jsonl(total_chunks):
    chunk_size = 100000
    missing = any(not os.path.exists(os.path.join(DATA_DIR, f"corpus_chunk_{i}.jsonl")) for i in range(total_chunks))
    if missing:
        print("Text corpus chunks are missing. Streaming from HuggingFace to rebuild them...")
        from datasets import load_dataset
        corpus_dataset = load_dataset("BeIR/hotpotqa", "corpus", split="corpus")
        current_chunk_idx = 0
        current_chunk_docs = []
        for doc in corpus_dataset:
            did = str(doc["_id"])
            title = doc.get("title", "")
            text = doc.get("text", "")
            doc_text = f"{title}\n{text}" if title else text
            current_chunk_docs.append({"id": did, "text": doc_text})
            
            if len(current_chunk_docs) >= chunk_size:
                chunk_file = os.path.join(DATA_DIR, f"corpus_chunk_{current_chunk_idx}.jsonl")
                with open(chunk_file, "w", encoding="utf-8") as f:
                    for d in current_chunk_docs:
                        f.write(json.dumps(d) + "\n")
                current_chunk_idx += 1
                current_chunk_docs = []
                if current_chunk_idx >= total_chunks:
                    break
                    
        if current_chunk_docs and current_chunk_idx < total_chunks:
            chunk_file = os.path.join(DATA_DIR, f"corpus_chunk_{current_chunk_idx}.jsonl")
            with open(chunk_file, "w", encoding="utf-8") as f:
                for d in current_chunk_docs:
                    f.write(json.dumps(d) + "\n")
        print("Rebuilt text corpus chunks!")

def main():
    total_chunks = 53
    ensure_corpus_jsonl(total_chunks)
    
    print("Loading queries...")
    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)
    
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
    
    aigvs_pipeline = AIGVSPipeline(embedding_dim=EMBEDDING_DIM)
    aigvs_pipeline.storage = InMemoryStorage()
    
    print("Ingesting chunks to AIGVS...")
    start_time = time.time()
    mem_before = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    for vectors, ids, texts in tqdm(yield_all_vectors_and_ids(total_chunks, load_text=True), total=total_chunks, desc="Ingesting chunks to AIGVS"):
        chunk_vecs = np.array(vectors, copy=True, dtype=np.float32)
        chunk_vecs = np.ascontiguousarray(chunk_vecs)
        faiss.normalize_L2(chunk_vecs) 
        payload = [{"id": ids[idx], "text": texts[idx], "embedding": chunk_vecs[idx]} for idx in range(len(ids))]
        aigvs_pipeline.ingest_batches(payload)
        del chunk_vecs, payload
        
    build_time_sec = time.time() - start_time
    memory_mb = (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024) - mem_before
    print(f"Ingestion done! Time: {build_time_sec:.2f}s, Mem: {memory_mb:.2f}MB")
    
    print("Evaluating AIGVS queries...")
    lats, r10s, r100s, mrrs, ndcgs = [], [], [], [], []
    
    for q in tqdm(queries, desc="AIGVS Read Race"):
        q_vec = model.encode([q["question"]])[0]
        q_vec_norm = q_vec / np.linalg.norm(q_vec)
        golden = q["golden_targets"]
        
        t0 = time.perf_counter()
        res_raw = aigvs_pipeline.query(q_vec_norm, top_k=TOP_K_DEEP)
        latency = (time.perf_counter() - t0) * 1000
        
        results = [item["id"] for item in res_raw]
        top_10 = results[:TOP_K_SHALLOW]
        
        lats.append(latency)
        r10s.append(1.0 if any(g in top_10 for g in golden) else 0.0)
        r100s.append(1.0 if any(g in results for g in golden) else 0.0)
        mrrs.append(next((1.0 / (rank + 1) for rank, res in enumerate(top_10) if res in golden), 0.0))
        ndcgs.append(calc_ndcg(results, golden, k=TOP_K_SHALLOW))
        
    print("\nRESULTS:")
    print(f"Avg Latency: {np.mean(lats):.2f} ms")
    print(f"Recall@10: {np.mean(r10s)*100:.2f}%")
    print(f"Recall@100: {np.mean(r100s)*100:.2f}%")
    print(f"MRR@10: {np.mean(mrrs):.4f}")
    print(f"NDCG@10: {np.mean(ndcgs):.4f}")
    
    metrics = {
        "build_time_sec": build_time_sec,
        "memory_mb": memory_mb,
        "avg_latency_ms": float(np.mean(lats)),
        "recall_10": float(np.mean(r10s)) * 100,
        "recall_100": float(np.mean(r100s)) * 100,
        "mean_mrr": float(np.mean(mrrs)),
        "mean_ndcg": float(np.mean(ndcgs))
    }
    
    out_file = os.path.join(os.path.dirname(__file__), "full_aigvs_metrics.json")
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {out_file}")

if __name__ == "__main__":
    main()
