import faulthandler
faulthandler.enable()
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import subprocess

if sys.platform == "win32" and os.environ.get("PYTHONUTF8") != "1":
    os.environ["PYTHONUTF8"] = "1"
    sys.exit(subprocess.call([sys.executable, "-X", "utf8"] + sys.argv))

# CRITICAL: Import pyarrow before absolutely anything else (especially tensorflow/pytorch)
import pyarrow
import pyarrow.dataset

import time
import json
import numpy as np
import psutil
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import chromadb

# Link your local AIGVS package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from aigvs.pipeline import AIGVSPipeline

# --- HYPERPARAMETERS ---
SAMPLE_SIZE = 100000  # Number of paragraphs to index (start small)
TEST_QUERIES = 200   # Number of questions to evaluate
EMBEDDING_DIM = 384  # MiniLM dimension size
TOP_K = 10           # For Recall@10

def load_msmarco_data():
    import ir_datasets
    print("[1/6] Loading MSMARCO...")
    dataset = ir_datasets.load("msmarco-passage/train")

    queries = []
    golden_doc_ids = set()

    print("Loading queries...")
    qrels = dataset.qrels_dict()
    for q in dataset.queries_iter():
        if q.query_id in qrels:
            rel_docs = qrels[q.query_id]
            queries.append({
                "question": q.text,
                "golden_titles": [str(did) for did in rel_docs.keys()]
            })
            golden_doc_ids.update([str(did) for did in rel_docs.keys()])
            if len(queries) >= TEST_QUERIES:
                break

    print(f"Loaded {len(queries)} queries with {len(golden_doc_ids)} golden documents.")

    filler_docs_needed = SAMPLE_SIZE - len(golden_doc_ids)
    if filler_docs_needed < 0:
        filler_docs_needed = 0

    found_golden = 0
    golden_docs_data = []
    filler_docs_data = []

    print(f"Scanning corpus to guarantee inclusion of golden targets (This takes ~30 seconds)...")
    for doc in dataset.docs_iter():
        did = str(doc.doc_id)
        if did in golden_doc_ids:
            golden_docs_data.append((did, doc.text))
            found_golden += 1
        elif len(filler_docs_data) < filler_docs_needed:
            filler_docs_data.append((did, doc.text))
            
        if len(filler_docs_data) >= filler_docs_needed and found_golden == len(golden_doc_ids):
            break

    docs = golden_docs_data + filler_docs_data
    print(f"Loaded {len(docs)} documents")

    return docs, queries

def calculate_mrr(results, golden_targets):
    """Calculates the Reciprocal Rank for a single query."""
    for rank, res_id in enumerate(results, 1):
        if res_id in golden_targets:
            return 1.0 / rank
    return 0.0

def print_database_states(faiss_index, chroma_collection, aigvs_pipeline):
    print("\n==============================================")
    print("         EXAMINING INTERNAL DB STATES")
    print("==============================================")
    
    # --- 1. FAISS STATE ---
    print(">>> [FAISS Internal State]")
    print(f"Total Vectors in Memory: {faiss_index.ntotal}")
    if hasattr(faiss_index, 'hnsw'):
        print(f"HNSW Max Links (M):      {faiss_index.hnsw.nb_neighbors(0) if faiss_index.ntotal > 0 else 'N/A'}")
        print(f"Graph Storage Type:      HNSW Graph")
    else:
        print(f"Graph Storage Type:      Flat / Uncompressed vectors (Fallback mode)")
    print("-" * 30)

    # --- 2. CHROMA STATE ---
    print(">>> [ChromaDB Internal State]")
    print(f"Total Items in Collection: {chroma_collection.count()}")
    peek_data = chroma_collection.peek(limit=2)
    print("Document Snapshot PEEK:")
    for doc_id, text in zip(peek_data["ids"], peek_data["documents"]):
        print(f"  - [ID: {doc_id}] -> {text[:60]}...")
    print("-" * 30)

    # --- 3. AIGVS (RUST) STATE ---
    print(">>> [AIGVS Rust System State]")
    root_node = aigvs_pipeline.root
    active_attractors = len(root_node.children)
    print(f"Top-Level Trunks/Attractors: {active_attractors}")
    
    print("Node Distribution Profile:")
    total_chunks_found = 0
    
    def count_chunks_in_tree(node):
        if hasattr(node, 'children') and node.children:
            total = 0
            for child in node.children.values():
                total += count_chunks_in_tree(child)
            return total
        else:
            if hasattr(node, '_count'):
                return node._count
            elif hasattr(node, 'chunk_ids'):
                return len(node.chunk_ids)
            return 0

    for idx, (child_id, child_node) in enumerate(root_node.children.items()):
        true_capacity = count_chunks_in_tree(child_node)
        total_chunks_found += true_capacity
        
        node_type = "Branch" if hasattr(child_node, 'children') else "Leaf"
        print(f"  - {node_type} [{child_id[-8:]}]: Contains {true_capacity} chunks deep")
        
    print(f"Total Chunks Tracked in Tree:  {total_chunks_found}")
    print("==============================================\n")

def main():
    print("=== THE ULTRA-GRANULAR MSMARCO BENCHMARK (RUST EDITION) ===\n")
    
    # 1. Load Data & Model
    docs, queries = load_msmarco_data()
    print(f"\n[3/6] Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Preparing {len(docs)} documents...")
    doc_ids = [str(d[0]) for d in docs]
    doc_texts = [d[1] for d in docs]
    
    # CACHED EMBEDDING LOADING
    cache_file = f"msmarco_embeddings_cache_v2_{SAMPLE_SIZE}.npy"
    if os.path.exists(cache_file):
        print("-> Found cached embeddings! Loading instantly...")
        doc_embeddings = np.load(cache_file)
    else:
        print(f"Embedding {len(docs)} documents (This will take a minute on CPU)...")
        doc_embeddings = model.encode(doc_texts, show_progress_bar=True)
        np.save(cache_file, doc_embeddings) 
    
    # Data container structured for deep paper analytics
    paper_metrics = {
        "experiment_parameters": {
            "sample_size": SAMPLE_SIZE,
            "total_queries_tested": TEST_QUERIES,
            "top_k_retrieved": TOP_K,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "global_performance": {
            "faiss": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0},
            "chroma": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0},
            "aigvs": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0, "memory_mb": 0}
        },
        "query_logs": [] 
    }
    
    print("\n[4/6] STARTING THE WRITE RACE...")
    
    # --- FAISS (HNSW) ---
    import gc
    gc.collect()
    mem_before_faiss = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    print("BREADCRUMB 1: Setting up FAISS vectors...", flush=True)
    faiss_vectors = np.ascontiguousarray(doc_embeddings.copy(), dtype=np.float32)
    faiss.normalize_L2(faiss_vectors)
    
    print("BREADCRUMB 2: Initializing FAISS Index (HNSW)...", flush=True)
    faiss.omp_set_num_threads(1)
    faiss_index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 32)
    faiss_index.hnsw.efConstruction = 200
    faiss_index.hnsw.efSearch = 128
    
    print("BREADCRUMB 3: Adding vectors to FAISS...", flush=True)
    start_time = time.time()
    faiss_index.add(faiss_vectors)
    paper_metrics["global_performance"]["faiss"]["build_time_sec"] = time.time() - start_time
    faiss_id_map = {i: doc_ids[i] for i in range(len(docs))}
    print("BREADCRUMB 4: FAISS survived!", flush=True)
    
    del faiss_vectors
    gc.collect()
    
    mem_after_faiss = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    faiss_mem_mb = mem_after_faiss - mem_before_faiss
    paper_metrics["global_performance"]["faiss"]["memory_mb"] = faiss_mem_mb
    print(f"Incremental Memory (FAISS): {faiss_mem_mb:.1f} MB\n")

    # =========================================================================
    # ChromaDB (PyChroma / hnswlib)
    # =========================================================================
    gc.collect()
    mem_before_chroma = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    print("BREADCRUMB 5: Initializing ChromaDB...", flush=True)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection(name="benchmark_msmarco_rust", metadata={"hnsw:space": "cosine", "hnsw:construction_ef": 32, "hnsw:M": 32})
    
    print("BREADCRUMB 6: Adding vectors to ChromaDB...", flush=True)
    start_time = time.time()
    
    batch_size = 5000
    for i in range(0, len(doc_ids), batch_size):
        chroma_collection.add(
            ids=doc_ids[i:i+batch_size], 
            embeddings=doc_embeddings[i:i+batch_size].tolist(), 
            documents=doc_texts[i:i+batch_size]
        )
        
    paper_metrics["global_performance"]["chroma"]["build_time_sec"] = time.time() - start_time
    print("BREADCRUMB 7: ChromaDB survived!", flush=True)
    
    gc.collect()
    mem_after_chroma = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    chroma_mem_mb = mem_after_chroma - mem_before_chroma
    paper_metrics["global_performance"]["chroma"]["memory_mb"] = chroma_mem_mb
    print(f"Incremental Memory (Chroma): {chroma_mem_mb:.1f} MB\n")

    # --- AIGVS RUST SYSTEM ---
    gc.collect()
    mem_before_aigvs = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    
    # Normalize vectors for AIGVS
    aigvs_vectors = np.ascontiguousarray(doc_embeddings.copy(), dtype=np.float32)
    norms = np.linalg.norm(aigvs_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    aigvs_vectors = aigvs_vectors / norms

    aigvs_payload = [
        {"id": doc_ids[i], "text": doc_texts[i], "embedding": aigvs_vectors[i].copy()} 
        for i in range(len(docs))
    ]
    print("\nBREADCRUMB 8: Initializing AIGVS Rust System...", flush=True)
    
    aigvs_pipeline = AIGVSPipeline(embedding_dim=EMBEDDING_DIM, ef_construction=32, max_connections=64)
    from aigvs.storage import InMemoryStorage
    aigvs_pipeline.storage = InMemoryStorage()
    
    start_time = time.time()
    aigvs_pipeline.ingest_hierarchical_json(aigvs_payload)
    paper_metrics["global_performance"]["aigvs"]["build_time_sec"] = time.time() - start_time
    
    print("BREADCRUMB 10: AIGVS Rust System survived!", flush=True)
    
    del aigvs_payload
    del aigvs_vectors
    gc.collect()
    
    mem_after_aigvs = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    aigvs_mem_mb = mem_after_aigvs - mem_before_aigvs
    paper_metrics["global_performance"]["aigvs"]["memory_mb"] = aigvs_mem_mb
    print(f"Incremental Memory (AIGVS Rust): {aigvs_mem_mb:.1f} MB")
    
    # ==========================================
    # TEST B: THE READ RACE (ACCURACY & LATENCY)
    # ==========================================
    print(f"\n[5/6] STARTING THE READ RACE (Running {TEST_QUERIES} Queries)...")
    
    f_lats, c_lats, a_lats = [], [], []
    f_mrrs, c_mrrs, a_mrrs = [], [], []
    
    for idx, q in enumerate(tqdm(queries, desc="Evaluating DB Vectors", unit="query")):
        q_text = q["question"]
        golden_targets = q["golden_titles"]
        
        q_vec = model.encode([q_text])[0]
        q_vec_norm = q_vec / np.linalg.norm(q_vec) 
        faiss_q = q_vec_norm.reshape(1, -1).astype(np.float32)
        
        query_trace = {
            "query_index": idx,
            "question": q_text,
            "golden_parameters": golden_targets,
            "faiss_metrics": {},
            "chroma_metrics": {},
            "aigvs_metrics": {}
        }
        
        # --- FAISS EXECUTION ---
        t0 = time.perf_counter()
        _, indices = faiss_index.search(faiss_q, k=TOP_K)
        faiss_time = (time.perf_counter() - t0) * 1000 
        faiss_results = [faiss_id_map[i] for i in indices[0] if i != -1]
        faiss_mrr = calculate_mrr(faiss_results, golden_targets)
        faiss_hit = 1 if any(target in faiss_results for target in golden_targets) else 0
        
        f_lats.append(faiss_time)
        f_mrrs.append(faiss_mrr)
        paper_metrics["global_performance"]["faiss"]["total_hits"] += faiss_hit
        
        query_trace["faiss_metrics"] = {
            "latency_ms": faiss_time,
            "mrr": faiss_mrr,
            "hit_success": bool(faiss_hit),
            "returned_ids": faiss_results
        }

        # --- CHROMA EXECUTION ---
        t0 = time.perf_counter()
        chroma_res = chroma_collection.query(query_embeddings=[q_vec.tolist()], n_results=TOP_K)
        chroma_time = (time.perf_counter() - t0) * 1000
        chroma_results = chroma_res["ids"][0]
        chroma_mrr = calculate_mrr(chroma_results, golden_targets)
        chroma_hit = 1 if any(target in chroma_results for target in golden_targets) else 0
        
        c_lats.append(chroma_time)
        c_mrrs.append(chroma_mrr)
        paper_metrics["global_performance"]["chroma"]["total_hits"] += chroma_hit
        
        query_trace["chroma_metrics"] = {
            "latency_ms": chroma_time,
            "mrr": chroma_mrr,
            "hit_success": bool(chroma_hit),
            "returned_ids": chroma_results
        }

        # --- AIGVS RUST EXECUTION ---
        t0 = time.perf_counter()
        aigvs_results_raw = aigvs_pipeline.query(q_vec_norm, top_k=TOP_K, target_confidence=0.98, beta=5)
        aigvs_time = (time.perf_counter() - t0) * 1000
        aigvs_results = [item["id"] for item in aigvs_results_raw]
        aigvs_mrr = calculate_mrr(aigvs_results, golden_targets)
        aigvs_hit = 1 if any(target in aigvs_results for target in golden_targets) else 0
        
        a_lats.append(aigvs_time)
        a_mrrs.append(aigvs_mrr)
        paper_metrics["global_performance"]["aigvs"]["total_hits"] += aigvs_hit
        
        query_trace["aigvs_metrics"] = {
            "latency_ms": aigvs_time,
            "mrr": aigvs_mrr,
            "hit_success": bool(aigvs_hit),
            "returned_ids": aigvs_results
        }
        
        paper_metrics["query_logs"].append(query_trace)

    for key, lats, mrrs in [("faiss", f_lats, f_mrrs), ("chroma", c_lats, c_mrrs), ("aigvs", a_lats, a_mrrs)]:
        paper_metrics["global_performance"][key]["avg_latency_ms"] = float(np.mean(lats))
        paper_metrics["global_performance"][key]["p99_latency_ms"] = float(np.percentile(lats, 99))
        paper_metrics["global_performance"][key]["mean_mrr"] = float(np.mean(mrrs))

    print("\n[6/6] Freezing full execution trace to benchmark_results_msmarco_rust.json...")
    with open("benchmark_results_msmarco_rust.json", "w") as f:
        json.dump(paper_metrics, f, indent=4)
    print("Done! Every question, retrieved document array, and millisecond metric is safely stored.")

    print("\n==============================================")
    print("             FINAL BENCHMARK SCORECARD")
    print("==============================================")
    for sys_name in ["faiss", "chroma", "aigvs"]:
        gp = paper_metrics["global_performance"][sys_name]
        recall = (gp["total_hits"] / TEST_QUERIES) * 100
        sys_display = "AIGVS (Rust)" if sys_name == "aigvs" else sys_name.upper()
        print(f"--- {sys_display} ---")
        print(f"Memory:      {gp['memory_mb']:.1f} MB")
        print(f"Build Time:  {gp['build_time_sec']:.2f} sec")
        print(f"Latency:     {gp['avg_latency_ms']:.2f} ms (Avg) | {gp['p99_latency_ms']:.2f} ms (P99)")
        print(f"Recall@{TOP_K}:    {recall:.1f}%")
        print(f"Mean RR:     {gp['mean_mrr']:.3f}\n")
    print("==============================================")
    
    print_database_states(faiss_index, chroma_collection, aigvs_pipeline)

    if hasattr(aigvs_pipeline.engine, "print_profile_stats"):
        aigvs_pipeline.engine.print_profile_stats()

    from src.aigvs.profiler import report
    report()

if __name__ == "__main__":
    main()

