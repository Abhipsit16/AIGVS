import faulthandler
faulthandler.enable()
import os
# --- CRITICAL C++ CRASH OVERRIDE ---
# Forces Windows to allow PyTorch and FAISS to share CPU threads without aborting
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import json
import numpy as np
import faiss
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm  # Added for tracking query loop progress

# Link your local AIGVS package
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from aigvs.pipeline import AIGVSPipeline

# --- HYPERPARAMETERS ---
SAMPLE_SIZE = 10000  # Number of paragraphs to index
TEST_QUERIES = 100   # Number of questions to evaluate
EMBEDDING_DIM = 384  # MiniLM dimension size
TOP_K = 10           # For Recall@10

def load_hotpot_data():
    """Downloads HotpotQA and formats it into isolated text chunks."""
    print("[1/6] Downloading HotpotQA Validation Set...")
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
    
    documents = {}
    queries = []
    
    print("[2/6] Parsing paragraphs and golden answers...")
    for item in dataset:
        # Directly access the 'title' array from the dictionary
        golden_titles = set(item["supporting_facts"]["title"])
        queries.append({
            "question": item["question"],
            "golden_titles": list(golden_titles)  # Converted to list for JSON serialization
        })
        
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            if title not in documents:
                documents[title] = " ".join(sentences)
                
        if len(documents) >= SAMPLE_SIZE:
            break
            
    return list(documents.items())[:SAMPLE_SIZE], queries[:TEST_QUERIES]

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
    import os
    
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

    # --- 3. AIGVS STATE ---
    print(">>> [AIGVS Custom SOTA State]")
    root_node = aigvs_pipeline.root
    active_attractors = len(root_node.children)
    print(f"Top-Level Hopfield Attractors: {active_attractors}")
    
    print("Leaf Node Distribution Profile:")
    total_chunks_found = 0
    # Added .items() to properly unpack the dictionary!
    for idx, (child_id, child_node) in enumerate(root_node.children.items()):
        leaf_size = len(child_node.chunks) if hasattr(child_node, 'chunks') else 0
        total_chunks_found += leaf_size
        print(f"  - Attractor [{child_id[-8:]}]: Capacity = {leaf_size}/{root_node.max_leaf_capacity}")
        
    print(f"Total Chunks Tracked in Tree:  {total_chunks_found}")
    print("==============================================\n")
    
    
def main():
    print("=== THE ULTRA-GRANULAR HOTPOT-QA BENCHMARK ===\n")
    
    # 1. Load Data & Model
    docs, queries = load_hotpot_data()
    print(f"\n[3/6] Loading local embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    print(f"Preparing {len(docs)} documents...")
    # 1. Define the texts and IDs FIRST, regardless of the cache
    doc_ids = [str(d[0]) for d in docs]
    doc_texts = [d[1] for d in docs]
    
    # 2. CACHED EMBEDDING LOADING
    cache_file = "hotpot_embeddings_cache.npy"
    if os.path.exists(cache_file):
        print("-> Found cached embeddings! Loading instantly...")
        doc_embeddings = np.load(cache_file)
    else:
        print(f"Embedding {len(docs)} documents (This will take a minute on CPU)...")
        doc_embeddings = model.encode(doc_texts, show_progress_bar=True)
        np.save(cache_file, doc_embeddings) # Save it so you never wait again!
    
    # Data container structured for deep paper analytics
    paper_metrics = {
        "experiment_parameters": {
            "sample_size": SAMPLE_SIZE,
            "total_queries_tested": TEST_QUERIES,
            "top_k_retrieved": TOP_K,
            "embedding_model": "all-MiniLM-L6-v2"
        },
        "global_performance": {
            "faiss": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0},
            "chroma": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0},
            "aigvs": {"build_time_sec": 0, "total_hits": 0, "avg_latency_ms": 0, "p99_latency_ms": 0, "mean_mrr": 0}
        },
        "query_logs": []  # This will store granular data for every single query
    }
    
    # ==========================================
    # TEST A: THE WRITE RACE (INDEXING SPEED)
    # ==========================================
    # ==========================================
    # TEST A: THE WRITE RACE (INDEXING SPEED)
    # ==========================================
    print("\n[4/6] STARTING THE WRITE RACE...")
    import sys
    
    # --- FAISS ---
    print("BREADCRUMB 1: Setting up FAISS vectors...", flush=True)
    faiss_vectors = np.ascontiguousarray(doc_embeddings.copy(), dtype=np.float32)
    faiss.normalize_L2(faiss_vectors)
    
    print("BREADCRUMB 2: Initializing FAISS Index...", flush=True)
    faiss.omp_set_num_threads(1)
    faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM) # Using Flat to rule out HNSW bug
    
    print("BREADCRUMB 3: Adding vectors to FAISS...", flush=True)
    start_time = time.time()
    faiss_index.add(faiss_vectors)
    paper_metrics["global_performance"]["faiss"]["build_time_sec"] = time.time() - start_time
    faiss_id_map = {i: doc_ids[i] for i in range(len(docs))}
    print("BREADCRUMB 4: FAISS survived!", flush=True)

    # --- CHROMA DB ---
    print("BREADCRUMB 5: Initializing ChromaDB...", flush=True)
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection(name="benchmark", metadata={"hnsw:space": "cosine"})
    
    print("BREADCRUMB 6: Adding vectors to ChromaDB...", flush=True)
    start_time = time.time()
    chroma_collection.add(ids=doc_ids, embeddings=doc_embeddings.tolist(), documents=doc_texts)
    paper_metrics["global_performance"]["chroma"]["build_time_sec"] = time.time() - start_time
    print("BREADCRUMB 7: ChromaDB survived!", flush=True)

    # --- AIGVS SOTA ---
    
    print("BREADCRUMB 8: Initializing AIGVS...", flush=True)
    
    # FIX: Use faiss_vectors[i] which we already normalized to the unit hypersphere!
    aigvs_payload = [
        {"id": doc_ids[i], "text": doc_texts[i], "embedding": faiss_vectors[i].copy()} 
        for i in range(len(docs))
    ]
    
    aigvs_pipeline = AIGVSPipeline(embedding_dim=EMBEDDING_DIM, storage_path="./benchmark_data")
    from aigvs.storage import InMemoryStorage
    aigvs_pipeline.storage = InMemoryStorage()
    aigvs_pipeline.root.max_leaf_capacity = 250
    
    start_time = time.time()
    aigvs_pipeline.ingest_hierarchical_json(aigvs_payload)
    paper_metrics["global_performance"]["aigvs"]["build_time_sec"] = time.time() - start_time
    
    print("BREADCRUMB 10: AIGVS survived!", flush=True)

    # ==========================================
    # TEST B: THE READ RACE (ACCURACY & LATENCY)
    # ==========================================
    print(f"\n[5/6] STARTING THE READ RACE (Running {TEST_QUERIES} Queries)...")
    
    # Local tracking arrays for macro stats calculation later
    f_lats, c_lats, a_lats = [], [], []
    f_mrrs, c_mrrs, a_mrrs = [], [], []
    
    # Wrapped with tqdm to provide a live, visual command-line progress bar
    for idx, q in enumerate(tqdm(queries, desc="Evaluating DB Vectors", unit="query")):
        q_text = q["question"]
        golden_targets = q["golden_titles"]
        
        q_vec = model.encode([q_text])[0]
        
        # FOOLPROOF L2 NORMALIZATION (Pure NumPy)
        # This guarantees AIGVS gets a vector with an exact length of 1.0
        q_vec_norm = q_vec / np.linalg.norm(q_vec) 
        
        # We still need to reshape it for FAISS, but we pass the 1D norm directly to AIGVS
        faiss_q = q_vec_norm.reshape(1, -1).astype(np.float32)
        
        # Object to collect every piece of information for this specific query step
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
        _, indices = faiss_index.search(q_vec_norm.reshape(1, -1), k=TOP_K)
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

        # --- AIGVS EXECUTION ---
        # --- AIGVS EXECUTION ---
        t0 = time.perf_counter()
        
        # FIX: q_vec_norm is already exactly (384,). Pass it directly!
        aigvs_results_raw = aigvs_pipeline.query(q_vec_norm, top_k=TOP_K)
        
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
        
        # Save this query's comprehensive journey to the global file trace
        paper_metrics["query_logs"].append(query_trace)

    # Compute macro analytics for the global headers
    for key, lats, mrrs in [("faiss", f_lats, f_mrrs), ("chroma", c_lats, c_mrrs), ("aigvs", a_lats, a_mrrs)]:
        paper_metrics["global_performance"][key]["avg_latency_ms"] = float(np.mean(lats))
        paper_metrics["global_performance"][key]["p99_latency_ms"] = float(np.percentile(lats, 99))
        paper_metrics["global_performance"][key]["mean_mrr"] = float(np.mean(mrrs))

    # ==========================================
    # DATA EXPORT
    # ==========================================
    print("\n[6/6] Freezing full execution trace to benchmark_results.json...")
    with open("benchmark_results.json", "w") as f:
        json.dump(paper_metrics, f, indent=4)
    print("Done! Every question, retrieved document array, and millisecond metric is safely stored.")

    # ==========================================
    # THE FINAL SCORECARD
    # ==========================================
    print("\n==============================================")
    print("             FINAL BENCHMARK SCORECARD")
    print("==============================================")
    for sys_name in ["faiss", "chroma", "aigvs"]:
        gp = paper_metrics["global_performance"][sys_name]
        recall = (gp["total_hits"] / TEST_QUERIES) * 100
        print(f"--- {sys_name.upper()} ---")
        print(f"Build Time:  {gp['build_time_sec']:.2f} sec")
        print(f"Latency:     {gp['avg_latency_ms']:.2f} ms (Avg) | {gp['p99_latency_ms']:.2f} ms (P99)")
        print(f"Recall@{TOP_K}:    {recall:.1f}%")
        print(f"Mean RR:     {gp['mean_mrr']:.3f}\n")
    print("==============================================")
    
    print_database_states(faiss_index, chroma_collection, aigvs_pipeline)
    # --- 3. AIGVS STATE (RECURSIVE X-RAY) ---
    print(">>> [AIGVS Custom SOTA State]")
    
    def trace_tree(node, depth=0):
        indent = "  " * depth
        if hasattr(node, 'children') and node.children:
            print(f"{indent}[Branch Level {depth}] -> {len(node.children)} sub-clusters")
            for child_id, child_node in list(node.children.items())[:3]: # Look at first 3
                trace_tree(child_node, depth + 1)
            if len(node.children) > 3:
                print(f"{indent}  ... (+ {len(node.children) - 3} more branches)")
        else:
            # Dynamically hunt for the list of IDs/Chunks regardless of what it's named
            contents = next((v for k, v in vars(node).items() if isinstance(v, list)), [])
            print(f"{indent}[LEAF] -> Holds {len(contents)} items")

    trace_tree(aigvs_pipeline.root)
    print("==============================================\n")
if __name__ == "__main__":
    main()