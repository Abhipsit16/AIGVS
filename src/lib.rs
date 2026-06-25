use pyo3::prelude::*;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet, HashMap};
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::sync::atomic::AtomicU64;

static ROUTER_SCORE_US: AtomicU64 = AtomicU64::new(0);
static ROUTER_SOFTMAX_US: AtomicU64 = AtomicU64::new(0);
static ROUTER_SORT_US: AtomicU64 = AtomicU64::new(0);
static ROUTER_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static CHILD_QUERY_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static RERANK_TOTAL_US: AtomicU64 = AtomicU64::new(0);
static CANDIDATE_VECTORS: AtomicU64 = AtomicU64::new(0);
static LEAVES_VISITED: AtomicU64 = AtomicU64::new(0);
static QUERY_COUNT: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------
// FLOATING POINT SAFETY & HEAP ORD WRAPPERS
// ---------------------------------------------------------

#[derive(PartialEq, Clone, Copy)]
struct Score(f32);
impl Eq for Score {}
impl PartialOrd for Score {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { self.0.partial_cmp(&other.0) }
}
impl Ord for Score {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

#[derive(PartialEq, Clone, Copy)]
struct MinScore(f32);
impl Eq for MinScore {}
impl PartialOrd for MinScore {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { other.0.partial_cmp(&self.0) }
}
impl Ord for MinScore {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering { self.partial_cmp(other).unwrap_or(Ordering::Equal) }
}

// ---------------------------------------------------------
// DATA STORES
// ---------------------------------------------------------

enum ExecutionNode {
    Branch {
        router_idx: usize,
        children_mapping: HashMap<String, usize>,
    },
    Leaf {
        graph_idx: usize,
    },
}

#[pyclass]
struct AIGVSEngine {
    embedding_dim: usize,
    nodes: Vec<ExecutionNode>,
    node_id_to_index: HashMap<String, usize>,
    root_index: Option<usize>,
    routers: Vec<RustHopfieldRouterInternal>,
    graphs: Vec<RustNSWGraphInternal>,
}

struct RustHopfieldRouterInternal {
    id_lookup: Vec<String>,
    /// Flattened row-major attractor matrix: row i = x_matrix[i*dim .. (i+1)*dim].
    x_matrix: Vec<f32>,
}

struct RustNSWGraphInternal {
    chunk_ids: Vec<String>,
    /// Flattened row-major vector cache: row i = vector_cache[i*dim .. (i+1)*dim].
    vector_cache: Vec<i8>,
    local_graph: Vec<RwLock<Vec<usize>>>,
    deleted_indices: HashSet<usize>,
    entry_point_idx: Option<usize>,
    max_connections: usize,
    ef_construction: usize,
}

// ---------------------------------------------------------
// SIMD DOT PRODUCT
// 4-way unrolled. For pre-normalised unit vectors this equals cosine similarity.
// The compiler will auto-vectorise this loop to AVX2/AVX-512 with -C target-cpu=native.
// ---------------------------------------------------------

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let unrolled = len - (len % 4);
    let mut s = 0.0f32;
    let mut i = 0;
    while i < unrolled {
        unsafe {
            s += a.get_unchecked(i)     * b.get_unchecked(i);
            s += a.get_unchecked(i + 1) * b.get_unchecked(i + 1);
            s += a.get_unchecked(i + 2) * b.get_unchecked(i + 2);
            s += a.get_unchecked(i + 3) * b.get_unchecked(i + 3);
        }
        i += 4;
    }
    for j in unrolled..len {
        unsafe { s += a.get_unchecked(j) * b.get_unchecked(j); }
    }
    s
}

#[inline(always)]
fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let len = a.len();
    let unrolled = len - (len % 8);
    let mut s = 0i32;
    let mut i = 0;
    while i < unrolled {
        unsafe {
            s += *a.get_unchecked(i) as i32 * *b.get_unchecked(i) as i32;
            s += *a.get_unchecked(i + 1) as i32 * *b.get_unchecked(i + 1) as i32;
            s += *a.get_unchecked(i + 2) as i32 * *b.get_unchecked(i + 2) as i32;
            s += *a.get_unchecked(i + 3) as i32 * *b.get_unchecked(i + 3) as i32;
            s += *a.get_unchecked(i + 4) as i32 * *b.get_unchecked(i + 4) as i32;
            s += *a.get_unchecked(i + 5) as i32 * *b.get_unchecked(i + 5) as i32;
            s += *a.get_unchecked(i + 6) as i32 * *b.get_unchecked(i + 6) as i32;
            s += *a.get_unchecked(i + 7) as i32 * *b.get_unchecked(i + 7) as i32;
        }
        i += 8;
    }
    for j in unrolled..len {
        unsafe { s += *a.get_unchecked(j) as i32 * *b.get_unchecked(j) as i32; }
    }
    s
}

// ---------------------------------------------------------
// ROUTER BEAM SEARCH
// Sequential scan — NOT parallelised with Rayon.
// Rayon's thread-pool overhead (~5–20µs spin-up) exceeds the actual dot-product
// work when a branch has fewer than ~200 children, which is typical here.
// At that scale a tight sequential loop over a contiguous f32 slice is faster
// because it stays entirely in L1 cache.
// ---------------------------------------------------------

#[inline]
fn router_beam_search(
    x_matrix: &[f32],
    id_lookup: &[String],
    children_mapping: &HashMap<String, usize>,
    q: &[f32],
    dim: usize,
    max_beam: usize,
    target_confidence: f32,
    beta: f32,
    activation_threshold: f32,
) -> Vec<usize> {
    if id_lookup.is_empty() { return vec![]; }

    let n = id_lookup.len();

    let t0 = std::time::Instant::now();
    // Score: tight sequential dot-product scan, fully in L1/L2 cache.
    let mut raw_scores: Vec<f32> = Vec::with_capacity(n);
    let mut max_s = f32::NEG_INFINITY;
    for row in x_matrix.chunks_exact(dim) {
        let s = dot(q, row);
        if s > max_s { max_s = s; }
        raw_scores.push(s);
    }
    let t1 = std::time::Instant::now();

    // Activation threshold: zero out children whose raw cosine is below minimum.
    // This is the OOD rejection gate — prevents routing into irrelevant branches.
    let mut any_above = false;
    for s in raw_scores.iter() {
        if *s >= activation_threshold { any_above = true; break; }
    }
    if !any_above { return vec![]; }

    // Numerically stable softmax with temperature beta, only over activated children.
    let mut exp_sum = 0.0f32;
    let mut probs: Vec<(usize, f32)> = raw_scores
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            if s < activation_threshold {
                (i, 0.0f32)
            } else {
                let e = ((s - max_s) * beta).exp();
                exp_sum += e;
                (i, e)
            }
        })
        .collect();
    for (_, p) in probs.iter_mut() {
        if *p > 0.0 { *p /= exp_sum + 1e-9; }
    }
    let t2 = std::time::Instant::now();

    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut selected = Vec::with_capacity(max_beam.min(n));
    let mut cumulative = 0.0f32;
    for (idx, prob) in probs.into_iter().take(max_beam) {
        if prob == 0.0 { break; }  // All remaining are below threshold
        if let Some(&child_node_idx) = children_mapping.get(&id_lookup[idx]) {
            selected.push(child_node_idx);
        }
        cumulative += prob;
        if cumulative >= target_confidence { break; }
    }
    let t3 = std::time::Instant::now();

    ROUTER_SCORE_US.fetch_add(t1.duration_since(t0).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTER_SOFTMAX_US.fetch_add(t2.duration_since(t1).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTER_SORT_US.fetch_add(t3.duration_since(t2).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    ROUTER_TOTAL_US.fetch_add(t3.duration_since(t0).as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
    CANDIDATE_VECTORS.fetch_add(n as u64, std::sync::atomic::Ordering::Relaxed);

    selected
}

// ---------------------------------------------------------
// NSW SEARCH CANDIDATES
// Standard HNSW Algorithm 2 (search layer).
// Two-heap design: max-heap for frontier expansion, min-heap for result set.
// Early exit when the best unseen candidate is worse than the worst result.
// ---------------------------------------------------------

fn nsw_search_candidates(
    vector_cache: &[i8],
    local_graph: &[RwLock<Vec<usize>>],
    deleted_indices: &HashSet<usize>,
    q: &[i8],
    dim: usize,
    entry: usize,
    ef: usize,
    visited: &mut [bool],
) -> BinaryHeap<(Score, usize)> {
    let mut candidates: BinaryHeap<(Score, usize)> = BinaryHeap::new();
    let mut results:    BinaryHeap<(MinScore, usize)> = BinaryHeap::new();

    let entry_score = dot_i8(q, &vector_cache[entry * dim..(entry + 1) * dim]) as f32;
    candidates.push((Score(entry_score), entry));
    if !deleted_indices.contains(&entry) {
        results.push((MinScore(entry_score), entry));
    }
    visited[entry] = true;

    while let Some((Score(cand_score), curr)) = candidates.pop() {
        if results.len() >= ef {
            if let Some(&(MinScore(worst), _)) = results.peek() {
                if cand_score < worst { break; }
            }
        }
        for &nb in &*local_graph[curr].read() {
            if !visited[nb] {
                visited[nb] = true;
                let nb_score = dot_i8(q, &vector_cache[nb * dim..(nb + 1) * dim]) as f32;
                candidates.push((Score(nb_score), nb));
                if !deleted_indices.contains(&nb) {
                    if results.len() < ef {
                        results.push((MinScore(nb_score), nb));
                    } else if let Some(&(MinScore(worst), _)) = results.peek() {
                        if nb_score > worst {
                            results.pop();
                            results.push((MinScore(nb_score), nb));
                        }
                    }
                }
            }
        }
    }

    results.into_iter()
        .map(|(MinScore(s), idx)| (Score(s), idx))
        .collect()
}

/// Query a single leaf, returning up to `ef` results as (chunk_id, score) sorted descending.
fn nsw_query(
    graph: &RustNSWGraphInternal,
    q: &[i8],
    dim: usize,
    ef: usize,
) -> Vec<(String, f32)> {
    let ep = match graph.entry_point_idx { Some(e) => e, None => return vec![] };
    let mut visited = vec![false; graph.local_graph.len()];
    let heap = nsw_search_candidates(
        &graph.vector_cache, &graph.local_graph, &graph.deleted_indices, q, dim, ep, ef, &mut visited
    );
    let mut v: Vec<(Score, usize)> = heap.into_sorted_vec();
    v.reverse();
    v.into_iter()
        .map(|(Score(s), idx)| (graph.chunk_ids[idx].clone(), s))
        .collect()
}

// ---------------------------------------------------------
// NSW INSERT HELPERS
// ---------------------------------------------------------

fn heuristic_select(
    candidates: &[(usize, f32)],
    vector_cache: &[i8],
    dim: usize,
    m: usize,
) -> Vec<usize> {
    let mut result = Vec::with_capacity(m);
    for &(cand_idx, cand_sim_to_pivot) in candidates {
        if result.len() >= m { break; }
        let mut is_good = true;
        let cand_vec = &vector_cache[cand_idx * dim..(cand_idx + 1) * dim];
        for &r_idx in &result {
            let r_vec = &vector_cache[r_idx * dim..(r_idx + 1) * dim];
            let sim_to_r = dot_i8(cand_vec, r_vec) as f32;
            if sim_to_r > cand_sim_to_pivot {
                is_good = false;
                break;
            }
        }
        if is_good {
            result.push(cand_idx);
        }
    }
    if result.len() < m {
        for &(cand_idx, _) in candidates {
            if result.len() >= m { break; }
            if !result.contains(&cand_idx) {
                result.push(cand_idx);
            }
        }
    }
    result
}

fn prune_neighbours(
    neighbours: &mut Vec<usize>,
    vector_cache: &[i8],
    pivot: usize,
    dim: usize,
    max_conn: usize,
) {
    if neighbours.len() <= max_conn { return; }
    let pivot_vec = &vector_cache[pivot * dim..(pivot + 1) * dim];
    let mut scored: Vec<(usize, f32)> = neighbours
        .iter()
        .map(|&nb| (nb, dot_i8(pivot_vec, &vector_cache[nb * dim..(nb + 1) * dim]) as f32))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    *neighbours = heuristic_select(&scored, vector_cache, dim, max_conn);
}

// ---------------------------------------------------------
// ENGINE
// ---------------------------------------------------------

#[pymethods]
impl AIGVSEngine {
    #[new]
    fn new(embedding_dim: usize) -> Self {
        AIGVSEngine {
            embedding_dim,
            nodes: Vec::new(),
            node_id_to_index: HashMap::new(),
            root_index: None,
            routers: Vec::new(),
            graphs: Vec::new(),
        }
    }

    // ------------------------------------------------------------------
    // Tree construction
    // ------------------------------------------------------------------

    #[pyo3(signature = (node_id, parent_id=None, is_root=false))]
    fn register_branch(&mut self, node_id: String, parent_id: Option<String>, is_root: bool) {
        let node_idx = self.nodes.len();
        let router_idx = self.routers.len();
        self.routers.push(RustHopfieldRouterInternal { id_lookup: Vec::new(), x_matrix: Vec::new() });
        self.nodes.push(ExecutionNode::Branch { router_idx, children_mapping: HashMap::new() });
        self.node_id_to_index.insert(node_id.clone(), node_idx);
        if is_root { self.root_index = Some(node_idx); }
        if let Some(pid) = parent_id {
            if let Some(&p_idx) = self.node_id_to_index.get(&pid) {
                if let ExecutionNode::Branch { ref mut children_mapping, .. } = self.nodes[p_idx] {
                    children_mapping.insert(node_id, node_idx);
                }
            }
        }
    }

    #[pyo3(signature = (node_id, parent_id, max_connections=16, ef_construction=100))]
    fn register_leaf(&mut self, node_id: String, parent_id: String, max_connections: usize, ef_construction: usize) {
        let node_idx = self.nodes.len();
        let graph_idx = self.graphs.len();
        self.graphs.push(RustNSWGraphInternal {
            chunk_ids: Vec::new(),
            vector_cache: Vec::new(),
            local_graph: Vec::new(),
            deleted_indices: HashSet::new(),
            entry_point_idx: None,
            max_connections,
            ef_construction,
        });
        self.nodes.push(ExecutionNode::Leaf { graph_idx });
        self.node_id_to_index.insert(node_id.clone(), node_idx);
        if let Some(&p_idx) = self.node_id_to_index.get(&parent_id) {
            if let ExecutionNode::Branch { ref mut children_mapping, .. } = self.nodes[p_idx] {
                children_mapping.insert(node_id, node_idx);
            }
        }
    }

    // ------------------------------------------------------------------
    // Router mutations
    // ------------------------------------------------------------------

    fn update_attractor(&mut self, router_id: String, child_id: String, centroid: PyReadonlyArray1<f32>) {
        let r_node_idx = match self.node_id_to_index.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx] {
            ExecutionNode::Branch { router_idx, .. } => *router_idx, _ => return,
        };
        let vec: Vec<f32> = centroid.as_array().iter().copied().collect();
        let dim = self.embedding_dim;
        let router = &mut self.routers[router_idx];
        if let Some(pos) = router.id_lookup.iter().position(|x| *x == child_id) {
            let start = pos * dim;
            router.x_matrix[start..start + dim].copy_from_slice(&vec);
        } else {
            router.id_lookup.push(child_id);
            router.x_matrix.extend_from_slice(&vec);
        }
    }

    fn remove_attractor(&mut self, router_id: String, child_id: String) {
        let r_node_idx = match self.node_id_to_index.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx] {
            ExecutionNode::Branch { router_idx, .. } => *router_idx, _ => return,
        };
        if let ExecutionNode::Branch { ref mut children_mapping, .. } = self.nodes[r_node_idx] {
            children_mapping.remove(&child_id);
        }
        let router = &mut self.routers[router_idx];
        if let Some(pos) = router.id_lookup.iter().position(|x| *x == child_id) {
            router.id_lookup.remove(pos);
            let dim = self.embedding_dim;
            let start = pos * dim;
            router.x_matrix.drain(start..start + dim);
        }
    }

    // ------------------------------------------------------------------
    // Graph mutations — real NSW insert
    // ------------------------------------------------------------------

    fn native_insert_to_leaf(&mut self, leaf_id: String, chunk_id: String, vector: PyReadonlyArray1<f32>) {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return };
        let graph_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { graph_idx } => *graph_idx, _ => return };

        let graph = &mut self.graphs[graph_idx];
        let vec_data_f32: Vec<f32> = vector.as_array().iter().copied().collect();
        let vec_data: Vec<i8> = vec_data_f32.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
        let new_idx = graph.chunk_ids.len();
        let dim = self.embedding_dim;
        let m = graph.max_connections;
        let ef = graph.ef_construction;

        graph.chunk_ids.push(chunk_id);
        graph.vector_cache.extend_from_slice(&vec_data);
        graph.local_graph.push(RwLock::new(Vec::with_capacity(m)));

        if new_idx == 0 {
            graph.entry_point_idx = Some(0);
            return;
        }

        let ep = graph.entry_point_idx.unwrap();

        // Phase 1: find ef_construction best candidates via greedy graph search.
        let mut visited = vec![false; graph.local_graph.len()];
        let candidates = nsw_search_candidates(
            &graph.vector_cache, &graph.local_graph, &graph.deleted_indices,
            &vec_data, dim, ep, ef, &mut visited
        );

        // Phase 2: select best M neighbours from candidates.
        let mut cand_vec: Vec<(usize, f32)> = candidates.into_sorted_vec().into_iter().rev().map(|(Score(s), idx)| (idx, s)).collect();
        let neighbours = heuristic_select(&cand_vec, &graph.vector_cache, dim, m);

        // Phase 3: bidirectional wiring with shrinkage.
        for nb_idx in neighbours {
            graph.local_graph[new_idx].write().push(nb_idx);
            let mut nb_lock = graph.local_graph[nb_idx].write();
            nb_lock.push(new_idx);
            if nb_lock.len() > m {
                let mut pruned = nb_lock.clone();
                prune_neighbours(&mut pruned, &graph.vector_cache, nb_idx, dim, m);
                *nb_lock = pruned;
            }
        }

        // Update entry point to the best-connected node (densest region = best entry).
        if graph.local_graph[new_idx].read().len() > graph.local_graph[ep].read().len() {
            graph.entry_point_idx = Some(new_idx);
        }
    }

    #[pyo3(signature = (leaf_id, chunk_ids, vectors))]
    fn native_batch_insert_to_leaf(&mut self, leaf_id: String, chunk_ids: Vec<String>, vectors: PyReadonlyArray2<f32>) {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return };
        let graph_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { graph_idx } => *graph_idx, _ => return };

        let graph = &mut self.graphs[graph_idx];
        let dim = self.embedding_dim;
        let m = graph.max_connections;
        let ef = graph.ef_construction;

        let vectors_arr = vectors.as_array();
        let n_new = chunk_ids.len();
        if n_new == 0 { return; }

        let start_idx = graph.chunk_ids.len();

        graph.chunk_ids.extend(chunk_ids.into_iter());
        if let Some(slc) = vectors_arr.as_slice() {
            let scaled: Vec<i8> = slc.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
            graph.vector_cache.extend_from_slice(&scaled);
        } else {
            for i in 0..n_new {
                let row = vectors_arr.row(i);
                let scaled: Vec<i8> = row.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
                graph.vector_cache.extend(scaled);
            }
        }
        for _ in 0..n_new {
            graph.local_graph.push(RwLock::new(Vec::with_capacity(m)));
        }

        let mut parallel_start = 0;
        if start_idx == 0 {
            graph.entry_point_idx = Some(0);
            parallel_start = 1;
        }

        let total_nodes = graph.local_graph.len();
        let vector_cache = &graph.vector_cache;
        let local_graph = &graph.local_graph;
        let deleted_indices = &graph.deleted_indices;
        let ep = graph.entry_point_idx.unwrap();

        let indices: Vec<usize> = (start_idx + parallel_start .. start_idx + n_new).collect();
        indices.into_par_iter().for_each(|new_idx| {
            let mut visited = vec![false; total_nodes];
            let vec_data = &vector_cache[new_idx * dim .. (new_idx + 1) * dim];

            let candidates = nsw_search_candidates(
                vector_cache, local_graph, deleted_indices,
                vec_data, dim, ep, ef, &mut visited
            );

            let mut cand_vec: Vec<(usize, f32)> = candidates.into_sorted_vec().into_iter().rev().map(|(Score(s), idx)| (idx, s)).collect();
            let neighbours = heuristic_select(&cand_vec, vector_cache, dim, m);

            for nb_idx in neighbours {
                local_graph[new_idx].write().push(nb_idx);
                let mut nb_lock = local_graph[nb_idx].write();
                nb_lock.push(new_idx);
                if nb_lock.len() > m {
                    let mut pruned = nb_lock.clone();
                    prune_neighbours(&mut pruned, vector_cache, nb_idx, dim, m);
                    *nb_lock = pruned;
                }
            }
        });

        let mut best_ep = graph.entry_point_idx.unwrap();
        let mut best_deg = graph.local_graph[best_ep].read().len();
        for i in start_idx .. start_idx + n_new {
            let deg = graph.local_graph[i].read().len();
            if deg > best_deg {
                best_deg = deg;
                best_ep = i;
            }
        }
        graph.entry_point_idx = Some(best_ep);
    }

    fn native_mark_deleted(&mut self, leaf_id: String, chunk_id: String) {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return };
        let graph_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { graph_idx } => *graph_idx, _ => return };
        let graph = &mut self.graphs[graph_idx];
        if let Some(pos) = graph.chunk_ids.iter().position(|x| *x == chunk_id) {
            graph.deleted_indices.insert(pos);
        }
    }

    // ------------------------------------------------------------------
    // Search — parallel leaf traversal
    //
    // Architecture:
    //   1. Single-threaded DFS/BFS to collect which leaf graph_idxs to search
    //      (routing is fast sequential; no Rayon overhead needed here).
    //   2. Rayon parallel_iter over the collected leaf indices — each leaf NSW
    //      search is independent, so this is perfectly data-parallel.
    //   3. Global rerank of all cross-leaf candidates, dedup, top-k.
    //
    // This eliminates the sequential leaf bottleneck while keeping Rayon cost
    // confined to work units that are large enough to justify it (a full NSW
    // traversal over hundreds of vectors per leaf).
    // ------------------------------------------------------------------

    #[pyo3(signature = (query_vector, top_k, max_beam=5, target_confidence=0.90, beta=5.0, ef_search=100))]
    fn unified_search(
        &self,
        py: Python,
        query_vector: PyReadonlyArray1<f32>,
        top_k: usize,
        max_beam: usize,
        target_confidence: f32,
        beta: f32,
        ef_search: usize,
    ) -> Vec<String> {
        let root = match self.root_index { Some(r) => r, None => return vec![] };
        let ef = ef_search.max(top_k);

        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let q_i8: Vec<i8> = q.iter().map(|&x| (x * 127.0).clamp(-127.0, 127.0) as i8).collect();
        let dim = self.embedding_dim;
        let nodes  = &self.nodes;
        let routers = &self.routers;
        let graphs  = &self.graphs;

        py.allow_threads(|| {
            // --- Phase 1: sequential routing to collect leaf graph indices ---
            // Routing is O(beam × children) per level with tiny N — no Rayon needed.
            let mut stack: Vec<usize> = vec![root];
            let mut leaf_graph_indices: Vec<usize> = Vec::new();

            while let Some(curr_idx) = stack.pop() {
                match &nodes[curr_idx] {
                    ExecutionNode::Branch { router_idx, children_mapping } => {
                        let router = &routers[*router_idx];
                        let selected = router_beam_search(
                            &router.x_matrix, &router.id_lookup, children_mapping,
                            &q, dim, max_beam, target_confidence, beta, f32::NEG_INFINITY,
                        );
                        stack.extend(selected);
                    }
                    ExecutionNode::Leaf { graph_idx } => {
                        leaf_graph_indices.push(*graph_idx);
                    }
                }
            }

            LEAVES_VISITED.fetch_add(leaf_graph_indices.len() as u64, std::sync::atomic::Ordering::Relaxed);

            // --- Phase 2: parallel NSW search across all selected leaves ---
            // Each leaf search is independent — perfect for Rayon.
            // Lock-free: each task owns its own result Vec, merged after.
            let t_child = std::time::Instant::now();
            let all_candidates: Vec<(String, f32)> = leaf_graph_indices
                .par_iter()
                .flat_map(|&gidx| {
                    nsw_query(&graphs[gidx], &q_i8, dim, ef)
                })
                .collect();
            CHILD_QUERY_TOTAL_US.fetch_add(t_child.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);

            // --- Phase 3: global rerank, dedup, top-k ---
            let t_rerank = std::time::Instant::now();
            let mut sorted = all_candidates;
            sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            let mut seen: HashSet<String> = HashSet::new();
            let top_results: Vec<String> = sorted.into_iter()
                .filter(|(id, _)| seen.insert(id.clone()))
                .take(top_k)
                .map(|(id, _)| id)
                .collect();
            RERANK_TOTAL_US.fetch_add(t_rerank.elapsed().as_micros() as u64, std::sync::atomic::Ordering::Relaxed);
            QUERY_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            top_results
        })
    }

    // ------------------------------------------------------------------
    // Scoped routing — used during insertion to find the best child of one
    // specific branch without traversing from root.
    // Sequential scan: insert-time routing calls happen one at a time and
    // the clone overhead of py.allow_threads is not worth it at this scale.
    // ------------------------------------------------------------------

    #[pyo3(signature = (node_id, query_vector, max_beam=1, target_confidence=1.0, beta=20.0))]
    fn route_query(
        &self,
        _py: Python,
        node_id: String,
        query_vector: PyReadonlyArray1<f32>,
        max_beam: usize,
        target_confidence: f32,
        beta: f32,
    ) -> Vec<String> {
        let node_idx = match self.node_id_to_index.get(&node_id) { Some(&i) => i, None => return vec![] };
        let (router_idx, children_mapping) = match &self.nodes[node_idx] {
            ExecutionNode::Branch { router_idx, children_mapping } => (*router_idx, children_mapping),
            ExecutionNode::Leaf { .. } => return vec![],
        };
        let router = &self.routers[router_idx];
        if router.id_lookup.is_empty() { return vec![]; }

        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let dim = self.embedding_dim;
        let n = router.id_lookup.len();

        // Sequential scan — fast at small N, avoids Rayon overhead.
        let mut raw_scores: Vec<f32> = Vec::with_capacity(n);
        let mut max_s = f32::NEG_INFINITY;
        for row in router.x_matrix.chunks_exact(dim) {
            let s = dot(&q, row);
            if s > max_s { max_s = s; }
            raw_scores.push(s);
        }

        let mut exp_sum = 0.0f32;
        let mut probs: Vec<(usize, f32)> = raw_scores
            .into_iter()
            .enumerate()
            .map(|(i, s)| { let e = ((s - max_s) * beta).exp(); exp_sum += e; (i, e) })
            .collect();
        for (_, p) in probs.iter_mut() { *p /= exp_sum + 1e-9; }
        probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        let mut result: Vec<String> = Vec::new();
        let mut cumulative = 0.0f32;
        for (idx, prob) in probs.into_iter().take(max_beam) {
            let child_str = &router.id_lookup[idx];
            if children_mapping.contains_key(child_str) {
                result.push(child_str.clone());
            }
            cumulative += prob;
            if cumulative >= target_confidence { break; }
        }
        result
    }

    #[pyo3(signature = ())]
    fn print_profile_stats(&self) {
        let count = QUERY_COUNT.load(std::sync::atomic::Ordering::Relaxed).max(1);
        println!("\n===== AIGVS RUST PROFILE =====");
        println!("Queries: {}", count);
        println!("Router Total:     {:>8.2} us/query", ROUTER_TOTAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("  - Score:        {:>8.2} us/query", ROUTER_SCORE_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("  - Softmax:      {:>8.2} us/query", ROUTER_SOFTMAX_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("  - Sort/Select:  {:>8.2} us/query", ROUTER_SORT_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Child Query Time: {:>8.2} us/query", CHILD_QUERY_TOTAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Rerank Time:      {:>8.2} us/query", RERANK_TOTAL_US.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Candidate Vecs:   {:>8.2} vectors/query", CANDIDATE_VECTORS.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("Leaves Visited:   {:>8.2} leaves/query", LEAVES_VISITED.load(std::sync::atomic::Ordering::Relaxed) as f64 / count as f64);
        println!("==============================\n");
    }
}

// ---------------------------------------------------------
// MODULE
// ---------------------------------------------------------

#[pymodule]
fn aigvs_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AIGVSEngine>()?;
    Ok(())
}