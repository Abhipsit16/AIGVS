use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet, HashMap};
use std::cmp::Ordering;
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
    vector_cache: Vec<f32>,
    local_graph: Vec<Vec<usize>>,
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
) -> Vec<usize> {
    if id_lookup.is_empty() { return vec![]; }

    let n = id_lookup.len();

    // Score: tight sequential dot-product scan, fully in L1/L2 cache.
    let mut raw_scores: Vec<f32> = Vec::with_capacity(n);
    let mut max_s = f32::NEG_INFINITY;
    for row in x_matrix.chunks_exact(dim) {
        let s = dot(q, row);
        if s > max_s { max_s = s; }
        raw_scores.push(s);
    }

    // Numerically stable softmax with temperature beta.
    let mut exp_sum = 0.0f32;
    let mut probs: Vec<(usize, f32)> = raw_scores
        .into_iter()
        .enumerate()
        .map(|(i, s)| {
            let e = ((s - max_s) * beta).exp();
            exp_sum += e;
            (i, e)
        })
        .collect();
    for (_, p) in probs.iter_mut() { *p /= exp_sum + 1e-9; }

    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let mut selected = Vec::with_capacity(max_beam.min(n));
    let mut cumulative = 0.0f32;
    for (idx, prob) in probs.into_iter().take(max_beam) {
        if let Some(&child_node_idx) = children_mapping.get(&id_lookup[idx]) {
            selected.push(child_node_idx);
        }
        cumulative += prob;
        if cumulative >= target_confidence { break; }
    }
    selected
}

// ---------------------------------------------------------
// NSW SEARCH CANDIDATES
// Standard HNSW Algorithm 2 (search layer).
// Two-heap design: max-heap for frontier expansion, min-heap for result set.
// Early exit when the best unseen candidate is worse than the worst result.
// ---------------------------------------------------------

fn nsw_search_candidates(
    vector_cache: &[f32],
    local_graph: &[Vec<usize>],
    deleted_indices: &HashSet<usize>,
    q: &[f32],
    dim: usize,
    entry: usize,
    ef: usize,
) -> BinaryHeap<(Score, usize)> {
    let mut candidates: BinaryHeap<(Score, usize)> = BinaryHeap::new();
    let mut results:    BinaryHeap<(MinScore, usize)> = BinaryHeap::new();
    let mut visited:    HashSet<usize> = HashSet::new();

    let entry_score = dot(q, &vector_cache[entry * dim..(entry + 1) * dim]);
    candidates.push((Score(entry_score), entry));
    if !deleted_indices.contains(&entry) {
        results.push((MinScore(entry_score), entry));
    }
    visited.insert(entry);

    while let Some((Score(cand_score), curr)) = candidates.pop() {
        if results.len() >= ef {
            if let Some(&(MinScore(worst), _)) = results.peek() {
                if cand_score < worst { break; }
            }
        }
        for &nb in &local_graph[curr] {
            if visited.insert(nb) {
                let nb_score = dot(q, &vector_cache[nb * dim..(nb + 1) * dim]);
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
    q: &[f32],
    dim: usize,
    ef: usize,
) -> Vec<(String, f32)> {
    let ep = match graph.entry_point_idx { Some(e) => e, None => return vec![] };
    let heap = nsw_search_candidates(
        &graph.vector_cache, &graph.local_graph, &graph.deleted_indices, q, dim, ep, ef,
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

fn select_neighbours(candidates: BinaryHeap<(Score, usize)>, m: usize) -> Vec<(usize, f32)> {
    let mut v: Vec<(Score, usize)> = candidates.into_sorted_vec();
    v.reverse();
    v.truncate(m);
    v.into_iter().map(|(Score(s), idx)| (idx, s)).collect()
}

fn prune_neighbours(
    neighbours: &mut Vec<usize>,
    vector_cache: &[f32],
    pivot: usize,
    dim: usize,
    max_conn: usize,
) {
    if neighbours.len() <= max_conn { return; }
    let pivot_vec = &vector_cache[pivot * dim..(pivot + 1) * dim];
    let mut scored: Vec<(usize, f32)> = neighbours
        .iter()
        .map(|&nb| (nb, dot(pivot_vec, &vector_cache[nb * dim..(nb + 1) * dim])))
        .collect();
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(max_conn);
    *neighbours = scored.into_iter().map(|(nb, _)| nb).collect();
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
        let vec_data: Vec<f32> = vector.as_array().iter().copied().collect();
        let new_idx = graph.chunk_ids.len();
        let dim = self.embedding_dim;
        let m = graph.max_connections;
        let ef = graph.ef_construction;

        graph.chunk_ids.push(chunk_id);
        graph.vector_cache.extend_from_slice(&vec_data);
        graph.local_graph.push(Vec::with_capacity(m));

        if new_idx == 0 {
            graph.entry_point_idx = Some(0);
            return;
        }

        let ep = graph.entry_point_idx.unwrap();

        // Phase 1: find ef_construction best candidates via greedy graph search.
        let candidates = nsw_search_candidates(
            &graph.vector_cache, &graph.local_graph, &graph.deleted_indices,
            &vec_data, dim, ep, ef,
        );

        // Phase 2: select best M neighbours from candidates.
        let neighbours = select_neighbours(candidates, m);

        // Phase 3: bidirectional wiring with shrinkage.
        for (nb_idx, _) in neighbours {
            graph.local_graph[new_idx].push(nb_idx);
            graph.local_graph[nb_idx].push(new_idx);
            if graph.local_graph[nb_idx].len() > m {
                let mut pruned = graph.local_graph[nb_idx].clone();
                prune_neighbours(&mut pruned, &graph.vector_cache, nb_idx, dim, m);
                graph.local_graph[nb_idx] = pruned;
            }
        }

        // Update entry point to the best-connected node (densest region = best entry).
        if graph.local_graph[new_idx].len() > graph.local_graph[ep].len() {
            graph.entry_point_idx = Some(new_idx);
        }
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

    #[pyo3(signature = (query_vector, top_k, max_beam=30, target_confidence=0.99, beta=5.0, ef_search=50))]
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
                            &q, dim, max_beam, target_confidence, beta,
                        );
                        stack.extend(selected);
                    }
                    ExecutionNode::Leaf { graph_idx } => {
                        leaf_graph_indices.push(*graph_idx);
                    }
                }
            }

            // --- Phase 2: parallel NSW search across all selected leaves ---
            // Each leaf search is independent — perfect for Rayon.
            // Lock-free: each task owns its own result Vec, merged after.
            let all_candidates: Vec<(String, f32)> = leaf_graph_indices
                .par_iter()
                .flat_map(|&gidx| {
                    nsw_query(&graphs[gidx], &q, dim, ef)
                })
                .collect();

            // --- Phase 3: global rerank, dedup, top-k ---
            let mut sorted = all_candidates;
            sorted.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

            let mut seen: HashSet<String> = HashSet::new();
            sorted.into_iter()
                .filter(|(id, _)| seen.insert(id.clone()))
                .take(top_k)
                .map(|(id, _)| id)
                .collect()
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
}

// ---------------------------------------------------------
// MODULE
// ---------------------------------------------------------

#[pymodule]
fn aigvs_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AIGVSEngine>()?;
    Ok(())
}