use std::time::Instant;
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashSet, HashMap};
use std::cmp::Ordering;
use rand::Rng;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
    Branch { router_idx: usize, children_mapping: HashMap<String, usize> },
    Leaf { graph_idx: usize },
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
    x_matrix: Vec<f32>, // Flattened row-major
}

struct RustNSWGraphInternal {
    chunk_ids: Vec<String>,
    vector_cache: Vec<f32>,
    edge_buffer: Vec<u32>,   // Size = N * max_connections
    edge_counts: Vec<u32>,   // Size = N
    deleted_indices: HashSet<usize>,
    entry_point_idx: Option<usize>,
    max_connections: usize,
    ef_construction: usize,
}

// ---------------------------------------------------------
// EXPLICIT SIMD (AVX2 / FMA)
// ---------------------------------------------------------

#[inline(always)]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_avx2(a, b) };
        }
    }
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b.iter()) { s += x * y; }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum256 = _mm256_setzero_ps();
    let mut i = 0;
    let len = a.len();
    let unrolled = len - (len % 8);

    while i < unrolled {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        sum256 = _mm256_fmadd_ps(va, vb, sum256);
        i += 8;
    }

    let mut align_array = [0.0f32; 8];
    _mm256_storeu_ps(align_array.as_mut_ptr(), sum256);
    let mut sum = align_array.iter().sum();

    while i < len {
        sum += *a.get_unchecked(i) * *b.get_unchecked(i);
        i += 1;
    }
    sum
}

// ---------------------------------------------------------
// GRAPH HELPERS (Flat Edge Buffer)
// ---------------------------------------------------------

#[inline(always)]
fn get_edges<'a>(graph: &'a RustNSWGraphInternal, node_idx: usize) -> &'a [u32] {
    let start = node_idx * graph.max_connections;
    let count = graph.edge_counts[node_idx] as usize;
    &graph.edge_buffer[start..start + count]
}

#[inline(always)]
fn set_edges(graph: &mut RustNSWGraphInternal, node_idx: usize, new_edges: &[u32]) {
    let start = node_idx * graph.max_connections;
    let count = new_edges.len();
    graph.edge_buffer[start..start + count].copy_from_slice(new_edges);
    graph.edge_counts[node_idx] = count as u32;
}

// ---------------------------------------------------------
// NSW SEARCH CANDIDATES
// ---------------------------------------------------------

fn nsw_search_candidates(
    graph: &RustNSWGraphInternal,
    q: &[f32],
    dim: usize,
    entry: usize,
    ef: usize,
) -> BinaryHeap<(Score, usize)> {
    let mut candidates: BinaryHeap<(Score, usize)> = BinaryHeap::with_capacity(ef * 2);
    let mut results:    BinaryHeap<(MinScore, usize)> = BinaryHeap::with_capacity(ef * 2);
    let mut visited:    HashSet<usize> = HashSet::with_capacity(ef * 4);

    let entry_score = dot(q, &graph.vector_cache[entry * dim..(entry + 1) * dim]);
    candidates.push((Score(entry_score), entry));
    if !graph.deleted_indices.contains(&entry) {
        results.push((MinScore(entry_score), entry));
    }
    visited.insert(entry);

    while let Some((Score(cand_score), curr)) = candidates.pop() {
        if results.len() >= ef {
            if let Some(&(MinScore(worst), _)) = results.peek() {
                if cand_score < worst { break; }
            }
        }
        
        let edges = get_edges(graph, curr);
        for &nb_u32 in edges {
            let nb = nb_u32 as usize;
            if visited.insert(nb) {
                let nb_score = dot(q, &graph.vector_cache[nb * dim..(nb + 1) * dim]);
                candidates.push((Score(nb_score), nb));
                
                if !graph.deleted_indices.contains(&nb) {
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

    results.into_iter().map(|(MinScore(s), idx)| (Score(s), idx)).collect()
}

fn nsw_query(
    graph: &RustNSWGraphInternal,
    q: &[f32],
    dim: usize,
    ef: usize,
) -> Vec<(String, f32)> {
    let n_items = graph.chunk_ids.len();
    if n_items == 0 { return vec![]; }

    if n_items < 250 {
        let mut results = Vec::with_capacity(n_items);
        for (i, row) in graph.vector_cache.chunks_exact(dim).enumerate() {
            if !graph.deleted_indices.contains(&i) {
                results.push((Score(dot(q, row)), i));
            }
        }
        results.sort_unstable_by(|a, b| b.cmp(a));
        return results.into_iter().take(ef)
            .map(|(Score(s), idx)| (graph.chunk_ids[idx].clone(), s))
            .collect();
    }

    let ep = match graph.entry_point_idx { Some(e) => e, None => return vec![] };
    let heap = nsw_search_candidates(graph, q, dim, ep, ef);
    let mut v: Vec<(Score, usize)> = heap.into_sorted_vec();
    v.reverse();
    v.into_iter().map(|(Score(s), idx)| (graph.chunk_ids[idx].clone(), s)).collect()
}

// ---------------------------------------------------------
// INSERT HELPERS
// ---------------------------------------------------------

fn prune_neighbours(
    neighbours: &mut Vec<u32>,
    vector_cache: &[f32],
    pivot: usize,
    dim: usize,
    max_conn: usize,
) {
    if neighbours.len() <= max_conn { return; }
    let pivot_vec = &vector_cache[pivot * dim..(pivot + 1) * dim];
    let mut scored: Vec<(u32, f32)> = neighbours.iter().map(|&nb| {
        let nb_idx = nb as usize;
        (nb, dot(pivot_vec, &vector_cache[nb_idx * dim..(nb_idx + 1) * dim]))
    }).collect();
    
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    scored.truncate(max_conn);
    *neighbours = scored.into_iter().map(|(nb, _)| nb).collect();
}

// ---------------------------------------------------------
// ENGINE IMPLEMENTATION
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

    #[pyo3(signature = (node_id, parent_id, max_connections=32, ef_construction=200))]
    fn register_leaf(&mut self, node_id: String, parent_id: String, max_connections: usize, ef_construction: usize) {
        let node_idx = self.nodes.len();
        let graph_idx = self.graphs.len();
        self.graphs.push(RustNSWGraphInternal {
            chunk_ids: Vec::new(),
            vector_cache: Vec::new(),
            edge_buffer: Vec::new(),
            edge_counts: Vec::new(),
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

    fn update_attractor(&mut self, router_id: String, child_id: String, centroid: PyReadonlyArray1<f32>) {
        let r_node_idx = match self.node_id_to_index.get(&router_id) { Some(&i) => i, None => return };
        let router_idx = match &self.nodes[r_node_idx] { ExecutionNode::Branch { router_idx, .. } => *router_idx, _ => return };
        
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
        let router_idx = match &self.nodes[r_node_idx] { ExecutionNode::Branch { router_idx, .. } => *router_idx, _ => return };
        
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
        
        graph.edge_buffer.resize(graph.edge_buffer.len() + m, 0);
        graph.edge_counts.push(0);

        if new_idx == 0 {
            graph.entry_point_idx = Some(0);
            return;
        }

        let ep = graph.entry_point_idx.unwrap();
        let candidates = nsw_search_candidates(graph, &vec_data, dim, ep, ef);

        let mut v: Vec<(Score, usize)> = candidates.into_sorted_vec();
        v.reverse();
        v.truncate(m);
        let neighbours: Vec<u32> = v.into_iter().map(|(_, idx)| idx as u32).collect();

        set_edges(graph, new_idx, &neighbours);
        for &nb_u32 in &neighbours {
            let nb_idx = nb_u32 as usize;
            let mut nb_edges = get_edges(graph, nb_idx).to_vec();
            nb_edges.push(new_idx as u32);
            
            if nb_edges.len() > m {
                prune_neighbours(&mut nb_edges, &graph.vector_cache, nb_idx, dim, m);
            }
            set_edges(graph, nb_idx, &nb_edges);
        }

        if neighbours.len() > get_edges(graph, ep).len() {
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

    #[pyo3(signature = (leaf_id, max_clusters=4, max_iters=15))]
    fn compute_disjoint_clusters(
        &self,
        leaf_id: String,
        max_clusters: usize,
        max_iters: usize,
    ) -> PyResult<Option<(Vec<Vec<f32>>, Vec<usize>)>> {
        let l_idx = match self.node_id_to_index.get(&leaf_id) { Some(&i) => i, None => return Ok(None) };
        let graph_idx = match &self.nodes[l_idx] { ExecutionNode::Leaf { graph_idx } => *graph_idx, _ => return Ok(None) };
        let graph = &self.graphs[graph_idx];
        let dim = self.embedding_dim;
        
        let mut active_indices = Vec::new();
        let mut vectors = Vec::new();
        for (i, row) in graph.vector_cache.chunks_exact(dim).enumerate() {
            if !graph.deleted_indices.contains(&i) {
                active_indices.push(i);
                vectors.extend_from_slice(row);
            }
        }

        let n = active_indices.len();
        if n < 2 { return Ok(None); }
        let k = max_clusters.min(n);

        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k * dim);
        let first_idx = rng.gen_range(0..n);
        centroids.extend_from_slice(&vectors[first_idx * dim..(first_idx + 1) * dim]);

        let mut min_dists = vec![2.0f32; n];
        for _ in 1..k {
            let current_centroid = &centroids[(centroids.len() - dim)..];
            let mut sum_dist = 0.0;
            for i in 0..n {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let dist = 1.0 - dot(vec, current_centroid);
                if dist < min_dists[i] { min_dists[i] = dist; }
                sum_dist += min_dists[i] * min_dists[i];
            }

            let target = rng.gen::<f32>() * sum_dist;
            let mut cumulative = 0.0;
            let mut next_idx = 0;
            for i in 0..n {
                cumulative += min_dists[i] * min_dists[i];
                if cumulative >= target {
                    next_idx = i;
                    break;
                }
            }
            centroids.extend_from_slice(&vectors[next_idx * dim..(next_idx + 1) * dim]);
        }

        let mut assignments = vec![0usize; n];
        for _ in 0..max_iters {
            let mut changed = false;
            let mut new_centroids = vec![0.0f32; k * dim];
            let mut counts = vec![0; k];

            for i in 0..n {
                let vec = &vectors[i * dim..(i + 1) * dim];
                let mut best_c = 0;
                let mut best_sim = f32::NEG_INFINITY;

                for c in 0..k {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let sim = dot(vec, cent);
                    if sim > best_sim {
                        best_sim = sim;
                        best_c = c;
                    }
                }

                if assignments[i] != best_c {
                    changed = true;
                    assignments[i] = best_c;
                }

                counts[best_c] += 1;
                for d in 0..dim { new_centroids[best_c * dim + d] += vec[d]; }
            }

            for c in 0..k {
                if counts[c] > 0 {
                    let start = c * dim;
                    let end = start + dim;
                    let mut sq_norm = 0.0;
                    for d in start..end { sq_norm += new_centroids[d] * new_centroids[d]; }
                    let norm = sq_norm.sqrt().max(1e-9);
                    for d in start..end { centroids[d] = new_centroids[d] / norm; }
                }
            }
            if !changed { break; }
        }

        let out_centroids: Vec<Vec<f32>> = centroids.chunks_exact(dim).map(|c| c.to_vec()).collect();
        Ok(Some((out_centroids, assignments)))
    }

    // SOTA FIX: Variables align precisely with PyO3 signature macro
    #[pyo3(signature = (query_vector, top_k, max_beam=30, target_confidence=0.99, beta=5.0, ef_search=50))]
    #[allow(unused_variables)]
    fn unified_search(
        &self, py: Python, query_vector: PyReadonlyArray1<f32>, top_k: usize, max_beam: usize,
        target_confidence: f32, beta: f32, ef_search: usize,
    ) -> (Vec<String>, HashMap<String, f64>) {
        let root = match self.root_index { Some(r) => r, None => return (vec![], HashMap::new()) };
        let ef = ef_search.max(top_k);

        let q: Vec<f32> = query_vector.as_array().iter().copied().collect();
        let dim = self.embedding_dim;
        let nodes = &self.nodes;
        let routers = &self.routers;
        let graphs = &self.graphs;

        py.allow_threads(|| {
            let mut telemetry = HashMap::new();
            let total_start = Instant::now();
            let route_start = Instant::now();

            let mut global_frontier: BinaryHeap<(Score, usize)> = BinaryHeap::new();
            global_frontier.push((Score(f32::INFINITY), root)); 
            
            let mut leaf_graph_indices = Vec::new();
            while let Some((_, curr_idx)) = global_frontier.pop() {
                if leaf_graph_indices.len() >= max_beam { break; }

                match &nodes[curr_idx] {
                    ExecutionNode::Branch { router_idx, children_mapping } => {
                        let router = &routers[*router_idx];
                        for (i, row) in router.x_matrix.chunks_exact(dim).enumerate() {
                            let child_id = &router.id_lookup[i];
                            if let Some(&child_node_idx) = children_mapping.get(child_id) {
                                global_frontier.push((Score(dot(&q, row)), child_node_idx));
                            }
                        }
                    }
                    ExecutionNode::Leaf { graph_idx } => { leaf_graph_indices.push(*graph_idx); }
                }
            }

            telemetry.insert("rust_01_routing_ms".to_string(), route_start.elapsed().as_secs_f64() * 1000.0);

            let leaf_start = Instant::now();
            let all_candidates: Vec<(String, f32)> = leaf_graph_indices
                .par_iter()
                .flat_map(|&gidx| nsw_query(&graphs[gidx], &q, dim, ef))
                .collect();
            telemetry.insert("rust_02_leaf_nsw_ms".to_string(), leaf_start.elapsed().as_secs_f64() * 1000.0);

            let rerank_start = Instant::now();
            let mut top_k_heap: BinaryHeap<(MinScore, String)> = BinaryHeap::with_capacity(top_k + 1);
            for (id, score) in all_candidates {
                top_k_heap.push((MinScore(score), id));
                if top_k_heap.len() > top_k { top_k_heap.pop(); }
            }

            let mut final_results: Vec<String> = top_k_heap.into_sorted_vec().into_iter().map(|(_, id)| id).collect();
            final_results.reverse();

            telemetry.insert("rust_03_rerank_ms".to_string(), rerank_start.elapsed().as_secs_f64() * 1000.0);
            telemetry.insert("rust_total_ms".to_string(), total_start.elapsed().as_secs_f64() * 1000.0);

            (final_results, telemetry)
        })
    }

    // SOTA FIX: Variables align precisely with PyO3 signature macro
    #[pyo3(signature = (node_id, query_vector, max_beam=1, target_confidence=1.0, beta=18.0))]
    #[allow(unused_variables)]
    fn route_query(
        &self, py: Python, node_id: String, query_vector: PyReadonlyArray1<f32>, max_beam: usize,
        target_confidence: f32, beta: f32,
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

        let mut top_k_heap: BinaryHeap<(MinScore, usize)> = BinaryHeap::with_capacity(max_beam + 1);
        for (i, row) in router.x_matrix.chunks_exact(dim).enumerate() {
            top_k_heap.push((MinScore(dot(&q, row)), i));
            if top_k_heap.len() > max_beam { top_k_heap.pop(); }
        }

        let mut result = Vec::with_capacity(max_beam);
        for (_, idx) in top_k_heap.into_sorted_vec().into_iter().rev() {
            let child_str = &router.id_lookup[idx];
            if children_mapping.contains_key(child_str) {
                result.push(child_str.clone());
            }
        }
        result
    }
}

#[pymodule]
fn aigvs_rust(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<AIGVSEngine>()?;
    Ok(())
}