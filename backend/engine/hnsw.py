"""Hierarchical Navigable Small World (HNSW) Vector Graph Indexer.

Implements an owned, in-memory, multi-layer skip-graph index for approximate
nearest neighbor (ANN) retrieval over high-dimensional dense embeddings
(e.g., 768-D semantic representations) without external database dependencies.

Algorithmic Invariants:
1. Multi-Layer Hierarchy: Each layer l in {0, ..., max_level} contains an
   induced subgraph where vertices appear with probability decay factor
   P(layer >= l) = exp(-l / m_L).
2. Greedy Multi-Level Routing: Top-down greedy routing identifies closest entry
   points before transitioning to beam search at the query level.
3. Heuristic Edge Pruning: Selects diverse nearest neighbors while pruning
   collinear or triangle-redundant edges to maintain logarithmic diameter O(log N).
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


class DistanceMetric:
    """Distance metrics for high-dimensional vector spaces."""

    @staticmethod
    def cosine(u: np.ndarray, v: np.ndarray) -> float:
        """Cosine distance: 1.0 - (u . v) / (||u|| * ||v||)."""
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0.0 or norm_v == 0.0:
            return 1.0
        dot = float(np.dot(u, v))
        sim = dot / (norm_u * norm_v)
        return max(0.0, 1.0 - sim)

    @staticmethod
    def euclidean(u: np.ndarray, v: np.ndarray) -> float:
        """Euclidean (L2) distance."""
        diff = u - v
        return float(np.linalg.norm(diff))

    @staticmethod
    def centered_cosine(u: np.ndarray, v: np.ndarray, mean_vector: Optional[np.ndarray] = None) -> float:
        """Centered cosine distance adjusting for anisotropic cluster drift."""
        if mean_vector is not None:
            u = u - mean_vector
            v = v - mean_vector
        return DistanceMetric.cosine(u, v)


class HNSWIndex:
    """Hierarchical Navigable Small World Graph Index."""

    def __init__(
        self,
        dim: int = 768,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        metric: str = "cosine",
        center_mean: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> None:
        self.dim = dim
        self.m = m
        self.m0 = 2 * m
        self.ef_construction = max(ef_construction, m)
        self.ef_search = ef_search
        self.metric_name = metric
        self.center_mean = center_mean
        self.m_l = 1.0 / math.log(m)

        self.rng = np.random.default_rng(seed)

        # Storage
        self.vectors: Dict[str, np.ndarray] = {}
        self.entry_point: Optional[str] = None
        self.max_level: int = -1

        # Multi-layer adjacency: layers[level][node_id] = list of neighbor node_ids
        self.layers: List[Dict[str, List[str]]] = []

    def _dist(self, u: np.ndarray, v: np.ndarray) -> float:
        if self.metric_name == "euclidean":
            return DistanceMetric.euclidean(u, v)
        if self.center_mean is not None:
            return DistanceMetric.centered_cosine(u, v, self.center_mean)
        return DistanceMetric.cosine(u, v)

    def _sample_level(self) -> int:
        """Assign random maximum layer according to exponential decay."""
        uniform_sample = self.rng.uniform(1e-9, 1.0)
        return int(-math.log(uniform_sample) * self.m_l)

    def size(self) -> int:
        """Return count of indexed vectors."""
        return len(self.vectors)

    def add_point(self, node_id: str, vector: Sequence[float] | np.ndarray) -> None:
        """Insert vector into the multi-layer HNSW graph."""
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise ValueError(f"Vector dim {vec.shape} does not match index dim {self.dim}")

        self.vectors[node_id] = vec
        target_level = self._sample_level()

        # Ensure layers capacity
        while len(self.layers) <= max(target_level, self.max_level):
            self.layers.append({})

        if self.entry_point is None:
            self.entry_point = node_id
            self.max_level = target_level
            for level in range(target_level + 1):
                self.layers[level][node_id] = []
            return

        current_entry = self.entry_point
        top_level = self.max_level

        # Phase 1: 1-greedy search from top down to target_level + 1
        for level in range(top_level, target_level, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[level].get(current_entry, [])
                best_dist = self._dist(self.vectors[current_entry], vec)
                for neighbor in neighbors:
                    d = self._dist(self.vectors[neighbor], vec)
                    if d < best_dist:
                        best_dist = d
                        current_entry = neighbor
                        changed = True

        # Phase 2: Beam search and bi-directional linking from min(top_level, target_level) down to 0
        search_entries = [current_entry]
        for level in range(min(top_level, target_level), -1, -1):
            # Filter search entries to those that actually exist in this layer
            valid_entries = [ep for ep in search_entries if ep in self.layers[level]]
            if not valid_entries:
                valid_entries = list(self.layers[level].keys())[:1]

            ef = self.ef_construction
            candidates = self._search_layer_candidates(vec, valid_entries, ef, level)
            selected_neighbors = self._select_heuristic(vec, candidates, self.m0 if level == 0 else self.m)

            if node_id not in self.layers[level]:
                self.layers[level][node_id] = []

            for neighbor in selected_neighbors:
                if neighbor not in self.layers[level]:
                    self.layers[level][neighbor] = []
                self.layers[level][node_id].append(neighbor)
                self.layers[level][neighbor].append(node_id)
                self._prune_connections(neighbor, level)

            search_entries = [cand_id for _, cand_id in candidates[:ef]]

        if target_level > self.max_level:
            self.max_level = target_level
            self.entry_point = node_id

    def _search_layer_candidates(
        self,
        query_vec: np.ndarray,
        entry_points: Sequence[str],
        ef: int,
        level: int,
    ) -> List[Tuple[float, str]]:
        """Beam search within a single layer graph."""
        visited: Set[str] = set(entry_points)
        # Min-heap for candidate exploration: (dist, node_id)
        candidates: List[Tuple[float, str]] = []
        # Max-heap for top results: (-dist, node_id)
        results: List[Tuple[float, str]] = []

        for ep in entry_points:
            d = self._dist(self.vectors[ep], query_vec)
            heapq.heappush(candidates, (d, ep))
            heapq.heappush(results, (-d, ep))

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_result_dist = -results[0][0]

            if c_dist > furthest_result_dist:
                break

            neighbors = self.layers[level].get(c_id, [])
            for n_id in neighbors:
                if n_id in visited:
                    continue
                visited.add(n_id)

                d = self._dist(self.vectors[n_id], query_vec)
                furthest_dist = -results[0][0]

                if d < furthest_dist or len(results) < ef:
                    heapq.heappush(candidates, (d, n_id))
                    heapq.heappush(results, (-d, n_id))
                    if len(results) > ef:
                        heapq.heappop(results)

        # Convert from max-heap (-dist, id) to sorted ascending list (dist, id)
        sorted_res = sorted([(-neg_d, nid) for neg_d, nid in results], key=lambda x: x[0])
        return sorted_res

    def _select_heuristic(
        self,
        base_vec: np.ndarray,
        candidates: List[Tuple[float, str]],
        max_edges: int,
    ) -> List[str]:
        """Heuristic edge selection to promote geometric diversity."""
        if len(candidates) <= max_edges:
            return [cid for _, cid in candidates]

        chosen: List[str] = []
        for dist_to_base, cand_id in candidates:
            if len(chosen) >= max_edges:
                break

            cand_vec = self.vectors[cand_id]
            is_diverse = True
            for ch_id in chosen:
                dist_to_ch = self._dist(cand_vec, self.vectors[ch_id])
                # If closer to already chosen neighbor than to query, skip (prune triangle)
                if dist_to_ch < dist_to_base:
                    is_diverse = False
                    break

            if is_diverse:
                chosen.append(cand_id)

        # Fallback to keep at least top neighbors if diversity was overly strict
        if len(chosen) < min(max_edges, len(candidates)):
            for _, cand_id in candidates:
                if cand_id not in chosen:
                    chosen.append(cand_id)
                if len(chosen) >= max_edges:
                    break

        return chosen

    def _prune_connections(self, node_id: str, level: int) -> None:
        """Prune neighbors of node_id if edge budget is exceeded."""
        max_m = self.m0 if level == 0 else self.m
        neighbors = self.layers[level].get(node_id, [])
        if len(neighbors) <= max_m:
            return

        base_vec = self.vectors[node_id]
        scored_neighbors = [(self._dist(base_vec, self.vectors[nid]), nid) for nid in neighbors]
        scored_neighbors.sort(key=lambda x: x[0])

        new_neighbors = self._select_heuristic(base_vec, scored_neighbors, max_m)
        self.layers[level][node_id] = new_neighbors

    def search(
        self,
        query: Sequence[float] | np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """Search top-K nearest neighbors.

        Returns list of (node_id, similarity) sorted descending by similarity.
        """
        if not self.vectors or self.entry_point is None:
            return []

        q_vec = np.asarray(query, dtype=np.float32)
        if q_vec.shape != (self.dim,):
            raise ValueError(f"Query dim {q_vec.shape} does not match index dim {self.dim}")

        ef = ef_search or max(self.ef_search, k)
        current_entry = self.entry_point

        # Phase 1: 1-greedy search down to layer 1
        for level in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                neighbors = self.layers[level].get(current_entry, [])
                best_dist = self._dist(self.vectors[current_entry], q_vec)
                for neighbor in neighbors:
                    d = self._dist(self.vectors[neighbor], q_vec)
                    if d < best_dist:
                        best_dist = d
                        current_entry = neighbor
                        changed = True

        # Phase 2: Beam search on layer 0
        candidates = self._search_layer_candidates(q_vec, [current_entry], ef, 0)
        top_k = candidates[:k]

        # Convert distance to similarity (1.0 - dist for cosine)
        results: List[Tuple[str, float]] = []
        for dist, nid in top_k:
            sim = max(0.0, 1.0 - dist)
            results.append((nid, float(sim)))

        return results
