"""Product Quantization (PQ) & Asymmetric Distance Computation (ADC) Engine.

Compresses high-dimensional dense embeddings (e.g. 768-D) by decomposing the vector space
into M orthogonal sub-spaces and vector-quantizing each sub-vector into K centroids.
Enables O(M) fast similarity evaluations via pre-computed Asymmetric Distance Tables.

Mathematical Formulation:
1. Subvector Decomposition:
   x in R^D is split into [x^(1), x^(2), ..., x^(M)], where each x^(m) in R^(D/M).
2. Codebook Quantization:
   Each subspace m has a codebook C^(m) = {c_1^(m), ..., c_K^(m)}.
   Quantization: q(x) = [argmin_k ||x^(1) - c_k^(1)||, ..., argmin_k ||x^(M) - c_k^(M)||].
3. Asymmetric Distance Computation (ADC):
   For query y and quantized database vector q(x):
   ||y - q(x)||^2 = sum_{m=1}^M ||y^(m) - c_{q_m(x)}^(m)||^2.
   Lookups are O(1) per subspace from pre-computed tables T[m, k] = ||y^(m) - c_k^(m)||^2.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


class ProductQuantizer:
    """Product Quantizer for dense embedding compression and fast ADC search."""

    def __init__(
        self,
        dim: int = 768,
        m: int = 16,
        k: int = 256,
        max_iters: int = 20,
        seed: int = 42,
    ) -> None:
        if dim % m != 0:
            raise ValueError(f"Vector dim {dim} must be divisible by subspace count {m}")

        self.dim = dim
        self.m = m
        self.d_sub = dim // m
        self.k = k
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)

        # Codebooks: shape (m, k, d_sub)
        self.codebooks: Optional[np.ndarray] = None
        self.is_trained: bool = False

    def train(self, vectors: np.ndarray) -> None:
        """Train codebooks on a training matrix of shape (N, dim) using k-means."""
        vecs = np.asarray(vectors, dtype=np.float32)
        n_samples, dim = vecs.shape
        if dim != self.dim:
            raise ValueError(f"Training vectors dimension {dim} != expected {self.dim}")

        effective_k = min(self.k, n_samples)
        self.codebooks = np.zeros((self.m, effective_k, self.d_sub), dtype=np.float32)

        for m_idx in range(self.m):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            sub_vecs = vecs[:, start:end]

            centroids = self._kmeans(sub_vecs, effective_k)
            self.codebooks[m_idx, :effective_k, :] = centroids

        self.is_trained = True

    def _kmeans(self, data: np.ndarray, k: int) -> np.ndarray:
        """Run vectorized Lloyd's k-means on a sub-vector matrix."""
        n_samples = data.shape[0]
        init_indices = self.rng.choice(n_samples, size=k, replace=False)
        centroids = data[init_indices].copy()

        for _ in range(self.max_iters):
            # Compute squared distances between data (N, D) and centroids (K, D)
            # (a - b)^2 = a^2 - 2ab + b^2
            dists = (
                np.sum(data**2, axis=1, keepdims=True)
                - 2 * np.dot(data, centroids.T)
                + np.sum(centroids**2, axis=1, keepdims=True).T
            )
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(centroids)
            counts = np.bincount(labels, minlength=k)

            for i in range(k):
                if counts[i] > 0:
                    new_centroids[i] = np.mean(data[labels == i], axis=0)
                else:
                    new_centroids[i] = data[self.rng.integers(0, n_samples)]

            if np.allclose(centroids, new_centroids, atol=1e-4):
                break
            centroids = new_centroids

        return centroids

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode (N, dim) vectors into (N, m) discrete centroid indices (uint8/uint16)."""
        if not self.is_trained or self.codebooks is None:
            raise RuntimeError("ProductQuantizer must be trained before encoding")

        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]

        n_samples = vecs.shape[0]
        codes = np.zeros((n_samples, self.m), dtype=np.uint8 if self.k <= 256 else np.uint16)

        for m_idx in range(self.m):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            sub_vecs = vecs[:, start:end]
            centroids = self.codebooks[m_idx]

            dists = (
                np.sum(sub_vecs**2, axis=1, keepdims=True)
                - 2 * np.dot(sub_vecs, centroids.T)
                + np.sum(centroids**2, axis=1, keepdims=True).T
            )
            codes[:, m_idx] = np.argmin(dists, axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct approximate (N, dim) vectors from quantized codes."""
        if not self.is_trained or self.codebooks is None:
            raise RuntimeError("ProductQuantizer must be trained before decoding")

        codes = np.asarray(codes)
        if codes.ndim == 1:
            codes = codes[np.newaxis, :]

        n_samples = codes.shape[0]
        reconstructed = np.zeros((n_samples, self.dim), dtype=np.float32)

        for m_idx in range(self.m):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            reconstructed[:, start:end] = self.codebooks[m_idx, codes[:, m_idx], :]

        return reconstructed

    def compute_asymmetric_distances(self, query: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric Distance Computation (ADC) against quantized database.

        Complexity: O(M * K * d_sub) to build lookup table + O(N * M) for table lookups.
        """
        if not self.is_trained or self.codebooks is None:
            raise RuntimeError("ProductQuantizer must be trained before computing distances")

        q = np.asarray(query, dtype=np.float32).ravel()
        if q.shape[0] != self.dim:
            raise ValueError(f"Query vector dimension {q.shape[0]} != expected {self.dim}")

        n_items = codes.shape[0]
        effective_k = self.codebooks.shape[1]

        # 1. Build Distance Table: shape (m, effective_k)
        lut = np.zeros((self.m, effective_k), dtype=np.float32)
        for m_idx in range(self.m):
            start = m_idx * self.d_sub
            end = start + self.d_sub
            q_sub = q[start:end]
            centroids = self.codebooks[m_idx]
            lut[m_idx] = np.sum((centroids - q_sub) ** 2, axis=1)

        # 2. Accumulate distances via direct lookup
        total_distances = np.zeros(n_items, dtype=np.float32)
        for m_idx in range(self.m):
            sub_codes = codes[:, m_idx]
            total_distances += lut[m_idx, sub_codes]

        return total_distances


class InvertedIndexPQ:
    """Inverted File with Product Quantization (IVF-PQ) Index.

    Combines coarse Voronoi cell partitioning with residual product quantization
    for sub-linear search across large corpora.
    """

    def __init__(self, dim: int = 768, n_lists: int = 8, m: int = 16, k_sub: int = 64) -> None:
        self.dim = dim
        self.n_lists = n_lists
        self.pq = ProductQuantizer(dim=dim, m=m, k=k_sub)

        self.coarse_centroids: Optional[np.ndarray] = None
        self.inv_lists_ids: Dict[int, List[str]] = {i: [] for i in range(n_lists)}
        self.inv_lists_codes: Dict[int, List[np.ndarray]] = {i: [] for i in range(n_lists)}
        self.is_trained: bool = False

    def train_and_index(self, doc_ids: Sequence[str], vectors: np.ndarray) -> None:
        """Train coarse centroids and sub-vector quantizers, then index entries."""
        vecs = np.asarray(vectors, dtype=np.float32)
        n_samples = vecs.shape[0]

        # 1. Train coarse centroids
        rng = np.random.default_rng(42)
        init_idx = rng.choice(n_samples, size=min(self.n_lists, n_samples), replace=False)
        self.coarse_centroids = vecs[init_idx].copy()

        # Coarse assignment
        dists = np.sum((vecs[:, None, :] - self.coarse_centroids[None, :, :]) ** 2, axis=2)
        cluster_assignments = np.argmin(dists, axis=1)

        # 2. Compute residuals: r = x - C_coarse
        residuals = vecs - self.coarse_centroids[cluster_assignments]

        # 3. Train PQ on residuals
        self.pq.train(residuals)
        codes = self.pq.encode(residuals)

        # 4. Populate inverted lists
        for i, doc_id in enumerate(doc_ids):
            c_id = int(cluster_assignments[i])
            self.inv_lists_ids[c_id].append(doc_id)
            self.inv_lists_codes[c_id].append(codes[i])

        self.is_trained = True

    def query(self, query_vector: np.ndarray, top_k: int = 10, n_probe: int = 2) -> List[Tuple[str, float]]:
        """Query index using IVF multi-probe and ADC on residuals."""
        if not self.is_trained or self.coarse_centroids is None:
            return []

        q = np.asarray(query_vector, dtype=np.float32).ravel()

        # Find closest coarse centroids
        coarse_dists = np.sum((self.coarse_centroids - q) ** 2, axis=1)
        probe_lists = np.argsort(coarse_dists)[:n_probe]

        candidates: List[Tuple[float, str]] = []

        for c_id in probe_lists:
            ids = self.inv_lists_ids[c_id]
            if not ids:
                continue

            codes = np.array(self.inv_lists_codes[c_id])
            residual_q = q - self.coarse_centroids[c_id]
            dists = self.pq.compute_asymmetric_distances(residual_q, codes)

            for d, doc_id in zip(dists, ids):
                candidates.append((float(d), doc_id))

        candidates.sort(key=lambda x: x[0])
        return [(doc_id, dist) for dist, doc_id in candidates[:top_k]]
