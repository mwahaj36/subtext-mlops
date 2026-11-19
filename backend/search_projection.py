"""Vector geometry shared by retrieval and the galaxy mapper."""
import numpy as np


def normalize(vector):
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def make_centered_similarity(query, mean):
    query = normalize(np.asarray(query, dtype=float) - mean)

    def similarity(vector):
        if vector is None:
            return 0.0
        return float(np.dot(normalize(np.asarray(vector, dtype=float) - mean), query))

    return similarity


def linear_project(vectors, path):
    with np.load(path) as data:
        return np.asarray(vectors) @ data["coef"].T + data["intercept"]
