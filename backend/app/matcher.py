import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering

from app.model import embed
from app.preprocess import clean_text, extract_quantity
from app.schemas import Cluster

SEMANTIC_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35
DISTANCE_THRESHOLD = 0.30
QUANTITY_TOLERANCE = 0.02


def _similarity_matrix(names: list[str]) -> np.ndarray:
    cleaned = [clean_text(name) for name in names]

    embeddings = embed(cleaned)
    semantic = embeddings @ embeddings.T
    semantic = np.clip(semantic, 0.0, 1.0)

    n = len(cleaned)
    lexical = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            score = fuzz.token_set_ratio(cleaned[i], cleaned[j]) / 100.0
            lexical[i, j] = lexical[j, i] = score

    return SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical


def _pick_canonical(members: list[str], sub_similarity: np.ndarray) -> tuple[str, float]:
    if len(members) == 1:
        return members[0], 1.0

    avg_scores = (sub_similarity.sum(axis=1) - 1.0) / (len(members) - 1)
    best_idx = int(np.argmax(avg_scores))
    return members[best_idx], float(avg_scores[best_idx])


def cluster_names(names: list[str]) -> list[Cluster]:
    if len(names) == 1:
        return [Cluster(canonical_name=names[0], members=names, similarity=1.0)]

    similarity = _similarity_matrix(names)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    quantities = [extract_quantity(clean_text(name)) for name in names]
    n = len(names)
    for i in range(n):
        if quantities[i] is None:
            continue
        for j in range(i + 1, n):
            if quantities[j] is None:
                continue
            larger = max(quantities[i], quantities[j])
            if larger == 0:
                continue
            relative_diff = abs(quantities[i] - quantities[j]) / larger
            if relative_diff > QUANTITY_TOLERANCE:
                distance[i, j] = distance[j, i] = 1.0

    labels = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
        n_clusters=None,
    ).fit_predict(distance)

    clusters = []
    for label in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == label]
        members = [names[i] for i in idx]
        sub_similarity = similarity[np.ix_(idx, idx)]
        canonical, avg_similarity = _pick_canonical(members, sub_similarity)
        clusters.append(
            Cluster(canonical_name=canonical, members=members, similarity=round(avg_similarity, 3))
        )
    return clusters
