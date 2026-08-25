import logging

import numpy as np
from rapidfuzz import fuzz
from sklearn.cluster import AgglomerativeClustering

from app.model import embed
from app.preprocess import clean_batch, clean_text, extract_quantity
from app.schemas import Cluster

logger = logging.getLogger(__name__)

SEMANTIC_WEIGHT = 0.65
LEXICAL_WEIGHT = 0.35
DISTANCE_THRESHOLD = 0.30
QUANTITY_TOLERANCE = 0.02


def _similarity_matrix(names: list[str]) -> np.ndarray:
    cleaned = clean_batch(names)

    embeddings = embed(cleaned)
    semantic = embeddings @ embeddings.T
    semantic = np.clip(semantic, 0.0, 1.0)
    logger.debug("matcher: semantic similarity matrix (cosine)=\n%s", semantic)

    n = len(cleaned)
    lexical = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            score = fuzz.token_set_ratio(cleaned[i], cleaned[j]) / 100.0
            lexical[i, j] = lexical[j, i] = score
    logger.debug("matcher: lexical similarity matrix (rapidfuzz token_set_ratio)=\n%s", lexical)

    combined = SEMANTIC_WEIGHT * semantic + LEXICAL_WEIGHT * lexical
    logger.info(
        "matcher: combined similarity matrix computed (semantic_weight=%s, lexical_weight=%s)",
        SEMANTIC_WEIGHT, LEXICAL_WEIGHT,
    )
    logger.debug("matcher: combined similarity matrix=\n%s", combined)
    return combined


def _pick_canonical(members: list[str], sub_similarity: np.ndarray) -> tuple[str, float]:
    if len(members) == 1:
        return members[0], 1.0

    avg_scores = (sub_similarity.sum(axis=1) - 1.0) / (len(members) - 1)
    best_idx = int(np.argmax(avg_scores))
    return members[best_idx], float(avg_scores[best_idx])


def cluster_names(names: list[str]) -> list[Cluster]:
    logger.info("matcher.cluster_names: called with %d name(s): %s", len(names), names)

    if len(names) == 1:
        logger.info("matcher.cluster_names: single name, trivial 1-member cluster")
        return [Cluster(canonical_name=names[0], members=names, similarity=1.0)]

    similarity = _similarity_matrix(names)
    distance = np.clip(1.0 - similarity, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)

    quantities = [extract_quantity(clean_text(name)) for name in names]
    logger.info("matcher: quantity guard -- parsed quantities=%s", quantities)
    n = len(names)
    guard_hits = 0
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
                guard_hits += 1
    logger.info(
        "matcher: quantity guard forced distance=1.0 for %d pair(s) (tolerance=%s)",
        guard_hits, QUANTITY_TOLERANCE,
    )

    labels = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=DISTANCE_THRESHOLD,
        n_clusters=None,
    ).fit_predict(distance)
    logger.info(
        "matcher: AgglomerativeClustering found %d cluster(s) (distance_threshold=%s)",
        len(set(labels)), DISTANCE_THRESHOLD,
    )

    clusters = []
    for label in sorted(set(labels)):
        idx = [i for i, l in enumerate(labels) if l == label]
        members = [names[i] for i in idx]
        sub_similarity = similarity[np.ix_(idx, idx)]
        canonical, avg_similarity = _pick_canonical(members, sub_similarity)
        logger.info(
            "matcher: cluster %s -> canonical=%r similarity=%.3f members=%s",
            label, canonical, avg_similarity, members,
        )
        clusters.append(
            Cluster(canonical_name=canonical, members=members, similarity=round(avg_similarity, 3))
        )

    logger.info("matcher.cluster_names: result -- %d cluster(s) from %d name(s)", len(clusters), len(names))
    return clusters
