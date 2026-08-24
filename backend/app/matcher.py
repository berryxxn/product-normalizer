import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity

from app.preprocess import clean_text
from app.schemas import Cluster

MODEL_NAME = "LazarusNLP/simcse-indobert-base"
_SIMILARITY_THRESHOLD = 0.65

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_names(names: List[str]) -> np.ndarray:
    model = get_model()
    cleaned = [clean_text(name) for name in names]
    embeddings = model.encode(cleaned, normalize_embeddings=True, show_progress_bar=False)
    return embeddings


def _pick_canonical(members: List[str], embeddings: np.ndarray, indices: List[int]) -> Tuple[str, float]:
    if len(members) == 1:
        return members[0], 1.0

    cluster_embeddings = embeddings[indices]
    similarities = cosine_similarity(cluster_embeddings)
    np.fill_diagonal(similarities, 0)
    avg_similarities = similarities.mean(axis=1)
    best_idx = int(np.argmax(avg_similarities))
    return members[best_idx], float(avg_similarities[best_idx])


def cluster_names(names: List[str]) -> List[Cluster]:
    if not names:
        return []

    embeddings = encode_names(names)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)

    unique_labels = set(labels)
    clusters = []

    for label in sorted(unique_labels):
        if label == -1:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        members = [names[i] for i in indices]
        canonical, avg_sim = _pick_canonical(members, embeddings, indices)
        clusters.append(
            Cluster(
                canonical_name=canonical,
                members=members,
                similarity=round(avg_sim, 3)
            )
        )

    noise_indices = [i for i, l in enumerate(labels) if l == -1]
    for idx in noise_indices:
        clusters.append(
            Cluster(
                canonical_name=names[idx],
                members=[names[idx]],
                similarity=1.0
            )
        )

    return clusters