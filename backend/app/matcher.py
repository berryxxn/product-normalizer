from difflib import SequenceMatcher

from app.preprocess import clean_text
from app.schemas import Cluster

SIMILARITY_THRESHOLD = 0.6


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, clean_text(a), clean_text(b)).ratio()


def _pick_canonical(members: list[str]) -> tuple[str, float]:
    if len(members) == 1:
        return members[0], 1.0

    best_name = members[0]
    best_avg = -1.0
    for candidate in members:
        scores = [similarity(candidate, other) for other in members if other != candidate]
        avg = sum(scores) / len(scores)
        if avg > best_avg:
            best_avg = avg
            best_name = candidate
    return best_name, best_avg


def cluster_names(names: list[str]) -> list[Cluster]:
    groups: list[list[str]] = []

    for name in names:
        best_group_idx = None
        best_score = 0.0
        for idx, group in enumerate(groups):
            score = max(similarity(name, member) for member in group)
            if score > best_score:
                best_score = score
                best_group_idx = idx

        if best_group_idx is not None and best_score >= SIMILARITY_THRESHOLD:
            groups[best_group_idx].append(name)
        else:
            groups.append([name])

    clusters = []
    for group in groups:
        canonical, avg_similarity = _pick_canonical(group)
        clusters.append(
            Cluster(canonical_name=canonical, members=group, similarity=round(avg_similarity, 3))
        )
    return clusters
