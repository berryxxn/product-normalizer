"""Evaluate a fine-tuned checkpoint before trusting it. Offline script, not
part of the running app.

Checks five things, comparing the candidate checkpoint against the pretrained
baseline:
  1. Raw semantic similarity on known hard cases (diagnostic, not pass/fail
     on its own -- the hybrid matcher's lexical weighting can compensate for
     weak embedding scores, see README.md's Fine-tuning section).
  2. The full hybrid matcher's clustering output on those same cases -- this
     IS pass/fail, since it's what the deployed app actually does.
  3. The same clustering check on a held-out category (diapers/baby care)
     that never appears in generate_dataset.py's training catalog -- this is
     the canary for catastrophic forgetting on categories the model wasn't
     trained on.
  4. Verdict: only PASS if both clustering checks (2 and 3) are fully correct.
  5. A per-category evidence table against data/eval_set.jsonl (disjoint
     product families, tagged A/B/D/F/G/H) -- lexical (rapidfuzz) vs.
     pretrained-semantic vs. fine-tuned-semantic, averaged per category.
     Diagnostic, not part of the ship/no-ship verdict: this is the evidence
     that feeds the "AI is needed, not forced" argument in
     .personal-storage/PADAN_dataset_finetuning_spec.md, not a pass/fail gate.

Usage: python training/evaluate.py [path-to-checkpoint]
Defaults to model_weights/ if no path given.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

EVAL_SET_PATH = Path(__file__).parent / "data" / "eval_set.jsonl"

PRETRAINED = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_CHECKPOINT = Path(__file__).parent.parent / "model_weights"

# (a, b, "HIGH"/"LOW", description) -- HIGH = same product, LOW = different product
SEMANTIC_PAIRS = [
    ("Indomie Goreng 85gr", "Mie Goreng Indomi 85g", "HIGH", "abbreviation"),
    ("Indomie Goreng 85gr", "INDOMIE GRNG 85 GR", "HIGH", "heavy abbreviation (flagship example)"),
    ("Indomie Goreng 85gr", "Indomie Ayam Bawang 85gr", "LOW", "different flavor"),
    ("Indomie Goreng 85gr", "Indomie Goreng 5x85gr", "LOW", "different pack size"),
    ("Kopi Kapal Api Sachet", "Kopi Kapal Api Renceng", "HIGH", "synonym"),
]

# expected clusters: each inner list is one group that must end up together,
# separate from every other group.
IN_DISTRIBUTION_CLUSTERS = [
    ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g", "INDOMIE GRNG 85 GR", "Goreng Indomie 85gr",
     "Indomie Goreng 0.085kg", "Indomei Goreng 85gr"],
    ["Indomie Ayam Bawang 85gr"],
    ["Indomie Goreng 5x85gr"],
    ["Teh Botol Sosro 450ml", "Teh Botol Sosro 450 ml"],
    ["Beras Ramos 5kg", "Beras Ramos 5 kilogram"],
    ["Kopi Kapal Api Sachet", "Kopi Kapal Api Renceng"],
]

# held-out category: never appears in generate_dataset.py's catalog. Tests
# only the exact relationships already verified as the working bar elsewhere
# (paraphrase-positive, flavor-negative, size-negative) within one unseen
# brand -- deliberately excludes cross-brand comparison, since that's a
# separate, pre-existing gap in the baseline itself (confirmed: even the
# pretrained model merges same-flavor-same-size-different-brand pairs) and
# would confound whether a failure here is caused by fine-tuning or not.
HELD_OUT_CLUSTERS = [
    ["Sarden ABC Saus Tomat 155gr", "Ikan Sarden ABC Saus Tomat 155gr"],
    ["Sarden ABC Extra Pedas 155gr"],
    ["Sarden ABC Saus Tomat 425gr"],
]


def cosine(m: SentenceTransformer, a: str, b: str) -> float:
    emb = m.encode([a, b], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def check_semantic_pairs(pretrained: SentenceTransformer, candidate: SentenceTransformer) -> None:
    print("--- raw semantic similarity (diagnostic) ---")
    print(f"{'pretrained':>10} {'candidate':>10}  description")
    for a, b, direction, desc in SEMANTIC_PAIRS:
        p_sim = cosine(pretrained, a, b)
        c_sim = cosine(candidate, a, b)
        print(f"{p_sim:10.3f} {c_sim:10.3f}  want {direction:4s} -- {desc}")
    print()


def check_clusters(expected_clusters: list[list[str]], label: str) -> bool:
    from app.matcher import cluster_names

    names = [n for group in expected_clusters for n in group]
    result = cluster_names(names)

    name_to_group = {}
    for gi, group in enumerate(expected_clusters):
        for n in group:
            name_to_group[n] = gi

    ok = True
    for cluster in result:
        expected_groups = {name_to_group[m] for m in cluster.members}
        if len(expected_groups) != 1:
            ok = False

    got_group_count = len(result)
    want_group_count = len(expected_clusters)
    if got_group_count != want_group_count:
        ok = False

    status = "PASS" if ok else "FAIL"
    print(f"--- clustering check: {label} -- {status} ({got_group_count} groups, want {want_group_count}) ---")
    for cluster in result:
        print(f"  {cluster.canonical_name!r} sim={cluster.similarity}  members={cluster.members}")
    print()
    return ok


def report_eval_set(pretrained: SentenceTransformer, candidate: SentenceTransformer) -> None:
    if not EVAL_SET_PATH.exists():
        print(f"--- eval set not found at {EVAL_SET_PATH}, skipping category table ---\n")
        return

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    print(f"--- per-category evidence table (eval_set.jsonl, {len(rows)} pairs, held out from training) ---")
    header = f"{'category':<20} {'label':<9} {'n':>3}  {'lexical':>8} {'pretrained':>11} {'fine-tuned':>11}"
    print(header)
    for category in sorted(by_category):
        pairs = by_category[category]
        label = pairs[0]["label"]
        lexical_scores, pretrained_scores, candidate_scores = [], [], []
        for row in pairs:
            a, b = row["text_a"], row["text_b"]
            lexical_scores.append(fuzz.token_set_ratio(a, b) / 100.0)
            pretrained_scores.append(cosine(pretrained, a, b))
            candidate_scores.append(cosine(candidate, a, b))
        print(
            f"{category:<20} {label:<9} {len(pairs):>3}  "
            f"{np.mean(lexical_scores):>8.3f} {np.mean(pretrained_scores):>11.3f} "
            f"{np.mean(candidate_scores):>11.3f}"
        )
    print("(label=positive wants HIGH scores, label=negative wants LOW scores)\n")


def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CHECKPOINT)

    pretrained = SentenceTransformer(PRETRAINED, device="cpu")
    candidate = SentenceTransformer(checkpoint, device="cpu")

    check_semantic_pairs(pretrained, candidate)
    report_eval_set(pretrained, candidate)

    import os
    os.environ["MODEL_PATH"] = checkpoint
    import app.model as model_module
    model_module.MODEL_PATH = checkpoint
    model_module._model = None  # force reload with the candidate checkpoint

    in_dist_ok = check_clusters(IN_DISTRIBUTION_CLUSTERS, "in-distribution (trained categories)")
    held_out_ok = check_clusters(HELD_OUT_CLUSTERS, "held-out canary (sardines, never trained on)")

    verdict = "PASS -- safe to ship" if (in_dist_ok and held_out_ok) else "FAIL -- do not ship"
    print(f"=== VERDICT: {verdict} ===")


if __name__ == "__main__":
    main()
