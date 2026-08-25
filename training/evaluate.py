import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

EVAL_SET_PATH = Path(__file__).parent / "data" / "eval_set.jsonl"
REPORT_PATH = Path(__file__).parent.parent / "model_weights" / "eval_report.json"

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

def check_clusters(expected_clusters: list[list[str]], label: str) -> dict:
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
    return {
        "label": label,
        "status": status,
        "got_group_count": got_group_count,
        "want_group_count": want_group_count,
        "clusters": [
            {"canonical_name": c.canonical_name, "similarity": c.similarity, "members": c.members}
            for c in result
        ],
    }

def report_eval_set(pretrained: SentenceTransformer, candidate: SentenceTransformer) -> list[dict]:
    if not EVAL_SET_PATH.exists():
        print(f"--- eval set not found at {EVAL_SET_PATH}, skipping category table ---\n")
        return []

    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]

    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    print(f"--- per-category evidence table (eval_set.jsonl, {len(rows)} pairs, held out from training) ---")
    header = f"{'category':<20} {'label':<9} {'n':>3}  {'lexical':>8} {'pretrained':>11} {'fine-tuned':>11}"
    print(header)
    table = []
    for category in sorted(by_category):
        pairs = by_category[category]
        label = pairs[0]["label"]
        lexical_scores, pretrained_scores, candidate_scores = [], [], []
        for row in pairs:
            a, b = row["text_a"], row["text_b"]
            lexical_scores.append(fuzz.token_set_ratio(a, b) / 100.0)
            pretrained_scores.append(cosine(pretrained, a, b))
            candidate_scores.append(cosine(candidate, a, b))
        lexical_avg = float(np.mean(lexical_scores))
        pretrained_avg = float(np.mean(pretrained_scores))
        candidate_avg = float(np.mean(candidate_scores))
        print(
            f"{category:<20} {label:<9} {len(pairs):>3}  "
            f"{lexical_avg:>8.3f} {pretrained_avg:>11.3f} "
            f"{candidate_avg:>11.3f}"
        )
        table.append({
            "category": category,
            "label": label,
            "n": len(pairs),
            "lexical_avg": round(lexical_avg, 3),
            "pretrained_semantic_avg": round(pretrained_avg, 3),
            "fine_tuned_semantic_avg": round(candidate_avg, 3),
        })
    print("(label=positive wants HIGH scores, label=negative wants LOW scores)\n")
    return table

def main() -> None:
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_CHECKPOINT)

    pretrained = SentenceTransformer(PRETRAINED, device="cpu")
    candidate = SentenceTransformer(checkpoint, device="cpu")

    check_semantic_pairs(pretrained, candidate)
    eval_set_table = report_eval_set(pretrained, candidate)

    import os
    os.environ["MODEL_PATH"] = checkpoint
    import app.model as model_module
    model_module.MODEL_PATH = checkpoint
    model_module._model = None 

    in_dist_report = check_clusters(IN_DISTRIBUTION_CLUSTERS, "in-distribution (trained categories)")
    held_out_report = check_clusters(HELD_OUT_CLUSTERS, "held-out canary (sardines, never trained on)")

    passed = in_dist_report["status"] == "PASS" and held_out_report["status"] == "PASS"
    verdict = "PASS -- safe to ship" if passed else "FAIL -- do not ship"
    print(f"=== VERDICT: {verdict} ===")

    report = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "verdict": "PASS" if passed else "FAIL",
        "in_distribution_check": in_dist_report,
        "held_out_canary_check": held_out_report,
        "per_category_evidence_table": eval_set_table,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"wrote eval report to {REPORT_PATH}")

    if not passed:
        sys.exit(1)

if __name__ == "__main__":
    main()
