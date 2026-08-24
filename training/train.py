"""Fine-tune the product-name embedding model on the synthetic dataset from
generate_dataset.py. Offline script, not part of the running app -- run it,
then the checkpoint at model_weights/ is picked up automatically by
backend/app/model.py on the next container start.

Known limitation: see the "Fine-tuning" section in the top-level README.md
before trusting a checkpoint from this script -- the current dataset design
regresses flavor discrimination.
"""
import json
from pathlib import Path

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DATA_PATH = Path(__file__).parent / "data" / "train_pairs.jsonl"
OUTPUT_PATH = Path(__file__).parent.parent / "model_weights"

BATCH_SIZE = 16
EPOCHS = 3


def load_examples() -> tuple[list[InputExample], list[InputExample]]:
    positive_examples = []
    triplet_examples = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("negative"):
                triplet_examples.append(
                    InputExample(texts=[row["anchor"], row["positive"], row["negative"]])
                )
            else:
                positive_examples.append(InputExample(texts=[row["anchor"], row["positive"]]))
    return positive_examples, triplet_examples


def main() -> None:
    positive_examples, triplet_examples = load_examples()
    print(f"positive pairs: {len(positive_examples)}, triplets w/ hard negative: {len(triplet_examples)}")

    model = SentenceTransformer(BASE_MODEL, device="cpu")

    train_objectives = []
    if positive_examples:
        loader = DataLoader(positive_examples, shuffle=True, batch_size=BATCH_SIZE)
        train_objectives.append((loader, losses.MultipleNegativesRankingLoss(model)))
    if triplet_examples:
        # TripletLoss enforces an absolute margin between anchor-positive and
        # anchor-negative distance, directly targeting the failure MNRL missed:
        # same-brand-different-flavor pairs need to be pushed apart, not just
        # ranked below the positive within a batch.
        loader = DataLoader(triplet_examples, shuffle=True, batch_size=BATCH_SIZE)
        triplet_loss = losses.TripletLoss(
            model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.3,
        )
        train_objectives.append((loader, triplet_loss))

    total_examples = len(positive_examples) + len(triplet_examples)
    steps_per_epoch = max(1, total_examples // BATCH_SIZE)
    warmup_steps = int(0.1 * steps_per_epoch * EPOCHS)

    model.fit(
        train_objectives=train_objectives,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        show_progress_bar=True,
    )

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    model.save(str(OUTPUT_PATH))
    print(f"saved fine-tuned checkpoint to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
