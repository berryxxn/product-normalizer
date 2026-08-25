import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
OUTPUT_PATH = REPO_ROOT / "model_weights" / "build_info.json"

BASE_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
LOSS_FUNCTION = "TripletLoss"

DATASET_GENERATION_PARAMS = {
    "random_seed": 42,
    "category_taxonomy": [
        "A_lexical",
        "B_sinonim",
        "D_brand_deskriptif",
        "F_size_trap",
        "G_rasa_trap",
        "H_brand_neighbor",
    ],
}


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip()


def main() -> None:
    info = {
        "training_commit": get_git_commit(),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "base_model": BASE_MODEL,
        "loss_function": LOSS_FUNCTION,
        "dataset_generation_params": DATASET_GENERATION_PARAMS,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    print(f"wrote build info to {OUTPUT_PATH}")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
