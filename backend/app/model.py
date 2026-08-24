import os

from sentence_transformers import SentenceTransformer

PRETRAINED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = os.environ.get("MODEL_PATH", "model_weights")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        if os.path.isfile(os.path.join(MODEL_PATH, "modules.json")):
            _model = SentenceTransformer(MODEL_PATH, device="cpu")
        else:
            _model = SentenceTransformer(PRETRAINED_MODEL_NAME, device="cpu")
    return _model


def embed(names: list[str]):
    return get_model().encode(names, convert_to_numpy=True, normalize_embeddings=True)
