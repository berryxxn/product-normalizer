import logging
import os

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_PATH = os.environ.get("MODEL_PATH", "model_weights")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        if os.path.isfile(os.path.join(MODEL_PATH, "modules.json")):
            _model = SentenceTransformer(MODEL_PATH, device="cpu")
            logger.info("Loaded fine-tuned checkpoint from %s", MODEL_PATH)
        else:
            _model = SentenceTransformer(PRETRAINED_MODEL_NAME, device="cpu")
            logger.warning(
                "No fine-tuned checkpoint found at %s — falling back to pretrained base model. "
                "This is NOT an acceptable state for the deployed demo.",
                MODEL_PATH,
            )
    return _model


def embed(names: list[str]):
    logger.info("model.embed: encoding %d name(s)", len(names))
    vectors = get_model().encode(names, convert_to_numpy=True, normalize_embeddings=True)
    logger.debug("model.embed: produced embedding matrix with shape %s", vectors.shape)
    return vectors
