import logging

from fastapi import FastAPI

from app.matcher import cluster_names
from app.schemas import NormalizeRequest, NormalizeResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Product Normalizer")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(payload: NormalizeRequest):
    logger.info("POST /normalize: received %d name(s)", len(payload.names))
    clusters = cluster_names(payload.names)
    logger.info("POST /normalize: returning %d cluster(s)", len(clusters))
    return NormalizeResponse(clusters=clusters)
