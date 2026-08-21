from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.matcher import cluster_names
from app.schemas import NormalizeRequest, NormalizeResponse

app = FastAPI(title="Product Normalizer")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/normalize", response_model=NormalizeResponse)
def normalize(payload: NormalizeRequest):
    clusters = cluster_names(payload.names)
    return NormalizeResponse(clusters=clusters)


app.mount("/", StaticFiles(directory="static", html=True), name="static")
