from pydantic import BaseModel, Field


class NormalizeRequest(BaseModel):
    names: list[str] = Field(..., min_length=1, description="Raw product names, one per entry")


class Cluster(BaseModel):
    canonical_name: str
    members: list[str]
    similarity: float


class NormalizeResponse(BaseModel):
    clusters: list[Cluster]
