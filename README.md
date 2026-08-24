# Product Normalizer

Normalisasi nama produk antar-supplier: tempel daftar nama produk mentah, dapatkan kelompok
produk yang sama beserta nama kanonis yang disarankan.

Matching is hybrid: semantic similarity from a multilingual sentence-embedding model
(`paraphrase-multilingual-MiniLM-L12-v2`, falls back to pretrained if no fine-tuned checkpoint
is mounted at `model_weights/`) combined with lexical similarity (`rapidfuzz`), clustered with
`AgglomerativeClustering`. See `CLAUDE.md` for the full design rationale.

## Run locally with Docker

```
docker compose up --build
```

Then open http://localhost:8000 in your browser.

The first build downloads the embedding model (~500MB) and bakes it into the image, so it can
take several minutes — this is a one-time cost. After that, the app has no runtime dependency
on internet access.

## Run locally without Docker

```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000. The first request downloads the embedding model from
Hugging Face (not baked in outside Docker), so it'll be slow once.

## API

- `GET /health` — health check
- `POST /normalize` — body `{"names": ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g"]}`,
  returns clustered product groups with a suggested canonical name and similarity score each.
