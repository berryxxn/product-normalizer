# Product Normalizer

Normalisasi nama produk antar-supplier: tempel daftar nama produk mentah, dapatkan kelompok
produk yang sama beserta nama kanonis yang disarankan.

This is an early scaffold: matching currently runs on plain string similarity (Python's
`difflib`), no embedding model yet. The AI-based hybrid matcher described in the project spec
will replace `backend/app/matcher.py` once the training pipeline is ready.

## Run locally with Docker

```
docker compose up --build
```

Then open http://localhost:8000 in your browser.

## Run locally without Docker

```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000.

## API

- `GET /health` — health check
- `POST /normalize` — body `{"names": ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g"]}`,
  returns clustered product groups with a suggested canonical name and similarity score each.
