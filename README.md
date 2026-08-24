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

## Fine-tuning

`training/generate_dataset.py` + `training/train.py` build a synthetic dataset and fine-tune
the embedding model with `MultipleNegativesRankingLoss`, saving to `model_weights/`. These are
offline scripts, not part of the running app -- `model.py` falls back to the pretrained model
automatically when `model_weights/` has no checkpoint (its current, verified-working state).

**Not currently usable as-is.** The current synthetic dataset improves abbreviation/typo
recognition but regresses same-brand-different-flavor separation badly enough to break real
clustering cases the pretrained model gets right (tested at both 1 and 3 epochs). Before trusting
a checkpoint from this script, verify it doesn't regress on hard negatives like "Indomie Goreng"
vs "Indomie Ayam Bawang" -- don't assume fine-tuned beats pretrained without checking.

## API

- `GET /health` — health check
- `POST /normalize` — body `{"names": ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g"]}`,
  returns clustered product groups with a suggested canonical name and similarity score each.
