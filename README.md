# Product Normalizer - AIC COMPFEST 18 MVP

Normalisasi nama produk antar-supplier menggunakan AI (IndoBERT + HDBSCAN clustering).
Tempel daftar nama produk mentah dari berbagai sumber, dapatkan kelompok produk yang sama beserta nama kanonis yang disarankan.

## Tech Stack

- **Backend**: FastAPI + Python 3.11
- **AI Model**: LazarusNLP/simcse-indobert-base (IndoBERT-based sentence transformer)
- **Clustering**: HDBSCAN (density-based, no need to specify number of clusters)
- **Frontend**: Single-page vanilla JS + CSS
- **Deployment**: Docker Compose

## Quick Start (Docker)

```bash
docker compose up --build
```

Then open http://localhost:8000

First run will download the IndoBERT model (~400MB) - this is cached in a Docker volume for subsequent runs.

## Local Development (Conda)

```bash
conda create -n product-normalizer python=3.11 -y
conda activate product-normalizer
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

# Pre-download model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('LazarusNLP/simcse-indobert-base')"

# Run
cd backend && uvicorn app.main:app --reload
```

Open http://localhost:8000

## API

- `GET /health` — health check
- `POST /normalize` — body `{"names": ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g"]}`, returns clustered product groups with suggested canonical name and similarity score.

## Example Input

```
Indomie Goreng 85gr
Mie Goreng Indomi 85g
INDOMIE GRNG 85 GR
Sari Roti Tawar 400gr
Roti Tawar Sari Roti 400g
Teh Botol Sosro 400ml
Teh Kotak Sosro 400 ml
```

## Example Output

```json
{
  "clusters": [
    {
      "canonical_name": "Indomie Goreng 85gr",
      "members": ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g", "INDOMIE GRNG 85 GR"],
      "similarity": 0.89
    },
    {
      "canonical_name": "Sari Roti Tawar 400gr",
      "members": ["Sari Roti Tawar 400gr", "Roti Tawar Sari Roti 400g"],
      "similarity": 0.92
    },
    {
      "canonical_name": "Teh Botol Sosro 400ml",
      "members": ["Teh Botol Sosro 400ml", "Teh Kotak Sosro 400 ml"],
      "similarity": 0.78
    }
  ]
}
```

## Architecture

```
Input (product names) 
    → Preprocessing (clean text, normalize units)
    → IndoBERT Embeddings (sentence-transformers)
    → HDBSCAN Clustering (density-based)
    → Canonical name selection (highest avg cosine similarity)
    → Output (clusters with canonical names + similarity scores)
```

## Project Structure

```
product-normalizer/
├── backend/
│   ├── app/
│   │   ├── main.py          # FastAPI app
│   │   ├── matcher.py       # Embedding + clustering logic
│   │   ├── preprocess.py    # Text cleaning
│   │   └── schemas.py       # Pydantic models
│   ├── static/
│   │   ├── index.html       # Frontend
│   │   ├── style.css
│   │   └── app.js
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml
└── README.md
```

## Future Phases (Post-MVP)

- **Tahap 2**: Integrasi ke sistem inventory - update stok otomatis
- **Tahap 3**: Perbandingan harga antar-supplier otomatis
- **Tahap 4**: API ke aplikasi kasir & marketplace
- **Tahap 5**: Basis data produk Indonesia terstandarisasi (katalog universal UMKM)