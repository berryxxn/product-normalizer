# ---- builder: fine-tune the embedding model ----
# Separate stage so the training-only dependencies (datasets, accelerate)
# never end up in the final runtime image. generate_dataset.py uses a fixed
# random seed, so this is fully reproducible from source -- no large binary
# checkpoint needs to live in git (it would exceed GitHub's 100MB file limit
# anyway).
FROM python:3.11-slim AS builder

WORKDIR /build

COPY backend/requirements.txt ./backend-requirements.txt
COPY training/requirements.txt ./training-requirements.txt
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r backend-requirements.txt -r training-requirements.txt

COPY training ./training
RUN python training/generate_dataset.py && python training/train.py

# ---- final: lean runtime image ----
FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt .
# CPU-only torch build: the app forces device="cpu" (see app/model.py), so the
# default CUDA-bundled wheel would only bloat the image for nothing.
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bake the pretrained embedding model into the image too, so app/model.py's
# fallback path also has no runtime dependency on internet access.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# From here on, never phone home for a model -- use what's baked into the image.
ENV HF_HUB_OFFLINE=1

COPY backend/app ./app
COPY backend/static ./static
COPY --from=builder /build/model_weights ./model_weights

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
