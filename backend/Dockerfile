FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch build: the app forces device="cpu" (see app/model.py), so the
# default CUDA-bundled wheel would only bloat the image for nothing.
RUN pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bake the pretrained embedding model into the image so `docker compose up`
# has no runtime dependency on internet access (judges may run this offline).
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# From here on, never phone home for the model — use what's baked into the image.
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY static ./static

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
