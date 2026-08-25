# ---- builder: fine-tune the embedding model ----
# Separate stage so the training-only dependencies (datasets, accelerate)
# never end up in the final runtime image. generate_dataset.py uses a fixed
# random seed, so this is fully reproducible from source -- no large binary
# checkpoint needs to live in git (it would exceed GitHub's 100MB file limit
# anyway).
FROM python:3.11-slim AS builder

WORKDIR /build

# git is only needed here (builder stage) so write_build_info.py can read
# `git rev-parse HEAD` for the checkpoint's traceability metadata -- not
# installed in the final runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend-requirements.txt
COPY training/requirements.txt ./training-requirements.txt
RUN pip install --no-cache-dir torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r backend-requirements.txt -r training-requirements.txt

COPY training ./training
# evaluate.py imports app.matcher to run the real hybrid-matcher clustering
# gate, so the app package needs to exist at ./backend/app here too.
COPY backend/app ./backend/app
COPY .git ./.git

# generate_dataset.py + train.py produce the checkpoint; generate_eval_set.py
# builds the held-out category table evaluate.py reports on; evaluate.py then
# gates the checkpoint (non-zero exit on FAIL fails this whole build, so a bad
# checkpoint can never ship silently) and writes model_weights/eval_report.json;
# write_build_info.py records what commit/config produced it. All offline,
# reproducible from source per the fixed random seed in generate_dataset.py.
RUN python training/generate_dataset.py \
    && python training/train.py \
    && python training/generate_eval_set.py \
    && python training/evaluate.py \
    && python training/write_build_info.py

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
