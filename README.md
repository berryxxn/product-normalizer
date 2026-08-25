# Product Normalizer

Normalisasi nama produk antar-supplier: tempel daftar nama produk mentah, dapatkan kelompok
produk yang sama beserta nama kanonis yang disarankan.

Matching is hybrid: semantic similarity from a fine-tuned multilingual sentence-embedding model
(base: `paraphrase-multilingual-MiniLM-L12-v2`; falls back to the pretrained base model if
`model_weights/` has no checkpoint) combined with lexical similarity (`rapidfuzz`), clustered
with `AgglomerativeClustering`. See `CLAUDE.md` for the full design rationale.

## Run locally with Docker

```
docker compose up --build
```

Then open http://localhost:8000 in your browser.

The first build downloads the embedding model, generates the synthetic training set, and
fine-tunes the model as part of the image build (see "Fine-tuning" below) -- this takes real
time (15-20+ minutes on a typical machine) but is a one-time cost. After that, the app has no
runtime dependency on internet access.

## Run locally without Docker

```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Then open http://localhost:8000. Without Docker, `model_weights/` won't exist unless you run
the fine-tuning pipeline yourself first (see below), so this uses the pretrained base model and
downloads it from Hugging Face on first request.

## Fine-tuning

`training/generate_dataset.py` builds a synthetic dataset from a seed catalog of real Indonesian
retail products (instant noodles, staples, beverages, snacks, toiletries), tagged by category:
noisy supplier-style variants (`A_lexical`) and hand-written synonym/brand-vs-generic pairs
(`B_sinonim`, `D_brand_deskriptif`) as positives; same-brand flavor/size traps (`F_size_trap`,
`G_rasa_trap`) and cross-brand neighbor pairs (`H_brand_neighbor`) as hard negatives.
`training/train.py` fine-tunes with `TripletLoss` and saves to `model_weights/`, which
`backend/app/model.py` picks up automatically -- it also supports continued fine-tuning
(`--from-checkpoint`, `--categories`, `--lr`, `--epochs`) for training further on a subset of
categories from an already-proven checkpoint, see attempt 8 below for why. `training/evaluate.py`
checks a checkpoint against known hard cases, a held-out generalization canary, and
`training/data/eval_set.jsonl` (a second held-out product family, disjoint from training,
reporting lexical-vs-semantic scores per category) before it's trusted. See
`.personal-storage/PADAN_dataset_finetuning_spec.md` for the full evidence table and what it does
and doesn't prove about why AI is needed here.

The checkpoint isn't committed to git (it's a ~470MB binary, over GitHub's 100MB file limit) --
`docker compose up --build` regenerates it from source during the image build instead
(`generate_dataset.py` uses a fixed random seed, so it's reproducible), in a separate build stage
so the training-only dependencies don't bloat the final runtime image.

**Getting here took several rejected attempts, kept for the record:**

1. `MultipleNegativesRankingLoss` on positive pairs + hard-negative triplets (50% of variants got
   a hard negative). Improved abbreviation recognition but regressed same-brand-different-flavor
   separation badly enough to break real clustering cases the pretrained model gets right.
2. `TripletLoss` with every variant getting a hard negative. Fixed attempt 1's regression, but
   only because the hybrid matcher's lexical weighting happened to compensate -- tested against
   product categories outside the synthetic vocabulary (soap, detergent), the raw embedding was
   measurably worse than pretrained: classic catastrophic forgetting from a narrow, repetitive
   163-product training vocabulary with no general-domain regularization.
3. `TripletLoss` + `MultipleNegativesRankingLoss` together, an expanded/more diverse catalog, and
   a gentler learning rate. Fixed the generalization problem (verified via a held-out canary
   category never seen in training) but reintroduced attempt 1's flavor-merge failure -- MNRL's
   pull apparently dominates even with TripletLoss active in the same training run.
4. `TripletLoss`-only + the expanded catalog + gentler learning rate. Generalization held, but
   flavor separation was still under-fit. Tracing the actual cause: hard-negative siblings were
   selected at random, so ~84% of them confounded a flavor difference with a size difference at
   the same time, diluting the one signal that mattered (size differences are already caught
   deterministically by the matcher's quantity guard regardless of embedding quality, so the
   embedding specifically needs to learn flavor, not size).
5. Fixed sibling selection to prefer same-size negatives (isolating flavor-only differences) --
   this worked, but surfaced a narrower regression: a unit-reformatted variant (`85gr` ->
   `0.085kg`) stopped merging with its group. Root cause: `augment_unit_format` only ever
   reformatted spacing/casing within the same unit, never true cross-unit conversion, so the
   model was never shown that relationship as a positive pair and lost whatever latent capability
   pretrained had for it.
6. Added genuine gr<->kg and ml<->L conversion to the augmenter. This is the checkpoint that
   ships: passes the in-distribution test suite, the held-out generalization canary, and beats
   pretrained on the flavor/size hard negatives at the raw embedding level, not just after the
   hybrid matcher's lexical weighting compensates.
7. Expanded the dataset with hand-written synonym (`B_sinonim`) and brand-vs-generic
   (`D_brand_deskriptif`) pairs, plus auto-derived cross-brand hard negatives
   (`H_brand_neighbor`), and retrained from scratch on everything mixed together. Regressed
   attempt 6's calibration: the flagship heavy-abbreviation similarity dropped from 0.789 to
   0.665 and the plain-abbreviation case fell to 0.729 -- *below* the pretrained model's 0.790 --
   while flavor-trap separation weakened (0.251 -> 0.503, want low). Discrete clustering checks
   still passed (the hybrid matcher's lexical weight papered over it), but the raw embedding was
   measurably worse -- same failure shape as attempt 2. A large volume of very-lexically-close
   cross-brand negatives, trained in the same pass as everything else, diluted the calibration
   attempt 6 had achieved.
8. Tried continued fine-tuning instead of retraining from scratch: start from attempt 6's
   checkpoint (not the base model) and train further on only the new `B_sinonim`/
   `D_brand_deskriptif`/`H_brand_neighbor` rows (238, vs. 1530 for the full set). A gentle pass
   (`lr=2e-6`, 2 epochs) avoided regressing attempt 6's numbers, but barely moved the new
   categories either (`B_sinonim` 0.658 -> 0.659, `H_brand_neighbor` 0.888 -> 0.878 on
   `eval_set.jsonl` -- both within noise). A stronger pass (`lr=6e-6`, 5 epochs) reproduced
   attempt 7's regression (abbreviation 0.912 -> 0.736, again below pretrained), even though it
   only trained on the new categories and started from the proven checkpoint. Neither shipped --
   attempt 6's checkpoint is still what's in `model_weights/`. Conclusion: the flavor/abbreviation
   calibration and brand-neighbor discrimination appear to be in tension for this small a model
   and dataset regardless of learning rate; closing the `B`/`D`/`H` gap for real would need a
   different training strategy (e.g. two-stage training with a frozen sub-layer, or curriculum
   weighting within a single pass), not further LR/epoch tuning. Left as future work -- see
   `.personal-storage/PADAN_dataset_finetuning_spec.md` §5-6 for the full before/after table.

Also worth knowing if you touch this again: an early local `pip install` was unpinned and picked
up much newer package versions (`sentence-transformers` 6.0.0, not the `3.1.1` originally pinned)
than what the Docker build used, which silently diverged everything tested locally from what
actually got built -- and specifically broke training with `SentenceTransformerTrainer.compute_loss()
got an unexpected keyword argument 'num_items_in_batch'` (an `accelerate`/`transformers` version
mismatch). Both `requirements.txt` files are now pinned to exactly what's been tested.

## API

- `GET /health` — health check
- `POST /normalize` — body `{"names": ["Indomie Goreng 85gr", "Mie Goreng Indomi 85g"]}`,
  returns clustered product groups with a suggested canonical name and similarity score each.
