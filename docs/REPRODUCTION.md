# Reproduction Guide

## Local Non-GPU Checks

```bash
python scripts/run_open_sae_feature_inspection.py \
  --dataset-kind creativity \
  --dry-run \
  --expected-units 320

python scripts/run_open_sae_feature_inspection.py \
  --dataset-kind safe_risky \
  --dry-run \
  --expected-units 4200

python tests/verify_release_artifacts.py
```

## GPT-5 Torrance Judge Rerun

Requires `OPENAI_API_KEY`.

```bash
python scripts/rerun_creativity_torrance_eval.py \
  --source-dir data/raw/creativity/product_innovation_20251102_202650 \
  --output-dir data/processed/creativity/torrance_gpt5_eval_rerun \
  --requested-model gpt-5 \
  --overwrite
```

## Creativity Open-SAE Rerun

Requires a GPU with enough memory for Llama 3.3 70B in 4-bit mode and a Hugging Face
token with access to Meta Llama 3.3 70B.

```bash
python scripts/run_open_sae_feature_inspection.py \
  --dataset-kind creativity \
  --activation-scope assistant_response \
  --feature-aggregation frequency \
  --activation-threshold 0.1 \
  --source-dir data/raw/creativity/product_innovation_20251102_202650 \
  --output-dir data/processed/creativity/open_sae_response_only_frequency_rerun \
  --model-id meta-llama/Llama-3.3-70B-Instruct \
  --sae-repo Goodfire/Llama-3.3-70B-Instruct-SAE-l50 \
  --hook model.layers.50 \
  --top-k 10 \
  --expected-units 320 \
  --load-in-4bit
```

## Safe-Risk Open-SAE Rerun

```bash
python scripts/run_open_sae_feature_inspection.py \
  --dataset-kind safe_risky \
  --activation-scope all_content \
  --source-dir data/raw/games/safe_risky/results_20251018_205613 \
  --output-dir data/processed/games/safe_risky/open_sae_calibration_rerun \
  --model-id meta-llama/Llama-3.3-70B-Instruct \
  --sae-repo Goodfire/Llama-3.3-70B-Instruct-SAE-l50 \
  --hook model.layers.50 \
  --top-k 10 \
  --expected-units 4200 \
  --goodfire-log data/raw/games/safe_risky/results_20251018_205613/feature_activations.txt \
  --load-in-4bit
```

## Pending Game Regeneration

The raw data for ultimatum and trust-game is included. The next implementation step is
to extend `scripts/run_open_sae_feature_inspection.py` with loaders for:

- `dataset-kind ultimatum`
- `dataset-kind trust`

After those loaders exist, use the same Open-SAE model/SAE/hook setup above.
