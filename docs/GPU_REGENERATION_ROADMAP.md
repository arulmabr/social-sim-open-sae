# GPU Regeneration Roadmap

This repo has complete Open-SAE reruns for creativity, safe-risk choice, ultimatum,
and trust-game saved outputs. This file records the finished GPU scope and the commands
needed to regenerate it.

## Completed

### Creativity

- Raw saved data: `data/raw/creativity/product_innovation_20251102_202650/`
- Processed Open-SAE output: `data/processed/creativity/open_sae_response_only_frequency/`
- Units: 320
- Top-k rows: 3,200
- GPU rerun command: `docs/REPRODUCTION.md`

### Safe-Risk Choice

- Raw saved data: `data/raw/games/safe_risky/results_20251018_205613/`
- Processed Open-SAE output: `data/processed/games/safe_risky/open_sae_calibration/`
- Units: 4,200
- Top-k rows: 42,000
- Behavior match against old saved summary: exact, max absolute difference 0.0

### Ultimatum Game

Raw saved data:

`data/raw/games/ultimatum/results_20251008_201139/`

Implemented loader behavior:

- Parse files named `ultimatum_<condition>_<offer>.csv`.
- Use `answer.ultimatum_response` as the answer.
- Use `comment.ultimatum_response_comment` as the explanatory text.
- Use `prompt.ultimatum_response_user_prompt` and `generated_tokens.ultimatum_response_generated_tokens`.
- Preserve offer as the numeric reward-like axis.
- Write behavior summary with accept/reject percentages.
- Accept old `feature_activations.txt` via `--goodfire-log` for label-overlap diagnostics.

Expected source scale from included data:

- 3 conditions: baseline, prompting, steering
- 17 offers: 10 through 90 by 5
- 40 agents per cell in the included selected run
- Expected response units: 2,040
- Expected top-k rows with `--top-k 10`: 20,400

Current source audit:

- Output: `data/processed/games/ultimatum/source_audit/`
- Response units: 2,040
- Behavior rows: 51
- Parsed old Goodfire rows: 20,260

Current Open-SAE output:

- Output: `data/processed/games/ultimatum/open_sae_full/`
- Response units: 2,040
- Top-k rows: 20,400
- Reward-condition cells: 51
- Special/control-token top-k hits: 0

### Trust Game

Raw saved data:

`data/raw/games/trust/results/`

Implemented loader behavior:

- Parse files named `trust_game_<condition>_sent_<amount>.csv`.
- Use `answer.trust_return` as the numeric return.
- Use `comment.trust_return_comment` as the explanatory text.
- Use `prompt.trust_return_user_prompt` and `generated_tokens.trust_return_generated_tokens`.
- Preserve sent amount as the numeric reward-like axis; behavior summary records tripled amount.
- Write behavior summary with average return and return ratio by condition and sent amount.

Expected source scale from included data:

- 2 conditions: baseline, intervention
- 10 sent amounts: 10 through 100 by 10
- 10 agents per cell
- Expected response units: 200
- Expected top-k rows with `--top-k 10`: 2,000

Current source audit:

- Output: `data/processed/games/trust/source_audit/`
- Response units: 200
- Behavior rows: 20

Current Open-SAE output:

- Output: `data/processed/games/trust/open_sae_full/`
- Response units: 200
- Top-k rows: 2,000
- Sent-amount-condition cells: 20
- Special/control-token top-k hits: 0

## GPU Target

Use the same setup as the completed runs:

- 1x H100 80GB
- `--load-in-4bit`
- `meta-llama/Llama-3.3-70B-Instruct`
- `Goodfire/Llama-3.3-70B-Instruct-SAE-l50`
- `model.layers.50`

Run local dry-runs before any GPU execution. The RunPod helper reruns ultimatum and
trust smoke tests plus full jobs:

```bash
bash ./runpod/run_remaining_games_open_sae.sh
```
