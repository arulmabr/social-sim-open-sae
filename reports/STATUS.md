# Current Replication Status

Date: 2026-05-23

## Complete

- Creativity GPT-5 Torrance judging over 320 saved response-task rows.
- Creativity Open-SAE response-only frequency decomposition over 320 response-task rows.
- Safe-risk Open-SAE calibration over 4,200 saved responses.
- Safe-risk behavior exactly matches the old saved behavior summary.

## Caveats

- Goodfire Ember hosted natural-language labels are unavailable.
- Natural-language labels in processed Open-SAE outputs are Neuronpedia/Open-SAE
  replacement labels.
- The old unmasked creativity Open-SAE output collapsed to `<|begin_of_text|>` and is
  retained only as a failure control in the ablation report.

## Pending

- Add Open-SAE loaders for ultimatum saved outputs.
- Add Open-SAE loaders for trust-game saved outputs.
- Run GPU feature decompositions for ultimatum and trust-game.
- Produce final paper-style figures for all games from the open pipeline.
