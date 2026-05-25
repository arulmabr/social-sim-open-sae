# Open-SAE Ablation Analysis

This report compares saved Open-SAE output folders without re-running the model.
The main failure mode under test is max-pooling over chat-template/control tokens,
which appears as rank-1 activations collapsing to one feature and one token.

## Summary

### legacy_unmasked

- Path: `creativity_experiment/product_innovation_20251102_202650_goodfire_open_sae_40agent_features`
- Status: `ok`
- Dataset: `None`
- Activation scope: `None`
- Feature aggregation: `None`
- Activation threshold: `None`
- Units: `320` / expected `320`
- Top-k rows: `3200` / expected `3200`
- Special/control top-k hits: `None`
- Rank-1 std: `0.0`
- Rank-1 max-activation std: `None`
- Rank-1 unique features: `1`
- Rank-1 unique tokens: `1`
- Rank-1 mode token: `<|begin_of_text|>`
- Artifact flag: `True`
- Artifact reason: `rank1 mode token is a chat-template/control token`

### response_only_max

- Path: `creativity_experiment/product_innovation_20251102_202650_goodfire_open_sae_40agent_features_response_only`
- Status: `ok`
- Dataset: `creativity`
- Activation scope: `assistant_response`
- Feature aggregation: `max_legacy`
- Activation threshold: `None`
- Units: `320` / expected `320`
- Top-k rows: `3200` / expected `3200`
- Special/control top-k hits: `0`
- Rank-1 std: `1.1620967437523149`
- Rank-1 max-activation std: `None`
- Rank-1 unique features: `38`
- Rank-1 unique tokens: `41`
- Rank-1 mode token: ` order`
- Artifact flag: `False`
- Artifact reason: ``

### response_only_frequency

- Path: `creativity_experiment/product_innovation_20251102_202650_goodfire_open_sae_40agent_features_response_only_frequency`
- Status: `ok`
- Dataset: `creativity`
- Activation scope: `assistant_response`
- Feature aggregation: `frequency`
- Activation threshold: `0.1`
- Units: `320` / expected `320`
- Top-k rows: `3200` / expected `3200`
- Special/control top-k hits: `0`
- Rank-1 std: `90.81287298443168`
- Rank-1 max-activation std: `2.879778208185355`
- Rank-1 unique features: `10`
- Rank-1 unique tokens: `33`
- Rank-1 mode token: ` a`
- Artifact flag: `False`
- Artifact reason: ``

### all_content_frequency

- Path: `creativity_experiment/product_innovation_20251102_202650_goodfire_open_sae_40agent_features_all_content_noncontrol`
- Status: `ok`
- Dataset: `creativity`
- Activation scope: `all_content`
- Feature aggregation: `frequency`
- Activation threshold: `0.1`
- Units: `320` / expected `320`
- Top-k rows: `3200` / expected `3200`
- Special/control top-k hits: `0`
- Rank-1 std: `83.55092584784053`
- Rank-1 max-activation std: `2.686378145961514`
- Rank-1 unique features: `5`
- Rank-1 unique tokens: `27`
- Rank-1 mode token: ` a`
- Artifact flag: `False`
- Artifact reason: ``

### user_prompt_frequency

- Path: `creativity_experiment/product_innovation_20251102_202650_goodfire_open_sae_40agent_features_user_prompt_noncontrol`
- Status: `ok`
- Dataset: `creativity`
- Activation scope: `user_prompt`
- Feature aggregation: `frequency`
- Activation threshold: `0.1`
- Units: `320` / expected `320`
- Top-k rows: `3200` / expected `3200`
- Special/control top-k hits: `0`
- Rank-1 std: `4.5014757245885`
- Rank-1 max-activation std: `0.09392178381618392`
- Rank-1 unique features: `2`
- Rank-1 unique tokens: `3`
- Rank-1 mode token: ` could`
- Artifact flag: `False`
- Artifact reason: ``

### safe_risky_calibration

- Path: `mech_interp_games/safe_risky_choice/results_20251018_205613_goodfire_open_sae_calibration`
- Status: `ok`
- Dataset: `safe_risky`
- Activation scope: `all_content`
- Feature aggregation: `max_legacy`
- Activation threshold: `None`
- Units: `4200` / expected `4200`
- Top-k rows: `42000` / expected `42000`
- Special/control top-k hits: `0`
- Rank-1 std: `0.7892810638741711`
- Rank-1 max-activation std: `None`
- Rank-1 unique features: `38`
- Rank-1 unique tokens: `44`
- Rank-1 mode token: ` resulting`
- Artifact flag: `False`
- Artifact reason: ``

## Interpretation

A valid response-level decomposition should have zero special/control-token top-k hits
and non-trivial variation in rank-1 feature and token identities. A collapsed rank-1
distribution is evidence that the pooling scope is dominated by a template token rather
than task content or generated response content.

## Completed Matrix

All supplied outputs were read from disk and summarized here. Treat any row with
`artifact_flag=True` as a diagnostic/failure control, not as a paper-style feature
decomposition.
