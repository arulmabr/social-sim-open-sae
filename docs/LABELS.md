# Feature Label Caveat

The old project used Goodfire's hosted Ember API for feature inspection and returned
hosted natural-language feature labels.

For this release, Goodfire's old hosted endpoint is unavailable. The open-source
replacement uses:

- `meta-llama/Llama-3.3-70B-Instruct`
- `Goodfire/Llama-3.3-70B-Instruct-SAE-l50`
- `model.layers.50`

The SAE weights provide feature indices and decoder/encoder parameters. They do not
include the original hosted Ember label database. We therefore fetch/cache
Neuronpedia/Open-SAE labels when available and fall back to feature indices otherwise.

Recommended language:

> Behavior and saved-response data are reproduced exactly where stated. Feature
> activations are reproduced from the open Goodfire SAE. Natural-language labels are
> replacement interpretability labels and may differ from historical Goodfire Ember
> labels.
