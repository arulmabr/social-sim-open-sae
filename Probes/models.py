"""Model loading + activation capture + steering hook installation.

Wraps HuggingFace Transformers. Exposes a `Wrapped` object with:
- `.generate(prompt, ..., lambda_value=0.0, direction=None, sign=+1)` -> text
- `.capture_activations(prompt, layer)` -> last-token hidden state at the layer
- `.train_mode_off()` (always; we never fine-tune the base model)
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config


@dataclass
class GenerationResult:
    text: str
    prompt_token_ids: torch.Tensor
    generated_token_ids: torch.Tensor


class Wrapped:
    """Wrapper that owns model + tokenizer and installs steering hooks."""

    def __init__(self, model_cfg: config.ModelConfig, device: str = "cuda", dtype: str = "bf16"):
        self.cfg = model_cfg
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.hf_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        self._steering_handle = None

    # -------------------------------------------------------------------
    # Activation capture (no steering, deterministic forward pass)
    # -------------------------------------------------------------------
    @torch.inference_mode()
    def capture_activations(self, prompt: str, layer_idx: int) -> np.ndarray:
        """Return the final-token hidden state at the given layer (1D numpy array)."""
        layer = self.model.model.layers[layer_idx]
        acts: list[torch.Tensor] = []

        def hook(_module, _inp, output):
            hidden = output[0] if isinstance(output, tuple) else output
            acts.append(hidden.detach())
            return output

        handle = layer.register_forward_hook(hook)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            _ = self.model(**inputs, use_cache=config.USE_KV_CACHE_FOR_ACTIVATIONS)
        finally:
            handle.remove()
        if not acts:
            raise RuntimeError(f"No activations captured at layer {layer_idx}")
        last_token = acts[0][0, -1, :].float().cpu().numpy()
        return last_token

    # -------------------------------------------------------------------
    # Steering hook installation
    # -------------------------------------------------------------------
    def install_steering(
        self,
        layer_idx: int,
        direction: np.ndarray,
        lambda_value: float,
        sign: int = +1,
    ) -> None:
        """Install the probe-steering hook at `layer_idx`.

        Intervene on the last 20% of tokens with a linearly-increasing
        position scale (0.5 -> 1.0).
        """
        self.remove_steering()
        if abs(lambda_value) < 1e-8:
            return  # no-op

        v = torch.as_tensor(direction, dtype=torch.float32, device=self.model.device)
        v_unit = v / (v.norm() + 1e-12)
        # Scale to match typical activation magnitudes at this layer.
        # We pick a fixed scaling constant; in practice this is set by the
        # caller via lambda calibration so the exact scale doesn't matter.
        v_scaled = v_unit
        sign_f = float(np.sign(sign) or 1)
        last_frac = config.STEERING["last_fraction"]
        lo = config.STEERING["position_scale_lo"]
        hi = config.STEERING["position_scale_hi"]

        layer = self.model.model.layers[layer_idx]

        def hook(_module, _inp, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            seq_len = hidden.shape[1]
            if seq_len == 0:
                return output
            start_idx = max(1, seq_len - max(1, int(math.ceil(seq_len * last_frac))))
            n = seq_len - start_idx
            if n <= 0:
                return output
            scale = torch.linspace(lo, hi, n, device=hidden.device, dtype=hidden.dtype)
            delta = (sign_f * lambda_value) * v_scaled.to(hidden.dtype)
            hidden[:, start_idx:, :] = hidden[:, start_idx:, :] + delta.view(1, 1, -1) * scale.view(1, -1, 1)
            if is_tuple:
                return (hidden,) + output[1:]
            return hidden

        self._steering_handle = layer.register_forward_hook(hook)

    def remove_steering(self) -> None:
        if self._steering_handle is not None:
            self._steering_handle.remove()
            self._steering_handle = None

    @contextmanager
    def steering_at(
        self, layer_idx: int, direction: np.ndarray, lambda_value: float, sign: int = +1
    ) -> Iterator[None]:
        self.install_steering(layer_idx, direction, lambda_value, sign=sign)
        try:
            yield
        finally:
            self.remove_steering()

    # -------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        seed: Optional[int] = None,
    ) -> GenerationResult:
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=max(temperature, 1e-5),
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        prompt_len = inputs.input_ids.shape[1]
        generated_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            prompt_token_ids=inputs.input_ids[0],
            generated_token_ids=generated_ids,
        )


def load(model_key: str, device: str = "cuda", dtype: str = "bf16") -> Wrapped:
    cfg = config.MODELS[model_key]
    return Wrapped(cfg, device=device, dtype=dtype)
