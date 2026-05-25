"""Multi-judge creativity scoring.

Adapters for five judge models across four providers:
- GPT-5                 (OpenAI)
- Claude Sonnet 4.6     (Anthropic)
- Gemini 3.1 Pro        (Google)
- Kimi K2.5             (Together AI, OpenAI-compatible endpoint)
- DeepSeek R1           (Together AI, OpenAI-compatible endpoint)

Each adapter implements `.call(prompt, temperature) -> str`. The shared
`score_response_with_judge(text, task, judge_name)` routes to the right
adapter, parses the JSON reply into the 4 Torrance sub-dimensions, and
returns `{fluency, flexibility, originality, elaboration, creativity_score}`
with `creativity_score = mean of the four`.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional


# =========================================================================
# API key resolution
# =========================================================================
DEFAULT_KEY_FILES = {
    "openai": Path("/Users/Shreyas2/Desktop/Berkeley/occupation_task/api_key_openai3.txt"),
    "anthropic": Path("/Users/Shreyas2/Desktop/Berkeley/occupation_task/api_key_anthropic.txt"),
    "google": Path("/Users/Shreyas2/Desktop/Berkeley/occupation_task/api_key_gemini_final.txt"),
    "together": Path("/Users/Shreyas2/Desktop/Berkeley/occupation_task/api_key_together.txt"),
}

ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "together": "TOGETHER_API_KEY",
}


def load_api_key(provider: str) -> str:
    """Resolve API key for a provider. Env var wins; file path is fallback."""
    env_name = ENV_VARS.get(provider)
    if env_name and os.environ.get(env_name):
        return os.environ[env_name].strip()
    path = DEFAULT_KEY_FILES.get(provider)
    if path and path.exists():
        return path.read_text().strip()
    raise RuntimeError(
        f"No API key for provider '{provider}'. "
        f"Set env var {env_name!r} or create file {path!s}."
    )


# =========================================================================
# Rubric prompt
# =========================================================================
RUBRIC_TEMPLATE = """Rate the creativity of the following task and its response on a scale of 1-10, where 1 is "Not creative (generic, obvious, conventional ideas)" and 10 is "Very creative (highly original, novel, unconventional, diverse ideas)".

The task is: {task_description}

Factors to consider:
1. Fluency: The ability to produce a significant number of relevant ideas in response to a given question.
2. Flexibility: The variety of categories from which one can generate ideas.
3. Originality: The uniqueness of the ideas generated.
4. Elaboration: The ability to expand upon, refine, and embellish an idea.

Response to evaluate:
{response_text}

Return ONLY a JSON object with four integer fields:
{{"fluency": X, "flexibility": X, "originality": X, "elaboration": X}}
Each X is an integer 1 through 10."""

LENGTH_CONTROLLED_SUFFIX = (
    "\n\nIMPORTANT: Score on idea quality, not response length. A response with "
    "fewer but more original or detailed ideas should not be penalized for brevity."
)

TASK_DESCRIPTIONS = {
    "divergent_creativity": "Listing detailed alternative uses for an everyday object.",
    "product_innovation": "Listing specific enhancements that would improve a given product.",
}


# =========================================================================
# Provider adapters
# =========================================================================
@dataclass
class JudgeAdapter:
    """One concrete judge model behind one provider's API."""
    judge_name: str            # short id used in JSON output (e.g. "gpt-5")
    display_name: str          # canonical name for the `judge` field
    provider: str
    model_id: str              # the provider's model identifier

    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 200) -> str:
        raise NotImplementedError


class OpenAIAdapter(JudgeAdapter):
    """GPT-5 endpoint requires `max_completion_tokens` (not `max_tokens`) and
    leaves room for internal reasoning tokens, so we default to a generous
    budget. Temperature is fixed at 1.0 because GPT-5 ignores user temperature
    and warns otherwise."""

    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2000) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=load_api_key("openai"))
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are a strict, calibrated creativity judge."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class AnthropicAdapter(JudgeAdapter):
    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 200) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=load_api_key("anthropic"))
        resp = client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            temperature=temperature,
            system="You are a strict, calibrated creativity judge.",
            messages=[{"role": "user", "content": prompt}],
        )
        # Concatenate text blocks
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)


class GeminiAdapter(JudgeAdapter):
    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8000) -> str:
        # Gemini 2.5/3.x Pro are thinking models: the thinking trace counts
        # against max_output_tokens, so the budget is generous to leave room
        # for the JSON answer after reasoning.
        # `google-genai` SDK (newer); falls back to `google-generativeai` if needed.
        try:
            from google import genai
            client = genai.Client(api_key=load_api_key("google"))
            resp = client.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "system_instruction": "You are a strict, calibrated creativity judge.",
                },
            )
            return resp.text or ""
        except ImportError:
            import google.generativeai as genai_old
            genai_old.configure(api_key=load_api_key("google"))
            model = genai_old.GenerativeModel(
                self.model_id,
                system_instruction="You are a strict, calibrated creativity judge.",
            )
            resp = model.generate_content(
                prompt,
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
            )
            return resp.text or ""


class TogetherAdapter(JudgeAdapter):
    """Together AI uses an OpenAI-compatible API at base_url=https://api.together.xyz/v1.

    Used here for Kimi K2.5 (moonshotai/Kimi-K2-Instruct) and DeepSeek-R1.
    """
    def call(self, prompt: str, temperature: float = 0.0, max_tokens: int = 8000) -> str:
        # Kimi-K2.6 and DeepSeek-V4-Pro are reasoning models: they burn a large
        # hidden reasoning trace (Kimi can exceed 2000 tokens) before emitting
        # the JSON answer. With too small a budget the response hits
        # finish_reason="length" with empty content, so the budget is generous.
        from openai import OpenAI
        client = OpenAI(
            api_key=load_api_key("together"),
            base_url="https://api.together.xyz/v1",
        )
        resp = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": "You are a strict, calibrated creativity judge."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


# =========================================================================
# Registry
# =========================================================================
JUDGES: Dict[str, JudgeAdapter] = {
    "gpt-5": OpenAIAdapter(
        judge_name="gpt-5",
        display_name="GPT-5",
        provider="openai",
        model_id="gpt-5",
    ),
    "claude-sonnet-4-6": AnthropicAdapter(
        judge_name="claude-sonnet-4-6",
        display_name="Claude Sonnet 4.6",
        provider="anthropic",
        model_id="claude-sonnet-4-6",
    ),
    "gemini-3.1-pro": GeminiAdapter(
        judge_name="gemini-3.1-pro",
        display_name="Gemini 3.1 Pro (Preview)",
        provider="google",
        model_id="gemini-3.1-pro-preview",
    ),
    # GA substitute for the Gemini judge: gemini-3.1-pro-preview is hard-capped
    # at 250 requests/day per project even on paid Tier 1 (preview-model limit),
    # so it cannot cover the full response set. gemini-2.5-pro is the GA flagship
    # with far higher Tier-1 daily limits, used to score the complete pool.
    "gemini-2.5-pro": GeminiAdapter(
        judge_name="gemini-2.5-pro",
        display_name="Gemini 2.5 Pro",
        provider="google",
        model_id="gemini-2.5-pro",
    ),
    # Substituted Kimi-K2.5 -> Kimi-K2.6 (Together listed K2.6 as the current
    # serverless Kimi; K2-Instruct returned 503; K2.5 not in their catalog).
    "kimi-k2.6": TogetherAdapter(
        judge_name="kimi-k2.6",
        display_name="Kimi K2.6",
        provider="together",
        model_id="moonshotai/Kimi-K2.6",
    ),
    # Substituted DeepSeek-R1 -> DeepSeek-V4-Pro because R1 (and the Llama
    # distill of it) are not serverless on Together; V4-Pro keeps the same
    # provider family (DeepSeek) and is the closest current serverless flagship.
    "deepseek-v4-pro": TogetherAdapter(
        judge_name="deepseek-v4-pro",
        display_name="DeepSeek V4 Pro",
        provider="together",
        model_id="deepseek-ai/DeepSeek-V4-Pro",
    ),
}

DEFAULT_PRIMARY_JUDGE = "gpt-5"


# =========================================================================
# Score parsing
# =========================================================================
def _parse_scores(text: str) -> Optional[Dict[str, int]]:
    """Pull the 4-field JSON out of a judge reply, with light cleanup."""
    if not text:
        return None
    text = text.strip()
    # Some judges wrap in code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", text, re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    required = {"fluency", "flexibility", "originality", "elaboration"}
    if not required.issubset(data.keys()):
        return None
    out: Dict[str, int] = {}
    for k in required:
        try:
            v = int(round(float(data[k])))
        except (ValueError, TypeError):
            return None
        out[k] = max(1, min(10, v))
    return out


def _build_prompt(response_text: str, task: str, length_controlled: bool) -> str:
    p = RUBRIC_TEMPLATE.format(
        task_description=TASK_DESCRIPTIONS[task],
        response_text=response_text,
    )
    if length_controlled:
        p = p + LENGTH_CONTROLLED_SUFFIX
    return p


# =========================================================================
# Public scoring API
# =========================================================================
def score_response_with_judge(
    response_text: str,
    task: str,
    judge_name: str = DEFAULT_PRIMARY_JUDGE,
    n_retries: int = 3,
    retry_backoff_sec: float = 2.0,
    length_controlled: bool = False,
) -> Optional[Dict[str, float]]:
    """Score a single response with one judge."""
    if judge_name not in JUDGES:
        raise KeyError(f"Unknown judge {judge_name!r}. Known: {list(JUDGES.keys())}")
    judge = JUDGES[judge_name]
    prompt = _build_prompt(response_text, task, length_controlled)
    last_err: Optional[Exception] = None
    for attempt in range(n_retries):
        try:
            text = judge.call(prompt, temperature=0.0)
            parsed = _parse_scores(text)
            if parsed is not None:
                mean = sum(parsed.values()) / 4.0
                return {**parsed, "creativity_score": mean, "judge": judge.display_name}
        except Exception as e:
            last_err = e
        time.sleep(retry_backoff_sec * (attempt + 1))
    return None


def score_response_multi(
    response_text: str,
    task: str,
    judge_names: List[str],
    length_controlled: bool = False,
) -> Dict[str, Optional[Dict[str, float]]]:
    """Score one response with every judge in `judge_names`.

    Returns {judge_name: scored_dict_or_None}.
    """
    return {
        jn: score_response_with_judge(
            response_text, task, judge_name=jn, length_controlled=length_controlled
        )
        for jn in judge_names
    }


# =========================================================================
# Back-compat shim for callers that use `score_response`
# =========================================================================
def score_response(
    response_text: str,
    task: str,
    judge_name: str = DEFAULT_PRIMARY_JUDGE,
    n_retries: int = 3,
) -> Optional[Dict[str, float]]:
    """Drop-in replacement for the original single-judge API."""
    return score_response_with_judge(response_text, task, judge_name, n_retries=n_retries)
