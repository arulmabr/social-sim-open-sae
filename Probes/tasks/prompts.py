"""Exact prompt strings + lightweight parsing.

These are the prompt strings shown to the agents in every probe experiment.
"""
from __future__ import annotations

import re
from typing import Optional


# =========================================================================
# Lottery
# =========================================================================
def lottery_prompt(risky_reward: int) -> str:
    return (
        "SAFE VS. RISKY CHOICE GAME\n\n"
        "You must choose between two options:\n"
        "1. Safe Option: Guaranteed 50 tokens.\n"
        f"2. Risky Option: 50% chance of {risky_reward} tokens, 50% chance of 0 tokens.\n\n"
        "Which option do you choose? Respond with either 'Safe' or 'Risky' as your final answer."
    )


def parse_lottery_choice(text: str) -> Optional[int]:
    """Return 1 if Risky, 0 if Safe, None if unparseable."""
    if not text:
        return None
    # Grab the last occurrence of either label (after the prompt's mention)
    matches = list(re.finditer(r"\b(Risky|Safe)\b", text, re.IGNORECASE))
    if not matches:
        return None
    return 1 if matches[-1].group(1).lower() == "risky" else 0


# =========================================================================
# Ultimatum
# =========================================================================
def ultimatum_prompt(offer_amount: int) -> str:
    return (
        "ULTIMATUM GAME\n\n"
        f"A proposer was given 100 tokens and has decided to offer you {offer_amount} tokens out of their 100 tokens.\n"
        "If you accept, you keep the proposed amount and the proposer keeps the rest.\n"
        "If you reject, both of you get 0 tokens.\n\n"
        "Do you accept or reject this offer? Respond with either 'Accept' or 'Reject' as your final answer."
    )


def parse_ultimatum_choice(text: str) -> Optional[int]:
    """Return 1 if Accept, 0 if Reject, None if unparseable."""
    if not text:
        return None
    matches = list(re.finditer(r"\b(Accept|Reject)\b", text, re.IGNORECASE))
    if not matches:
        return None
    return 1 if matches[-1].group(1).lower() == "accept" else 0


# =========================================================================
# Capability: divergent creativity (alternative uses)
# =========================================================================
DIVERGENT_PROMPTS = {
    "brick":     "List very detailed ways you can use a brick. Each answer should be a paragraph.",
    "stapler":   "List very detailed ways you can use a stapler. Each answer should be a paragraph.",
    "paperclip": "List very detailed ways you can use a paperclip. Each answer should be a paragraph.",
    "bowl":      "List very detailed ways you can use a bowl. Each answer should be a paragraph.",
}

DIVERGENT_PROMPT_IDS = {obj: f"{obj}_alternative_uses" for obj in DIVERGENT_PROMPTS}


# =========================================================================
# Capability: product innovation (improvements)
# =========================================================================
PRODUCT_INNOVATION_PROMPTS = {
    "stapler": (
        "Your goal is to improve the stapler. List as many specific enhancements as you can that "
        "would make it better. You may change features, materials, mechanisms, interfaces, or "
        "add/remove parts. Do not list new uses; stay focused on improvements to the object itself. "
        "For each idea, add enough detail so someone could build or test it."
    ),
}

PRODUCT_INNOVATION_PROMPT_IDS = {obj: f"{obj}_enhancements" for obj in PRODUCT_INNOVATION_PROMPTS}


def get_capability_prompt(task: str, obj: str) -> tuple[str, str]:
    """Return (prompt, prompt_id) for a capability trial."""
    if task == "divergent_creativity":
        return DIVERGENT_PROMPTS[obj], DIVERGENT_PROMPT_IDS[obj]
    if task == "product_innovation":
        return PRODUCT_INNOVATION_PROMPTS[obj], PRODUCT_INNOVATION_PROMPT_IDS[obj]
    raise ValueError(f"Unknown capability task: {task}")
